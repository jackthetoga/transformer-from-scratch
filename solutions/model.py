"""A decoder-only transformer language model.

This is the GPT shape of a transformer, not the encoder-decoder in the
original "Attention Is All You Need" paper. A decoder-only model reads a
prefix and predicts the next token. That is the training task: at every
position, predict the token that comes next.

Data flow, with shapes:

    token ids (batch, time)
      -> token embedding + position embedding     (batch, time, n_embd)
      -> N blocks, each:
           x = x + Attention(LayerNorm(x))
           x = x + MLP(LayerNorm(x))
      -> final LayerNorm
      -> linear map to vocab logits               (batch, time, vocab)

The MLP inside a block is the same kind of network you already know. The
new pieces around it are attention, the residual add, and LayerNorm.

Pre-norm (normalize, then the sublayer, then add back to x) is the easier
arrangement to train. The 2017 paper used post-norm. Residuals matter
because each block can start near the identity: if Attention(x) and MLP(x)
are small, the block passes x through and deep stacks still train.

Position embeddings are learned vectors, one per index 0 .. block_size-1.
The original paper used fixed sinusoids. Recent models use rotary
embeddings. Learned absolute positions are the right first version: the
model needs some way to tell position 3 from position 4, because attention
by itself has no order.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention


class MLP(nn.Module):
    """The feed-forward network inside one transformer block.

    Width expands by 4 and comes back. That 4x ratio is the usual choice:
    attention moves information between positions, and this MLP processes
    each position on its own (the same weights at every time step).
    """

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return self.drop(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        block_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        # The head maps hidden states back to one logit per vocabulary item.
        # Tying its weight to the token embedding means "how a token is read"
        # and "how a token is predicted" share one matrix.
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, time = idx.shape
        if time > self.block_size:
            raise ValueError(
                f"sequence length {time} is longer than block_size {self.block_size}"
            )
        positions = torch.arange(time, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(positions)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            # Flatten batch and time into one big classification problem.
            # ignore_index=-100 drops positions we do not want to grade
            # (used by the reverse-string task for the prompt tokens).
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Append tokens one at a time.

        temperature divides the logits before softmax.
          temperature < 1  makes the distribution sharper (more of the top token)
          temperature > 1  makes it flatter
          temperature == 0 uses argmax, so the result does not change between calls
        top_k throws away every logit except the k largest before the softmax.
        """
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size :]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]
                if temperature == 0:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        k = min(top_k, logits.size(-1))
                        cutoff = torch.topk(logits, k).values[:, -1]
                        logits = logits.masked_fill(
                            logits < cutoff[:, None], float("-inf")
                        )
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_id], dim=1)
            return idx
        finally:
            self.train(was_training)


def count_parameters(model: nn.Module) -> int:
    # Shared (tied) parameters are stored once, and parameters() yields them once.
    return sum(p.numel() for p in model.parameters())
