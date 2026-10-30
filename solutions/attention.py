"""Scaled dot-product attention and multi-head attention.

Attention is a weighted average of value vectors. The weights come from
comparing a query to every key. Nothing here is recurrent and nothing here
is an MLP: it is two matrix multiplies and a softmax.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V

Shapes used throughout, which is the whole trick:

    batch      how many sequences are processed together
    heads      independent attention computations, concatenated at the end
    time       position in the sequence (also called the context length)
    head_dim   width of one head. n_embd == heads * head_dim
    channels   n_embd, the width of the residual stream

A causal language model may look at the current token and the past, never
the future. The mask sets those illegal scores to -inf before the softmax,
so their weights become 0.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def causal_mask(time: int, device: torch.device) -> torch.Tensor:
    """Boolean mask of shape (time, time). True means that score is kept.

    Row t is the token being predicted. Column s is the token it may read.
    tril keeps the diagonal and everything below it, so token t can read
    tokens 0 .. t and nothing later.
    """
    return torch.ones(time, time, device=device, dtype=torch.bool).tril()


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    attn_dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention for one already-split stack of heads.

    query, key, value: (batch, heads, time, head_dim)
    mask:              (time, time) bool, True where attention is allowed

    Returns the mixed values and the weights that mixed them.
    output:  (batch, heads, time, head_dim)
    weights: (batch, heads, time, time). Rows sum to 1 before dropout.
    """
    head_dim = query.size(-1)
    # (batch, heads, time, head_dim) @ (batch, heads, head_dim, time)
    # -> (batch, heads, time, time)
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    if attn_dropout is not None:
        weights = attn_dropout(weights)
    output = weights @ value
    return output, weights


class MultiHeadAttention(nn.Module):
    """Run several attention heads and mix them back to n_embd channels.

    One linear layer produces Q, K, and V together. Splitting the channel
    axis into heads is a view, not a second learned map: each head sees a
    slice of the channels. The output projection is what lets heads share
    information after they have attended separately.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        qkv = self.qkv(x)
        query, key, value = qkv.split(channels, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # (batch, time, heads, head_dim) -> (batch, heads, time, head_dim)
            return t.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)

        query, key, value = split_heads(query), split_heads(key), split_heads(value)
        mask = causal_mask(time, x.device)
        # Dropout hits the attention weights, then the output projection.
        # In eval mode both Dropout modules return their input unchanged.
        mixed, _ = scaled_dot_product_attention(
            query, key, value, mask, attn_dropout=self.attn_drop
        )
        # transpose gives a non-contiguous layout; contiguous() copies so view can run
        mixed = mixed.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_drop(self.out(mixed))


def _demo() -> None:
    torch.manual_seed(0)
    time, head_dim = 4, 2
    query = torch.randn(1, 1, time, head_dim)
    key = torch.randn(1, 1, time, head_dim)
    # Distinct values so a mix is visible. Column 0 holds the position index.
    value = torch.zeros(1, 1, time, head_dim)
    value[0, 0, :, 0] = torch.arange(time, dtype=torch.float)

    mask = causal_mask(time, query.device)
    output, weights = scaled_dot_product_attention(query, key, value, mask)
    w = weights[0, 0]
    print("causal mask (1 = allowed)")
    print(mask.int())
    print("attention weights, rows sum to 1, future columns are 0")
    print(w.round(decimals=3))
    print("output shape", tuple(output.shape))
    future = torch.triu(torch.ones(time, time, dtype=torch.bool), diagonal=1)
    assert torch.allclose(w.sum(dim=-1), torch.ones(time), atol=1e-5)
    assert torch.equal(w[future], torch.zeros(int(future.sum())))
    # Row 0 can only see position 0, so its weight on itself is 1 and the
    # output equals that position's value.
    assert torch.allclose(w[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(output[0, 0, 0], value[0, 0, 0])

    block = MultiHeadAttention(n_embd=16, n_head=4, dropout=0.0)
    x = torch.randn(2, time, 16)
    y = block(x)
    assert y.shape == x.shape
    print("multi-head forward", tuple(x.shape), "->", tuple(y.shape))


if __name__ == "__main__":
    _demo()
