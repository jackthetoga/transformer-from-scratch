"""Check the building blocks against PyTorch and against the causal property.

    .venv/bin/python transformer-from-scratch/checks.py

No training corpus required. A short reverse-task run at the end only checks
that the loss moves down; train.py is what you use to actually fit a model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from attention import causal_mask, scaled_dot_product_attention
from data import REVERSE_VOCAB_SIZE, get_reverse_batch
from model import TransformerLM, count_parameters
from pytorch_basics import cross_entropy_from_logits, stable_softmax


def check_softmax_and_loss() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.0, 0.0, 0.5], [-3.0, 4.0, 1.0]])
    targets = torch.tensor([0, 2, 1])
    assert torch.allclose(stable_softmax(logits), torch.softmax(logits, dim=-1))
    ours = cross_entropy_from_logits(logits, targets)
    theirs = F.cross_entropy(logits, targets)
    assert torch.allclose(ours, theirs), (ours, theirs)

    z = logits[0].detach().clone().requires_grad_(True)
    F.cross_entropy(z[None, :], targets[:1]).backward()
    expected = torch.softmax(z, dim=-1).detach()
    expected[0] -= 1
    assert torch.allclose(z.grad, expected)


def check_causal_weights() -> None:
    time, head_dim = 5, 4
    query = torch.randn(2, 3, time, head_dim)
    key = torch.randn(2, 3, time, head_dim)
    value = torch.randn(2, 3, time, head_dim)
    _, weights = scaled_dot_product_attention(
        query, key, value, causal_mask(time, query.device)
    )
    future = torch.triu(torch.ones(time, time, dtype=torch.bool), diagonal=1)
    leaked = weights.masked_select(future)
    assert torch.equal(leaked, torch.zeros_like(leaked))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3, time), atol=1e-5)


def check_future_tokens_do_not_leak() -> None:
    torch.manual_seed(0)
    model = TransformerLM(
        vocab_size=32, n_embd=32, n_head=4, n_layer=2, block_size=16, dropout=0.0
    )
    model.eval()
    x = torch.randint(0, 32, (3, 16))
    logits, _ = model(x)
    changed = x.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 32
    logits_changed, _ = model(changed)
    assert torch.allclose(logits[:, :-1], logits_changed[:, :-1], atol=1e-5)
    assert not torch.allclose(logits[:, -1], logits_changed[:, -1])


def check_reverse_targets_and_loss_mask() -> None:
    length = 8
    x, y = get_reverse_batch(batch_size=4, length=length, device=torch.device("cpu"))
    assert x.shape == (4, 2 * length)
    assert torch.equal(y[:, :length], torch.full((4, length), -100))
    assert torch.equal(y[:, length:], x[:, :length].flip(dims=(1,)))

    model = TransformerLM(
        vocab_size=REVERSE_VOCAB_SIZE,
        n_embd=32,
        n_head=4,
        n_layer=1,
        block_size=2 * length,
        dropout=0.0,
    )
    model.eval()
    logits, loss = model(x, y)
    assert logits.shape == (4, 2 * length, REVERSE_VOCAB_SIZE)
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_y = y.reshape(-1)
    keep = flat_y != -100
    manual = F.cross_entropy(flat_logits[keep], flat_y[keep])
    assert torch.allclose(loss, manual)


def check_weight_tying() -> None:
    model = TransformerLM(vocab_size=20, n_embd=16, n_head=4, n_layer=1, block_size=8)
    assert model.head.weight is model.tok_emb.weight
    ids = [id(p) for p in model.parameters()]
    assert len(ids) == len(set(ids))
    assert count_parameters(model) == sum(p.numel() for p in model.parameters())


def check_loss_decreases() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    length = 8
    model = TransformerLM(
        vocab_size=REVERSE_VOCAB_SIZE,
        n_embd=64,
        n_head=4,
        n_layer=2,
        block_size=2 * length,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(40):
        x, y = get_reverse_batch(64, length, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.3, losses


if __name__ == "__main__":
    check_softmax_and_loss()
    check_causal_weights()
    check_future_tokens_do_not_leak()
    check_reverse_targets_and_loss_mask()
    check_weight_tying()
    check_loss_decreases()
    print("all checks passed")
