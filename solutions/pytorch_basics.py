"""PyTorch, softmax, and cross-entropy, before the transformer.

Run this file. The printed sections match the functions below.

    .venv/bin/python transformer-from-scratch/pytorch_basics.py

What this file is for
----------------------
You already know an MLP: affine layer, nonlinearity, affine layer, then a loss.
PyTorch is the library that builds that computation and fills in the gradients
for you. This file introduces only the pieces the transformer uses:

  Tensor          a numeric array. If requires_grad=True, PyTorch records ops.
  nn.Module       a layer or a whole model. Submodules assigned as attributes
                  are found by model.parameters(), which is what the optimizer
                  updates.
  nn.Linear       one affine map, y = x W^T + b. This is the layer inside an MLP.
  loss.backward   writes d(loss)/d(parameter) into parameter.grad
  optimizer.step  applies those gradients (AdamW does a fancier version of
                  "parameter -= learning_rate * gradient")

Two facts that matter for every language model below:

  1. The model outputs logits, not probabilities. Logits are raw scores,
     one per class, and they do not sum to 1.
  2. F.cross_entropy expects those logits. It runs log-softmax internally.
     Passing probabilities into it is a silent bug: the number is finite,
     but it is the wrong loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Turn scores into a probability distribution along `dim`.

    softmax(z)_i = exp(z_i) / sum_j exp(z_j)

    Subtracting the max does not change the result, because exp(z - c) =
    exp(z) / exp(c), and that constant cancels in the fraction. It does
    keep exp() from overflowing when a score is large.
    """
    shifted = logits - logits.amax(dim=dim, keepdim=True)
    exp_scores = shifted.exp()
    return exp_scores / exp_scores.sum(dim=dim, keepdim=True)


def cross_entropy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood of the correct class.

    logits:  (N, C) raw scores for N examples and C classes
    targets: (N,) integer class ids, each in 0 .. C-1

    For a single example with correct class k:
        loss = -log(softmax(z)_k)

    Written without forming the probabilities, that is:
        loss = -z_k + log(sum_j exp(z_j))

    logsumexp is the stable way to compute log(sum(exp(z))). Averaging over
    the batch is what F.cross_entropy does by default. A one-hot target
    vector y gives the same number, because -sum_i y_i log p_i keeps only
    the term for the correct class.
    """
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    chosen = log_probs.gather(dim=-1, index=targets[:, None]).squeeze(-1)
    return -chosen.mean()


class MLP(nn.Module):
    """Two-layer classifier. Same shape of model you already know.

    forward returns logits of shape (batch, n_classes). It does not call
    softmax. The loss owns that step.
    """

    def __init__(self, n_in: int, n_hidden: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def demo_softmax_and_loss() -> None:
    _section("1. softmax")
    logits = torch.tensor([2.0, 1.0, 0.1])
    ours = stable_softmax(logits)
    theirs = torch.softmax(logits, dim=-1)
    print("logits          ", logits.tolist())
    print("softmax (ours)  ", [round(v, 4) for v in ours.tolist()])
    print("softmax (torch) ", [round(v, 4) for v in theirs.tolist()])
    print("sums to         ", round(float(ours.sum()), 6))
    print("largest score becomes the largest probability, and every entry is > 0")

    _section("2. cross-entropy is -log p(correct class)")
    # One example, three classes, correct class is 0.
    row = logits[None, :]
    target = torch.tensor([0])
    ours_loss = cross_entropy_from_logits(row, target)
    torch_loss = F.cross_entropy(row, target)
    prob_correct = ours[0]
    print(f"p(class 0)              {float(prob_correct):.4f}")
    print(f"-log p(class 0)         {-prob_correct.log():.4f}")
    print(f"our cross-entropy       {float(ours_loss):.4f}")
    print(f"F.cross_entropy(logits) {float(torch_loss):.4f}")

    already_probabilities = ours[None, :]
    wrong = F.cross_entropy(already_probabilities, target)
    print(f"F.cross_entropy(softmax(logits)) = {float(wrong):.4f}")
    print("that last call is wrong: cross_entropy applies log-softmax itself")
    print("so it treated probabilities as if they were logits")

    _section("3. the gradient of softmax + cross-entropy is p - one_hot")
    z = logits.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(z[None, :], target)
    loss.backward()
    probs = torch.softmax(z, dim=-1)
    # Class 0 is correct, so we subtract 1 from that entry of p.
    expected = probs.detach().clone()
    expected[0] -= 1
    print("softmax p           ", [round(v, 4) for v in probs.tolist()])
    print("d(loss)/d(logits)   ", [round(v, 4) for v in z.grad.tolist()])
    print("p - one_hot(target) ", [round(v, 4) for v in expected.tolist()])
    print("the correct class is pushed down, every other class is pushed up")
    print("in proportion to how much probability it currently has")

    _section("4. backward adds into .grad, so zero it before each step")
    weight = torch.tensor([1.0], requires_grad=True)
    (weight * 2).sum().backward()
    (weight * 3).sum().backward()
    print("two backwards without zeroing, grad is", float(weight.grad), "(2 + 3)")
    weight.grad = None
    (weight * 3).sum().backward()
    print("after clearing, one backward, grad is", float(weight.grad))


def demo_training_step() -> None:
    """Fit a tiny MLP. Same loop the transformer trainer uses, with less model."""
    _section("5. one training loop: forward, loss, zero_grad, backward, step")
    torch.manual_seed(0)
    n_classes = 4
    # Each class is a cluster in 2D. The MLP has to predict the cluster id.
    centers = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    model = MLP(n_in=2, n_hidden=32, n_classes=n_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    def batch(n: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.randint(0, n_classes, (n,))
        x = centers[y] + 0.4 * torch.randn(n, 2)
        return x, y

    for step in range(1, 81):
        x, y = batch()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (1, 20, 80):
            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean()
            print(f"step {step:3d}  loss {loss.item():.3f}  batch accuracy {acc.item():.2f}")

    print("parameters the optimizer owns:")
    for name, param in model.named_parameters():
        print(f"  {name:20s} shape {tuple(param.shape)}")


if __name__ == "__main__":
    demo_softmax_and_loss()
    demo_training_step()
