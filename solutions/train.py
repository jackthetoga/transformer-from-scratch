"""Train the decoder-only transformer.

From solutions/, with the repo virtualenv activated:

    python train.py reverse
    python train.py shakespeare

`reverse` learns a function (write the string backwards after '>'). It is
the check that the model, the mask, and this loop are working. A few
hundred steps on a laptop is enough for exact reversals.

`shakespeare` is character-level language modeling on Tiny Shakespeare.
Loss starts near log(vocabulary size), which is what a uniform guess
scores, and should fall. The sample at the end will show local patterns
(line breaks, names), not a finished play. Raise --steps to keep going.

The loop is the same five lines as the MLP in pytorch_basics.py:

    logits, loss = model(x, y)     # forward
    optimizer.zero_grad()          # grads accumulate, so clear them
    loss.backward()                # fill parameter.grad
    clip_grad_norm_(...)           # stop one wild batch from taking a huge step
    optimizer.step()               # update the parameters
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from data import (
    REVERSE_VOCAB_SIZE,
    SEP_ID,
    CharVocab,
    decode_reverse,
    get_lm_batch,
    get_reverse_batch,
    load_shakespeare,
)
from model import TransformerLM, count_parameters

RUNS = Path(__file__).resolve().parent / "runs"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(model: TransformerLM, get_batch, batches: int) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = get_batch()
        _, loss = model(x, y)
        total += loss.item()
    model.train(was_training)
    return total / batches


@torch.no_grad()
def reverse_accuracy(
    model: TransformerLM, length: int, device: torch.device, n: int = 128
) -> tuple[float, float]:
    """Fraction of characters right, and fraction of whole strings right."""
    letters = torch.randint(0, 26, (n, length), device=device)
    sep = torch.full((n, 1), SEP_ID, device=device, dtype=torch.long)
    prompt = torch.cat([letters, sep], dim=1)
    out = model.generate(prompt, max_new_tokens=length, temperature=0)
    pred = out[:, length + 1 :]
    target = letters.flip(dims=(1,))
    token_acc = float((pred == target).float().mean())
    string_acc = float((pred == target).all(dim=1).float().mean())
    return token_acc, string_acc


def show_reversals(model: TransformerLM, length: int, device: torch.device) -> None:
    letters = torch.randint(0, 26, (4, length), device=device)
    sep = torch.full((4, 1), SEP_ID, device=device, dtype=torch.long)
    prompt = torch.cat([letters, sep], dim=1)
    out = model.generate(prompt, max_new_tokens=length, temperature=0)
    print("reversals (greedy):")
    for row, src in zip(out.tolist(), letters.tolist()):
        text = decode_reverse(row)
        source = decode_reverse(src)
        print(f"  {source} -> {text[length + 1 :]}   target {source[::-1]}")


def train_reverse(args, device: torch.device) -> None:
    length = 8
    # Input rows have length 2 * length (the last reversed character is the target only).
    block_size = 2 * length
    model = TransformerLM(
        vocab_size=REVERSE_VOCAB_SIZE,
        n_embd=64,
        n_head=4,
        n_layer=2,
        block_size=block_size,
        dropout=0.0,
    ).to(device)
    # Weight decay would pull weights toward 0. This task is a function we
    # want to fit exactly, so leave it off.
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    def get_batch():
        return get_reverse_batch(args.batch_size, length, device)

    print(f"device {device}  parameters {count_parameters(model):,}")
    print(f"uniform-guess loss log({REVERSE_VOCAB_SIZE}) = {math.log(REVERSE_VOCAB_SIZE):.3f}")
    run(args, model, optimizer, get_batch, eval_batches=8, task="reverse", length=length, device=device)
    show_reversals(model, length, device)


def train_shakespeare(args, device: torch.device) -> None:
    text = load_shakespeare()
    vocab = CharVocab(text)
    ids = torch.tensor(vocab.encode(text), dtype=torch.long)
    split = int(0.9 * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]
    block_size = 64
    model = TransformerLM(
        vocab_size=len(vocab),
        n_embd=128,
        n_head=4,
        n_layer=4,
        block_size=block_size,
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    def get_batch(split_ids=train_ids):
        return get_lm_batch(split_ids, block_size, args.batch_size, device)

    def get_val_batch():
        return get_lm_batch(val_ids, block_size, args.batch_size, device)

    print(f"device {device}  characters {len(ids):,}  vocab {len(vocab)}  parameters {count_parameters(model):,}")
    print(f"uniform-guess loss log({len(vocab)}) = {math.log(len(vocab)):.3f}")
    run(
        args,
        model,
        optimizer,
        get_batch,
        eval_batches=8,
        task="shakespeare",
        get_val_batch=get_val_batch,
        vocab=vocab,
        device=device,
    )


def run(
    args,
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    get_batch,
    eval_batches: int,
    task: str,
    device: torch.device,
    length: int | None = None,
    get_val_batch=None,
    vocab: CharVocab | None = None,
) -> None:
    model.train()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            train_loss = estimate_loss(model, get_batch, eval_batches)
            message = f"step {step:4d}  train {train_loss:.3f}"
            if get_val_batch is not None:
                val_loss = estimate_loss(model, get_val_batch, eval_batches)
                message += f"  val {val_loss:.3f}"
            if task == "reverse":
                token_acc, string_acc = reverse_accuracy(model, length, device)
                message += f"  char {token_acc:.2f}  string {string_acc:.2f}"
            print(message)

        if step == args.steps:
            break

        x, y = get_batch()
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{task}.pt"
    torch.save({"model": model.state_dict()}, path)
    print(f"saved {path}")

    if vocab is not None:
        prompt = "ROMEO:\n"
        idx = torch.tensor([vocab.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=400, temperature=0.8, top_k=20)
        print("\nsample:\n")
        print(vocab.decode(out[0].tolist()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("reverse", "shakespeare"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()
    if args.task == "reverse":
        args.steps = 500 if args.steps is None else args.steps
        args.batch_size = 128 if args.batch_size is None else args.batch_size
    else:
        args.steps = 1500 if args.steps is None else args.steps
        args.batch_size = 64 if args.batch_size is None else args.batch_size
    return args


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    if args.task == "reverse":
        train_reverse(args, device)
    else:
        train_shakespeare(args, device)
