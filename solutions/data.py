"""Two training sets.

reverse
    Synthetic. A string of random letters, a separator, then those letters
    backwards:  "krmqpdw>wdpqmrk". No download. A causal model can learn this
    because, once it has read the separator, the whole string is in its
    context. Loss is applied only to the reversed half. If this task's
    accuracy goes to 1, attention, the mask, and the training loop are wired
    up correctly.

shakespeare
    Public-domain plays, about 1.1 MB of characters, saved at
    data/tinyshakespeare.txt (the Tiny Shakespeare corpus). The vocabulary
    is the set of characters that occur, about 65 of them. The model reads
    a chunk and predicts the next character at every position.

    A uniform guess over 65 characters has loss log(65) ≈ 4.17. Training
    should start there and fall. With this small model you get local
    English (spaces, names, line breaks), not a play.

Bigger corpora, when this one is too easy. Do not start here:

    WikiText-2     a classic word-level language-modeling benchmark, small
                   enough to download from Hugging Face (`wikitext`).
    TinyStories    short synthetic children's stories. A small transformer
                   actually learns to write them. Paper: Eldan & Li, 2023.
    FineWeb-Edu    a cleaned slice of web text used to pretrain real models.
                   Far too large for a from-scratch laptop run.
"""

from __future__ import annotations

from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parent / "data"
SHAKESPEARE_PATH = DATA_DIR / "tinyshakespeare.txt"

REVERSE_CHARS = "abcdefghijklmnopqrstuvwxyz"
SEP_CHAR = ">"
SEP_ID = len(REVERSE_CHARS)
REVERSE_VOCAB_SIZE = SEP_ID + 1


class CharVocab:
    """Map between characters and integer ids. Built from the training text."""

    def __init__(self, text: str) -> None:
        self.itos = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def load_shakespeare() -> str:
    if not SHAKESPEARE_PATH.is_file():
        raise FileNotFoundError(
            f"missing {SHAKESPEARE_PATH}. It is the Tiny Shakespeare file "
            "from https://github.com/karpathy/char-rnn "
            "(data/tinyshakespeare/input.txt)."
        )
    return SHAKESPEARE_PATH.read_text(encoding="utf-8")


def get_lm_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random chunks for next-character prediction.

    If the chunk of ids is c[0] c[1] c[2] c[3], we train on

        input  c[0] c[1] c[2]
        target c[1] c[2] c[3]

    implemented by taking a window of length block_size for both, offset by
    one. Position t of the input has seen c[0]..c[t] (because of the causal
    mask) and is graded on c[t+1].
    """
    if len(data) <= block_size + 1:
        raise ValueError("text is shorter than one training window")
    last_start = len(data) - block_size - 1
    starts = torch.randint(0, last_start + 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts.tolist()])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in starts.tolist()])
    return x.to(device), y.to(device)


def get_reverse_batch(
    batch_size: int,
    length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A batch of `length` random letters, reversed after a separator.

    length = 3 builds rows like:

        full sequence:   c0 c1 c2  >  c2 c1 c0
        model input x:   c0 c1 c2  >  c2 c1
        targets y:       c1 c2  >  c2 c1 c0
        loss mask:       -- -- --  (graded) (graded) (graded)

    Targets before the reversed half are -100. F.cross_entropy skips those
    positions, so the loss only measures "did you write the string backwards
    after you saw '>'?".
    """
    letters = torch.randint(0, len(REVERSE_CHARS), (batch_size, length), device=device)
    sep = torch.full((batch_size, 1), SEP_ID, device=device, dtype=torch.long)
    full = torch.cat([letters, sep, letters.flip(dims=(1,))], dim=1)
    x = full[:, :-1]
    y = full[:, 1:].clone()
    y[:, :length] = -100
    return x, y


def decode_reverse(ids) -> str:
    alphabet = REVERSE_CHARS + SEP_CHAR
    return "".join(alphabet[int(i)] for i in ids)
