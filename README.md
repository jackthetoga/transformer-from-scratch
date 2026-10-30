# Write a transformer, from an empty file

A decoder-only transformer, typed and trained from scratch. The model reads a prefix and predicts the next token: one stack of blocks, a causal mask, and a cross-entropy loss. That is the GPT layout.

[`tutorial.ipynb`](tutorial.ipynb) is the lesson. Cells marked `EXERCISE` are yours. Cells marked `PROVIDED` and `TEST` are the lesson and the checker. Run the notebook from top to bottom. A passing test is the specification of the previous exercise.

[`solutions/`](solutions/) is a finished version of the same pieces. Open one function after a test has failed and the hint was not enough. Transcribing a whole file will not teach you the shapes.

## What you build

1. One training step, on a small classifier.
2. Softmax and cross-entropy, the loss every language model uses.
3. A causal mask, then scaled dot-product attention, then multi-head attention.
4. One pre-norm transformer block: attention, an MLP, and residual adds.
5. The full language model.
6. Batches, then training on string reversals, then on Shakespeare.

```text
token ids (batch, time)
  -> token embedding + position embedding          (batch, time, n_embd)
  -> N blocks:
       x = x + Attention(LayerNorm(x))
       x = x + MLP(LayerNorm(x))
  -> LayerNorm
  -> linear map to vocabulary logits               (batch, time, vocab)
  -> cross-entropy against the next token
```

Attention moves information between positions. The MLP runs on its own at every position. The residual add (`x + ...`) lets a block start close to the identity, which is why a stack of blocks still trains.

## Setup

Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead of `source`.

Open `tutorial.ipynb` and select this project's `.venv` as the kernel. In Cursor, use the kernel picker on the notebook if another Python is selected. Training uses CUDA, then Apple MPS, then CPU. The unit tests stay on CPU so the numbers are easy to compare.

`.venv/` is local and listed in `.gitignore`. Do not commit it.

## Run the reference model

`solutions/train.py` is the same model, already written.

- `reverse` learns a function: write a random string of letters backwards after `>`. A few hundred steps on a laptop is enough for exact reversals. If accuracy goes to 1, attention, the mask, and the training loop are wired up.
- `shakespeare` is character-level language modeling on [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) (public-domain plays, about 1.1 MB, ~65 characters). Loss starts near `log(65) ≈ 4.17`, the score of a uniform guess, and should fall. The sample at the end shows local patterns (line breaks, names), not a finished play. Raise `--steps` to keep going.

```bash
source .venv/bin/activate
cd solutions
python train.py reverse
python train.py shakespeare
```

Flags: `--steps`, `--batch-size`, `--eval-every`, `--seed`, and `--device` (`auto`, `cpu`, `mps`, or `cuda`). Checkpoints are written to `solutions/runs/`, which is gitignored.

## Layout

| Path | What it is |
| --- | --- |
| `tutorial.ipynb` | The lesson. Fill in the exercises here. |
| `solutions/` | Reference implementations and `train.py`. |
| `data/tinyshakespeare.txt` | Tiny Shakespeare, from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn). Public domain. |
| `requirements.txt` | PyTorch, NumPy, and a Jupyter kernel. |
