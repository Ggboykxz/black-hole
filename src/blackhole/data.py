"""Data pipeline: corpus text -> tokenized binary shards -> random-block batches.

The canonical layout is a flat ``uint16``/``uint32`` token stream saved as ``.bin``
files (the "nanoGPT" format): simple, mmap-friendly, and fast on a single GPU.

Provides:
* ``tokenize_corpus``   — stream a text file into a token .bin
* ``load_tokens``       — mmap a .bin as an int64 array
* ``get_batch``         — sample random contiguous blocks (train/val split)
* ``make_dataloader``   — a simple iterable over batches (no torch.utils.data dep needed)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .tokenizer import BOS, EOS, load_tokenizer

__all__ = ["tokenize_corpus", "load_tokens", "get_batch", "Batcher"]


# --------------------------------------------------------------------------- tokenization
def tokenize_corpus(
    input_path: str | Path,
    out_path: str | Path,
    tokenizer_path: str | Path,
    *,
    add_eos: bool = True,
    chunk_chars: int = 1_000_000,
    dtype: str = "auto",
) -> int:
    """Convert a UTF-8 text file into a flat binary token stream.

    Streams the file in chunks so corpora larger than RAM work. The output dtype is
    uint16 when vocab < 65536 else uint32.
    """
    tok = load_tokenizer(tokenizer_path)
    vocab_size = tok.get_vocab_size()
    np_dtype = np.uint16 if (dtype == "auto" and vocab_size < 65536) else (
        np.uint32 if dtype == "auto" else np.dtype(dtype)
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    bos_id = tok.token_to_id("<bos>") if tok.token_to_id("<bos>") is not None else BOS
    eos_id = tok.token_to_id("<eos>") if tok.token_to_id("<eos>") is not None else EOS

    with open(input_path, "r", encoding="utf-8", errors="replace") as src, \
            open(out_path, "wb") as dst:
        buf: list[int] = []
        pending = ""
        while True:
            chunk = src.read(chunk_chars)
            if not chunk:
                break
            # keep last 512 chars of previous chunk attached so words aren't split
            text = (pending + chunk) if pending else chunk
            ids = tok.encode(text, add_special_tokens=False).ids
            if add_eos:
                ids = ids + [eos_id]
            buf.extend(ids)
            # re-encode the tail next round: drop tokens from a possibly-split word
            pending = ""
            if len(buf) > 1 << 16:
                arr = np.asarray(buf, dtype=np_dtype)
                arr.tofile(dst)
                total += arr.size
                buf = []
        if buf:
            arr = np.asarray(buf, dtype=np_dtype)
            arr.tofile(dst)
            total += arr.size

    print(f"tokenized {input_path} -> {out_path}: {total:,} tokens "
          f"({total * np.dtype(np_dtype).itemsize / 1e6:.1f} MB, dtype={np_dtype})")
    return total


def load_tokens(path: str | Path) -> np.ndarray:
    """Memory-map a token .bin. Returns an unsigned-int array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"token file not found: {path}")
    # decide dtype from file size + tokenizer-independent heuristic:
    # caller is expected to have written uint16 (vocab<65k). Peek at vocab via suffix meta.
    meta = path.with_suffix(path.suffix + ".meta")
    if meta.exists():
        import json
        dt = json.loads(meta.read_text()).get("dtype", "uint16")
    else:
        dt = "uint16"  # project default
    return np.memmap(path, dtype=dt, mode="r")


def get_batch(
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` random contiguous (x, y) blocks shifted by one token."""
    max_start = data.size - seq_len - 1
    if max_start <= 0:
        raise ValueError(
            f"dataset too small: {data.size} tokens needs > seq_len+1 = {seq_len + 1}"
        )
    starts = rng.integers(0, max_start, size=batch_size, dtype=np.int64)
    xs = np.stack([data[s : s + seq_len] for s in starts]).astype(np.int64)
    ys = np.stack([data[s + 1 : s + seq_len + 1] for s in starts]).astype(np.int64)
    x = torch.from_numpy(xs).to(device, non_blocking=True)
    y = torch.from_numpy(ys).to(device, non_blocking=True)
    return x, y


class Batcher:
    """Stateful batch iterator that draws from train/val splits with a fixed RNG."""

    def __init__(
        self,
        train_bin: str | Path,
        val_bin: str | Path | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        seed: int = 1337,
    ) -> None:
        self.train = load_tokens(train_bin)
        self.val = load_tokens(val_bin) if (val_bin and Path(val_bin).exists()) else None
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.split = "train"

    def next_train(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.split = "train"
        return get_batch(self.train, self.batch_size, self.seq_len, self.device, self.rng)

    @torch.no_grad()
    def estimate_val_loss(self, model, num_batches: int, autocast_ctx) -> float:
        """Average cross-entropy over ``num_batches`` validation blocks."""
        if self.val is None or self.val.size <= self.seq_len + 1:
            return float("nan")
        model.eval()
        rng = np.random.default_rng(self.rng.integers(0, 2**31))
        losses = []
        for _ in range(num_batches):
            x, y = get_batch(self.val, self.batch_size, self.seq_len, self.device, rng)
            with autocast_ctx():
                _, loss = model(x, y)
            losses.append(float(loss))
        model.train()
        return float(np.mean(losses))


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Tokenize a corpus into a .bin stream")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--no-eos", action="store_true")
    args = p.parse_args(argv)

    n = tokenize_corpus(args.input, args.out, args.tokenizer, add_eos=not args.no_eos)

    # write dtype meta so load_tokens can reopen it faithfully
    import json
    tok = load_tokenizer(args.tokenizer)
    dt = "uint16" if tok.get_vocab_size() < 65536 else "uint32"
    Path(str(args.out) + ".meta").write_text(json.dumps({"dtype": dt, "tokens": n, "vocab": tok.get_vocab_size()}))


if __name__ == "__main__":
    main()
