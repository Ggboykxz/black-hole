"""Parallel text -> token .bin conversion.

The single-process path in ``blackhole.data.tokenize_corpus`` runs at ~1.45M
chars/s (one Rust encode_batch call at a time). With 2 cores we can roughly
double that by splitting the file into byte-range chunks and encoding each in
its own process, each with its own tokenizer instance.

Chunk boundaries are aligned on newlines so no document is split.

Usage:
    python scripts/tokenize_parallel.py --input data/raw/train.txt \\
        --out data/tokenized/train.bin --tokenizer data/tokenizer.json --workers 2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _worker(args) -> tuple[int, int, int]:
    """Encode a byte-range chunk. Returns (start, n_chars, n_tokens) and writes to a temp file."""
    path, tok_path, out_tmp, start, end = args
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_path)
    vocab = tok.get_vocab_size()
    np_dtype = np.uint16 if vocab < 65536 else np.uint32

    with open(path, "rb") as f:
        f.seek(start)
        # skip to the next newline unless we are at the true start
        if start > 0:
            f.readline()
            start = f.tell()
        raw = f.read(end - start)

    text = raw.decode("utf-8", errors="replace")
    # drop the trailing partial line for all but the last chunk
    if end < Path(path).stat().st_size:
        cut = text.rfind("\n")
        if cut != -1:
            text = text[: cut + 1]

    docs = [d for d in text.split("\n\n") if d.strip()]
    ids: list[int] = []
    eos = tok.token_to_id("<eos>") or 2
    for enc in tok.encode_batch(docs):
        ids.extend(enc.ids)
        ids.append(eos)

    arr = np.asarray(ids, dtype=np_dtype)
    arr.tofile(out_tmp)
    return start, len(text), len(ids)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--chunk-mb", type=int, default=128)
    args = p.parse_args(argv)

    inp = Path(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    size = inp.stat().st_size
    chunk = args.chunk_mb * 1024 * 1024
    n_chunks = max(1, min(args.workers * 4, size // chunk + 1))

    bounds = [round(i * size / n_chunks) for i in range(n_chunks + 1)]
    jobs = [(str(inp), args.tokenizer, str(out) + f".part{i}", bounds[i], bounds[i + 1])
            for i in range(n_chunks)]

    print(f"{inp.name}: {size/1e6:.0f} MB -> {n_chunks} chunks, {args.workers} workers", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_worker, jobs):
            results.append(r)
            done = sum(x[2] for x in results)
            print(f"   {len(results)}/{n_chunks} chunks | {done/1e6:.1f}M tokens | {time.time()-t0:.0f}s",
                  flush=True)

    # concatenate in byte order
    results.sort(key=lambda r: r[0])
    with open(out, "wb") as dst:
        for i in range(n_chunks):
            part = Path(str(out) + f".part{i}")
            with open(part, "rb") as src:
                while True:
                    buf = src.read(1 << 24)
                    if not buf:
                        break
                    dst.write(buf)
            part.unlink()

    total_tokens = sum(r[2] for r in results)
    total_chars = sum(r[1] for r in results)
    dt = time.time() - t0
    from blackhole.tokenizer import load_tokenizer
    vocab = load_tokenizer(args.tokenizer).get_vocab_size()
    meta = {"dtype": "uint16" if vocab < 65536 else "uint32", "vocab": vocab, "tokens": total_tokens}
    Path(str(out) + ".meta").write_text(json.dumps(meta))

    print(f"\nOK {out}: {total_tokens:,} tokens ({total_chars/1e6:.0f} chars) "
          f"in {dt:.0f}s = {total_chars/1e6/dt:.2f} M chars/s", flush=True)


if __name__ == "__main__":
    main()
