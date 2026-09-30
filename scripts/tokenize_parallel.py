"""Chunked, memory-bounded text -> token .bin conversion.

Why this exists: ``blackhole.data.tokenize_corpus`` runs at ~1.06M chars/s and a
naive parallel version OOMs, because it asks the pool for ``workers*4`` chunks
regardless of ``--chunk-mb`` (so a 6.5GB file became 8 chunks of 816MB -> 5.9GB
RSS -> kernel OOM kill).

Here:
* the chunk count comes from the FILE SIZE divided by ``--chunk-mb``
* each worker decodes one chunk and encodes documents in small sub-batches,
  flushing tokens to disk as it goes -> memory stays ~O(chunk), not O(corpus)
* ``TOKENIZERS_PARALLELISM`` lets encode_batch itself fan out over cores, so
  extra processes add little; we keep ``--workers`` for CPU-bound machines

Usage:
    python scripts/tokenize_parallel.py --input data/raw/train.txt \\
        --out data/tokenized/train.bin --tokenizer data/tokenizer.json --chunk-mb 64
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SUBBATCH_DOCS = 512  # documents per encode_batch call -> bounds Encoding memory
FLUSH_EVERY = 1 << 20  # tokens per write()


def _worker(args) -> tuple[int, int, int]:
    """Encode one byte-range chunk -> part file. Returns (start, chars, tokens)."""
    path, tok_path, out_tmp, start, end, file_size = args
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_path)
    vocab = tok.get_vocab_size()
    np_dtype = np.uint16 if vocab < 65536 else np.uint32
    eos = tok.token_to_id("<eos>")
    if eos is None:
        eos = 2

    with open(path, "rb") as f:
        f.seek(start)
        if start > 0:
            f.readline()            # never start mid-document
            start = f.tell()
        raw = f.read(end - start)

    text = raw.decode("utf-8", errors="replace")
    del raw
    if end < file_size:
        cut = text.rfind("\n")
        if cut != -1:
            text = text[: cut + 1]   # trailing partial line belongs to the next chunk

    docs = [d for d in text.split("\n\n") if d.strip()]
    del text

    n_chars = sum(len(d) for d in docs)
    n_tokens = 0
    buf: list[int] = []
    wrote = 0

    with open(out_tmp, "wb") as out:
        for i in range(0, len(docs), SUBBATCH_DOCS):
            batch = docs[i : i + SUBBATCH_DOCS]
            for enc in tok.encode_batch(batch):
                buf.extend(enc.ids)
                buf.append(eos)
            if len(buf) >= FLUSH_EVERY:
                arr = np.asarray(buf, dtype=np_dtype)
                arr.tofile(out)
                wrote += arr.size
                buf.clear()
        if buf:
            arr = np.asarray(buf, dtype=np_dtype)
            arr.tofile(out)
            wrote += arr.size

    return start, n_chars, wrote


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--chunk-mb", type=int, default=64)
    args = p.parse_args(argv)

    # let encode_batch's own rayon pool use every core inside each process
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    inp = Path(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    size = inp.stat().st_size
    chunk = max(1, args.chunk_mb) * 1024 * 1024
    n_chunks = max(1, (size + chunk - 1) // chunk)          # <- derived from file size
    n_chunks = min(n_chunks, max(args.workers * 8, args.workers))

    bounds = [round(i * size / n_chunks) for i in range(n_chunks + 1)]
    jobs = [(str(inp), args.tokenizer, str(out) + f".part{i}",
             bounds[i], bounds[i + 1], size) for i in range(n_chunks)]

    print(f"{inp.name}: {size/1e9:.2f} GB -> {n_chunks} chunks of "
          f"~{size/n_chunks/1e6:.0f} MB, {args.workers} workers", flush=True)

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_worker, jobs):
            results.append(r)
            done = sum(x[2] for x in results)
            chars = sum(x[1] for x in results)
            el = time.time() - t0
            print(f"   {len(results)}/{n_chunks} | {chars/1e6:.0f}M chars | {done/1e6:.1f}M tok "
                  f"| {el:.0f}s | {chars/1e6/max(el,1e-9):.2f} M chars/s", flush=True)

    results.sort(key=lambda r: r[0])
    tmp = Path(str(out) + ".tmp")
    with open(tmp, "wb") as dst:
        for i in range(n_chunks):
            part = Path(str(out) + f".part{i}")
            with open(part, "rb") as src:
                while True:
                    buf = src.read(1 << 24)
                    if not buf:
                        break
                    dst.write(buf)
            part.unlink()
    tmp.replace(out)

    total_tokens = sum(r[2] for r in results)
    total_chars = sum(r[1] for r in results)
    dt = time.time() - t0

    from blackhole.tokenizer import load_tokenizer
    vocab = load_tokenizer(args.tokenizer).get_vocab_size()
    Path(str(out) + ".meta").write_text(json.dumps(
        {"dtype": "uint16" if vocab < 65536 else "uint32",
         "vocab": vocab, "tokens": total_tokens}))

    print(f"\nOK {out}: {total_tokens:,} tokens from {total_chars/1e6:.0f}M chars "
          f"in {dt/60:.1f} min = {total_chars/1e6/dt:.2f} M chars/s", flush=True)


if __name__ == "__main__":
    main()
