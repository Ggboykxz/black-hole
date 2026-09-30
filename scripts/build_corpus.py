"""Build a high-quality training corpus from FineWeb-Edu parquet shards.

Pipeline: read parquet -> filter (language, educational score, length, junk
patterns) -> exact dedupe -> near-duplicate ngram dedupe -> write train/val text.

FineWeb-Edu columns: text, id, dump, url, file_path, language, language_score,
token_count, score, int_score.

Usage:
    python scripts/build_corpus.py --shards "data/parquet/*.parquet" \\
        --max-chars 600000000 --out data/raw
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

# --------------------------------------------------------------------- filters
JUNK_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"\bsign in\b.{0,40}\bsign up\b",
        r"cookie policy",
        r"all rights reserved",
        r"subscribe to our newsletter",
        r"skip to (main|content|navigation) content",
        r"enable javascript",
        r"click here to (accept|agree)",
        r"404 not found",
        r"temporarily unavailable",
        r"captcha",
    ]
]


def looks_junk(text: str) -> bool:
    if len(text) < 200:
        return True
    if len(text) > 100_000:
        return True
    if any(p.search(text[:1500]) for p in JUNK_PATTERNS):
        return True
    # too many digits/symbols -> tables, code dumps, base64
    sample = text[:4000]
    alnum = sum(c.isalnum() or c.isspace() for c in sample) / len(sample)
    if alnum < 0.80:
        return True
    # too little prose: average word length extremes
    words = sample.split()
    if len(words) < 30:
        return True
    avg_word = sum(len(w) for w in words) / len(words)
    if avg_word < 3.0 or avg_word > 12.0:
        return True
    # repeated line spam
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines and len(set(lines)) / len(lines) < 0.5:
        return True
    return False


def normalize(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ngram_signature(text: str, n: int = 13, k: int = 12) -> frozenset:
    """Cheap near-dup key: hash of shingles, sampled to k items."""
    words = text.lower().split()
    if len(words) < n:
        return frozenset()
    sig = set()
    for i in range(0, len(words) - n + 1, 7):  # stride -> fast
        gram = " ".join(words[i : i + n])
        sig.add(hash(gram))
        if len(sig) >= k * 4:
            break
    return frozenset(list(sig)[:k])


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", default="data/parquet/*.parquet")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--max-chars", type=int, default=600_000_000)
    p.add_argument("--min-score", type=float, default=3.0, help="FineWeb-Edu educational score")
    p.add_argument("--val-chars", type=int, default=3_000_000)
    p.add_argument("--near-dup", action="store_true", help="enable near-duplicate filtering (slower)")
    args = p.parse_args(argv)

    files = sorted(glob.glob(args.shards))
    if not files:
        sys.exit(f"no parquet files match {args.shards}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    seen_exact: set[int] = set()
    near_sigs: list[frozenset] = []
    kept_chars = 0
    stats = dict(read=0, junk=0, lowscore=0, dup=0, kept=0)

    train_f = open(out / "train.txt", "w", encoding="utf-8")
    # only touch val.txt when we were asked to (re)write it, so a run can keep the
    # previous validation split for run-to-run comparability
    val_f = open(out / "val.txt", "w", encoding="utf-8") if args.val_chars > 0 else None
    val_written = 0

    for fp in files:
        if kept_chars >= args.max_chars:
            break
        pf = pq.ParquetFile(fp)
        print(f"-> {fp}: {pf.metadata.num_rows:,} rows")
        for batch in pf.iter_batches(batch_size=2048, columns=["text", "language", "score"]):
            if kept_chars >= args.max_chars:
                break
            texts = batch.column("text").to_pylist()
            langs = batch.column("language").to_pylist()
            scores = batch.column("score").to_pylist()

            for text, lang, score in zip(texts, langs, scores):
                stats["read"] += 1
                if lang != "en" or (score is not None and score < args.min_score):
                    stats["lowscore"] += 1
                    continue
                text = normalize(text or "")
                if looks_junk(text):
                    stats["junk"] += 1
                    continue
                h = hash(text)
                if h in seen_exact:
                    stats["dup"] += 1
                    continue
                if args.near_dup:
                    sig = ngram_signature(text)
                    if sig and any(jaccard(sig, s) > 0.6 for s in near_sigs[-20000:]):
                        stats["dup"] += 1
                        continue
                    near_sigs.append(sig)
                seen_exact.add(h)

                stats["kept"] += 1
                if val_f is not None and val_written < args.val_chars:
                    val_f.write(text + "\n\n")
                    val_written += len(text)
                else:
                    train_f.write(text + "\n\n")
                    kept_chars += len(text)

            if stats["read"] % 20000 < 2048:
                print(f"   read {stats['read']:,} | kept {stats['kept']:,} "
                      f"| {kept_chars/1e6:.0f}M chars ({kept_chars*0.25/1e6:.0f}M tok est)")

    train_f.close()
    if val_f is not None:
        val_f.close()

    print("\n=== corpus ===")
    print(f"read      {stats['read']:>12,}")
    print(f"lowscore  {stats['lowscore']:>12,}")
    print(f"junk      {stats['junk']:>12,}")
    print(f"dup       {stats['dup']:>12,}")
    print(f"kept      {stats['kept']:>12,}")
    print(f"train     {kept_chars/1e6:>12.1f} M chars (~{kept_chars*0.25/1e6:.0f} M tokens)")
    print(f"val       {val_written/1e6:>12.1f} M chars")


if __name__ == "__main__":
    main()
