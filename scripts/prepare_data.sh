#!/usr/bin/env bash
# Build a demo corpus for Black Hole.
# With no arguments it generates a synthetic English corpus (works offline);
# pass file paths to concatenate real text instead.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

if [ "$#" -gt 0 ]; then
  cat "$@" > data/raw/corpus.txt
else
  python - <<'PY'
"""Synthetic demo corpus: diverse English sentences so the tokenizer/model have signal."""
import random
random.seed(1337)

subjects = ["The black hole", "A language model", "The transformer", "Scientists",
            "The galaxy", "An astronomer", "The neural network", "Time"]
verbs = ["observes", "predicts", "bends", "learns", "reveals", "computes", "measures", "transforms"]
objects = ["the next token", "light and time", "the distant stars", "attention weights",
           "the curvature of space", "patterns in data", "the event horizon", "gradients"]
tails = ["with surprising accuracy.", "across the void of space.", "under the pull of gravity.",
         "one layer at a time.", "without ever seeing the whole sentence.", "in a single forward pass.",
         "as it falls past the horizon.", "until the loss converges."]

lines = []
for _ in range(60000):
    s = random.choice(subjects); v = random.choice(verbs); o = random.choice(objects); t = random.choice(tails)
    lines.append(f"{s} {v} {o} {t}")
    if random.random() < 0.08:
        lines.append("")
    if random.random() < 0.05:
        lines.append(f"For example, when {s.lower()} {v} {o}, {t.lower()}")

with open("data/raw/corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"synthetic corpus: {len(lines):,} lines -> data/raw/corpus.txt")
PY
fi

# 90/10 train/val split (line-level)
python - <<'PY'
from pathlib import Path
lines = Path("data/raw/corpus.txt").read_text(encoding="utf-8").splitlines()
cut = int(len(lines) * 0.9)
Path("data/raw/train.txt").write_text("\n".join(lines[:cut]), encoding="utf-8")
Path("data/raw/val.txt").write_text("\n".join(lines[cut:]), encoding="utf-8")
print(f"train {cut:,} lines / val {len(lines)-cut:,} lines")
PY
