#!/usr/bin/env bash
# Real-data pretraining pipeline for Black Hole.
#   corpus (already built) -> BPE tokenizer -> tokenized .bin -> pretrain -> eval -> sample
set -euo pipefail
cd "$(dirname "$0")/.."

VOCAB="${VOCAB:-32000}"
CONFIG="${1:-configs/train-small.json}"
OUT_DIR=$(python -c "import json;print(json.load(open('$CONFIG')).get('out_dir','runs/blackhole'))")

echo "== Black Hole real-data pretraining =="
echo "   config=$CONFIG  vocab=$VOCAB  out=$OUT_DIR"

# ---------------------------------------------------------------- tokenizer
if [ ! -f data/tokenizer.json ] || [ ! -f data/tokenizer.json.stamp ] || \
   [ "$(cat data/tokenizer.json.stamp)" != "$VOCAB" ]; then
  echo "-- training byte-level BPE tokenizer (vocab=$VOCAB)"
  # train on a sample: full corpus is overkill and slows BPE down
  head -c 40000000 data/raw/train.txt > /tmp/tok_sample.txt
  python -m blackhole.tokenizer --input /tmp/tok_sample.txt \
      --out data/tokenizer.json --vocab-size "$VOCAB" --min-frequency 2
  printf '%s' "$VOCAB" > data/tokenizer.json.stamp
else
  echo "-- tokenizer present (vocab=$VOCAB)"
fi

# ---------------------------------------------------------------- tokenize
echo "-- tokenizing corpus"
python - <<'PY'
import json
from pathlib import Path
from blackhole.data import tokenize_corpus
from blackhole.tokenizer import load_tokenizer

tok = load_tokenizer("data/tokenizer.json")
vocab = tok.get_vocab_size()
dt = "uint16" if vocab < 65536 else "uint32"
total = 0
for split in ("train", "val"):
    src = Path(f"data/raw/{split}.txt")
    dst = Path(f"data/tokenized/{split}.bin")
    if src.exists() and (not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime):
        n = tokenize_corpus(src, dst, "data/tokenizer.json", add_eos=True)
        Path(str(dst) + ".meta").write_text(json.dumps({"dtype": dt, "vocab": vocab, "tokens": n}))
        total += n
    else:
        print(f"{split}: up to date ({dst})")
if total:
    print(f"tokenized total: {total:,} tokens")
PY

# ---------------------------------------------------------------- sync vocab
python - <<'PY'
import json
from pathlib import Path
from blackhole.tokenizer import load_tokenizer
vocab = load_tokenizer("data/tokenizer.json").get_vocab_size()
for cfg in ("configs/blackhole-tiny.json", "configs/blackhole-small.json"):
    p = Path(cfg)
    d = json.loads(p.read_text())
    if d.get("vocab_size") != vocab:
        d["vocab_size"] = vocab
        p.write_text(json.dumps(d, indent=2) + "\n")
        print(f"{cfg}: vocab_size -> {vocab}")
PY

# ---------------------------------------------------------------- train
echo "-- pretraining ($CONFIG)"
python -m blackhole.train --config "$CONFIG" "$@"

# ---------------------------------------------------------------- eval + sample
python -m blackhole.evaluate --checkpoint "$OUT_DIR/ckpt_final.pt" \
    --data data/tokenized/val.bin --seq-len 1024 --batches 50 \
    --out "$OUT_DIR/metrics.json" || true

python -m blackhole.generate --checkpoint "$OUT_DIR/ckpt_final.pt" \
    --tokenizer data/tokenizer.json \
    --prompt "The history of artificial intelligence" --max-new-tokens 150 || true
