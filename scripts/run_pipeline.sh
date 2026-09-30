#!/usr/bin/env bash
# End-to-end pipeline: corpus -> tokenizer -> tokenized bins -> train -> eval -> generate
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/train-tiny.json}"
echo "== Black Hole pipeline | config=$CONFIG =="

# 1. tokenizer
if [ ! -f data/tokenizer.json ]; then
  echo "-- training BPE tokenizer"
  python -m blackhole.tokenizer --input data/raw/corpus.txt --out data/tokenizer.json --vocab-size 32000
else
  echo "-- tokenizer already present"
fi

# 2. tokenize
mkdir -p data/tokenized
python - <<'PY'
from pathlib import Path
from blackhole.data import tokenize_corpus
import json
for split in ("train", "val"):
    src = Path(f"data/raw/{split}.txt")
    dst = Path(f"data/tokenized/{split}.bin")
    if src.exists() and (not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime):
        tokenize_corpus(src, dst, "data/tokenizer.json", add_eos=True)
        from blackhole.tokenizer import load_tokenizer
        tok = load_tokenizer("data/tokenizer.json")
        dt = "uint16" if tok.get_vocab_size() < 65536 else "uint32"
        Path(str(dst) + ".meta").write_text(json.dumps({"dtype": dt, "vocab": tok.get_vocab_size()}))
    elif not src.exists():
        print(f"missing {src} - run scripts/prepare_data.sh first")
PY

# 3. train
echo "-- pretraining"
python -m blackhole.train --config "$CONFIG"

# 4. evaluate
OUT_DIR=$(python -c "import json;print(json.load(open('$CONFIG')).get('out_dir','runs/blackhole'))")
python -m blackhole.evaluate --checkpoint "$OUT_DIR/ckpt_final.pt" --data data/tokenized/val.bin --out "$OUT_DIR/metrics.json" || true

# 5. sample
python -m blackhole.generate --checkpoint "$OUT_DIR/ckpt_final.pt" --prompt "Once upon a time" --max-new-tokens 100 || true
