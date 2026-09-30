# 🕳️ Black Hole

**Black Hole** — a large language model (LLM) built **from scratch**: model architecture,
tokenizer (BPE), data pipeline, pretraining loop, inference and evaluation.

Everything here is written from zero — no `AutoModelForCausalLM` shortcut for the
architecture itself; the model is a clean decoder-only Transformer implemented in PyTorch.

---

## Architecture

Decoder-only (GPT/LLaMA-style) Transformer with modern components:

| Component        | Choice                                     |
|------------------|--------------------------------------------|
| Normalization    | RMSNorm (pre-norm, with final norm)        |
| Positional enc.  | RoPE (Rotary Position Embeddings)          |
| Attention        | Multi-Head / Grouped-Query (GQA / MQA)     |
| MLP              | SwiGLU                                     |
| Activations      | FP16/BF16 AMP, fused where possible        |
| Weight tying     | Optional embedding ↔ LM head sharing       |
| Initialization   | GPT-2 scaled residuals                     |

Two reference sizes live in `configs/`:

- `blackhole-tiny.json` — 10M params, smoke tests & CI on a single GPU.
- `blackhole-small.json` — 100M params, real training on a T4.

## Repository layout

```
black-hole/
├── configs/               # model + training hyperparameter JSON configs
├── data/                  # raw/processed/tokenized datasets (git-ignored)
├── notebooks/             # analysis & exploration
├── scripts/               # shell entrypoints (data, train, eval, export)
├── src/blackhole/
│   ├── model.py           # BlackHole model (attention, MLP, block, LM head)
│   ├── tokenizer.py       # BPE tokenizer training + encode/decode helpers
│   ├── data.py            # corpus -> tokenized memmap dataset, sampling weights
│   ├── train.py           # pretraining loop (AMP, grad accum, cosine LR, ckpt)
│   ├── generate.py        # autoregressive inference (greedy / top-k / top-p)
│   ├── evaluate.py        # val loss + zero-shot style metrics
│   ├── config.py          # dataclasses <-> JSON config loading
│   └── utils.py           # seeding, device, logging, checkpoint IO
├── tests/                 # pytest suite (shapes, gradients, determinism)
└── pyproject.toml
```

## Quickstart

```bash
pip install -e .

# 1. train the tokenizer on a corpus
python -m blackhole.tokenizer --input data/raw/corpus.txt --out data/tokenizer.json --vocab-size 32000

# 2. pretrain
python -m blackhole.train --config configs/blackhole-small.json

# 3. generate
python -m blackhole.generate --checkpoint runs/blackhole-small/ckpt_final.pt --prompt "Once upon a time"

# 4. tests
pytest -q
```

## Status

- [x] Environment (CUDA GPU, PyTorch, HF stack)
- [x] Repository + CI-ready structure
- [x] Model architecture implemented from scratch
- [x] BPE tokenizer pipeline
- [x] Data pipeline (memmap, weighted mixing)
- [x] Pretraining loop (AMP, cosine schedule, checkpointing)
- [x] Inference + evaluation scripts
- [ ] Pretrain at scale
- [ ] HF Hub release
