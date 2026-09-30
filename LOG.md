# 🕳️ Black Hole — training log

## Environment

| Item | Value |
|---|---|
| GPU | Tesla T4, 15 GB VRAM, sm_75 |
| CUDA | 13.0 driver / torch 2.11.0+cu128 |
| Python | 3.13.15 |
| Stack | transformers 5.16, datasets 4.8, tokenizers 0.23, accelerate 1.14 |
| Disk | 65 GB free |

## Milestones

- [x] Env bootstrap + GitHub repo (`Ggboykxz/black-hole`)
- [x] Model from scratch: RMSNorm, RoPE (orthogonality bug found & fixed by tests), GQA/MQA, SwiGLU
- [x] BPE tokenizer (byte-level, `tokenizers`)
- [x] Data pipeline: text → `uint16` memmap → random blocks
- [x] Training loop: AMP bf16, grad accum, cosine LR, AdamW (decoupled decay), ckpt/resume
- [x] 17/17 tests green (causality, shapes, grads, RoPE, GQA, round-trip)
- [x] HF Hub export (`Ggboykxz/black-hole-tiny`) incl. tied-weight safetensors handling
- [x] 20/20 tests green (+ checkpoint & safetensors round-trip, generation determinism)
- [x] Real corpus: FineWeb-Edu 4.2GB -> filtered/deduped -> 131.9M tokens train + 661k val
- [x] Real BPE tokenizer, vocab 32000 (was 516 on synthetic data)
- [x] fp16 AMP + fixed grad-accum loss logging (was 4x too high)
- [x] Pretraining on 100M-param model (blackhole-small) launched
- [ ] Scale up: more data, longer run, bigger config

## Notes / decisions

- **RoPE cache layout**: must use `cat(freqs, freqs)` (not `stack`) so index `i` and
  `i + head_dim/2` share a frequency — matches `rotate_half` and keeps rotations
  orthogonal. Caught by `test_rope_relative_positions`.
- **Inference path** computes logits for the last position only (memory + speed);
  full logits are produced when `targets` are passed (used in eval for accuracy).
- Vocab set to tokenizer reality (516) rather than a nominal 32000 — avoids 95% of
  the embedding matrix being dead weight while the corpus grows.

## Run history

| Run | Config | Steps | Tokens | Loss | Notes |
|---|---|---|---|---|---|
| smoke | train-tiny | 1 | 4.1K | 6.34 | ≈ ln(516)=6.25 ✓ sanity |
| bh-001 | train-tiny | 400 | 1.64M | 0.665 | val 0.661 / ppl 1.94 / next-token acc 0.728, 24s on T4 |

**bh-002 (en cours)** — vrai corpus FineWeb-Edu, 100M params, fp16, batch effectif 32x1024 :
loss initiale 10.498 ≈ ln(32000)=10.37 ✓, val 6.18 @250 pas.

bh-001 generation sample (prompt "The black hole"):

> The black hole learns the curvature of space across the void of space.
> The galaxy observes attention weights one layer at a time.
