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
- [ ] Scale up: more data, bigger config, longer run

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
| | | | | | |
