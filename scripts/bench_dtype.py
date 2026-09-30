"""Microbenchmark: which training precision is actually fastest on THIS GPU?

Measures real tokens/sec + peak VRAM for fp32 / fp16(AMP) / bf16(AMP) on the
Black Hole small model. Run: python scripts/bench_dtype.py
"""

import gc
import time

import torch

from blackhole.config import BlackHoleConfig
from blackhole.model import BlackHole

STEPS = 5
BATCH, SEQ = 4, 1024


def bench(dtype_name, autocast_dtype=None, use_scaler=False):
    torch.cuda.empty_cache()
    gc.collect()
    cfg = BlackHoleConfig.load("configs/blackhole-small.json")
    model = BlackHole(cfg).cuda()
    if dtype_name == "fp32":
        model = model.float()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    x = torch.randint(0, cfg.vocab_size, (BATCH, SEQ), device="cuda")
    y = torch.randint(0, cfg.vocab_size, (BATCH, SEQ), device="cuda")

    def step():
        opt.zero_grad(set_to_none=True)
        if autocast_dtype is None:
            _, loss = model(x, y)
        else:
            with torch.autocast("cuda", dtype=autocast_dtype):
                _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        return float(loss.detach())

    # warmup (kernel autotune / cudnn benchmark)
    for _ in range(2):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    last = None
    for _ in range(STEPS):
        last = step()
    torch.cuda.synchronize()
    dt = time.time() - t0

    tok_s = (STEPS * BATCH * SEQ) / dt
    mem = torch.cuda.max_memory_allocated() / 1024**2
    del model, opt, scaler
    return tok_s, mem, last


if __name__ == "__main__":
    dev = torch.cuda.get_device_name(0)
    print(f"GPU: {dev} | sm_{torch.cuda.get_device_properties(0).major}{torch.cuda.get_device_properties(0).minor} | "
          f"bf16_native={torch.cuda.is_bf16_supported()}")
    print(f"model: blackhole-small (76M) | batch {BATCH}x{SEQ} | {STEPS} steps\n")
    print(f"{'mode':<28}{'tok/s':>12}{'VRAM MiB':>12}{'loss':>10}")
    results = {}
    for name, kwargs in [
        ("fp32", dict(dtype_name="fp32")),
        ("fp16 AMP + scaler", dict(dtype_name="fp16", autocast_dtype=torch.float16, use_scaler=True)),
        ("bf16 AMP (config actuelle)", dict(dtype_name="bf16", autocast_dtype=torch.bfloat16)),
    ]:
        try:
            tok_s, mem, loss = bench(**kwargs)
            results[name] = tok_s
            print(f"{name:<28}{tok_s:>12,.0f}{mem:>12,.0f}{loss:>10.3f}")
        except Exception as e:
            print(f"{name:<28}ERREUR: {type(e).__name__}: {str(e)[:70]}")

    if len(results) >= 2:
        best = max(results, key=results.get)
        print(f"\n=> le plus rapide ici: {best} ({results[best]:,.0f} tok/s)")
        for k, v in results.items():
            if k != best:
                print(f"   {k}: {results[best]/v:.2f}x plus lent")
