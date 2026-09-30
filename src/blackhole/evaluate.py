"""Evaluation: validation cross-entropy/perplexity + zero-shot style probes.

Reports:
* val loss / perplexity over N random blocks (comparable across runs)
* greedy next-token accuracy
* optional external test set (a separate tokenized .bin)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import get_batch, load_tokens
from .generate import load_model
from .utils import get_logger, resolve_device, resolve_dtype, seed_everything

log = get_logger("blackhole.evaluate")


@torch.no_grad()
def evaluate(
    model,
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    num_batches: int,
    seed: int = 1234,
) -> dict:
    rng = np.random.default_rng(seed)
    model.eval()
    losses, accs = [], []
    for _ in range(num_batches):
        x, y = get_batch(data, batch_size, seq_len, device, rng)
        logits, loss = model(x, y)
        losses.append(float(loss))
        preds = logits.float().argmax(dim=-1)
        accs.append(float((preds == y).float().mean()))
    model.train()
    loss = float(np.mean(losses))
    return {
        "loss": loss,
        "perplexity": float(torch.exp(torch.tensor(min(loss, 30.0)))),
        "next_token_accuracy": float(np.mean(accs)),
        "batches": num_batches,
        "tokens_evaluated": num_batches * batch_size * seq_len,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a Black Hole checkpoint")
    p.add_argument("--checkpoint", default="runs/blackhole/ckpt_final.pt")
    p.add_argument("--data", default="data/tokenized/val.bin")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="fp32")
    p.add_argument("--out", default=None, help="write metrics JSON here")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    seed_everything(1234)
    model, _ = load_model(args.checkpoint, device, resolve_dtype(args.dtype))
    data = load_tokens(args.data)

    metrics = evaluate(model, data, args.batch_size, args.seq_len, device, args.batches)
    metrics["checkpoint"] = args.checkpoint
    metrics["data"] = args.data
    log.info(json.dumps(metrics, indent=2))
    print(f"\nloss={metrics['loss']:.4f}  ppl={metrics['perplexity']:.2f}  "
          f"next-token acc={metrics['next_token_accuracy']:.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2))
        log.info(f"metrics -> {args.out}")


if __name__ == "__main__":
    main()
