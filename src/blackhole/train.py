"""Black Hole pretraining loop.

Features:
* AMP autocast (bf16/fp16) with dynamic loss scaling for fp16
* gradient accumulation + global-norm clipping
* AdamW with weight decay applied only to 2D+ params (no decay on biases/norms/embeddings)
* linear warmup + cosine decay to min_lr
* periodic validation, checkpointing (resume-safe), throughput logging
* optional Weights & Biases and torch.compile

Single-process by design for this environment (1 GPU); DDP is a drop-in via
``torchrun --nproc_per_node=N -m blackhole.train`` if more GPUs appear.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import torch

from .config import BlackHoleConfig, TrainConfig
from .data import Batcher
from .model import BlackHole
from .utils import (
    Throughput,
    cosine_decay_lr,
    count_params,
    get_logger,
    gpu_report,
    human_int,
    load_checkpoint,
    resolve_device,
    resolve_dtype,
    save_checkpoint,
    seed_everything,
)

log = get_logger("blackhole.train")


# --------------------------------------------------------------------------- optimizer
def build_optimizer(model: BlackHole, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with decay on matrices, no decay on biases / norms / 1D params."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=1e-8, fused=torch.cuda.is_available()
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pretrain a Black Hole model")
    parser.add_argument("--config", default="configs/blackhole-small.json")
    parser.add_argument("--max-steps", type=int, default=None, help="override config max_steps")
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    parser.add_argument("--init-from", default=None,
                        help="warm start: load MODEL WEIGHTS ONLY (fresh optimizer + LR schedule). "
                             "Use when the data or schedule changes; use --resume to continue exactly.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="build model, 1 step, exit")
    args = parser.parse_args(argv)

    t0 = time.time()

    # ---------------------------------------------------------------- config
    train_cfg = TrainConfig.load(args.config)
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    if args.out_dir is not None:
        train_cfg.out_dir = args.out_dir
    if args.resume:
        train_cfg.resume = args.resume
    if args.init_from:
        train_cfg.extra["init_from"] = args.init_from
    if args.dry_run:
        train_cfg.max_steps = 1
        train_cfg.eval_every = 10**9
        train_cfg.save_every = 10**9

    seed_everything(train_cfg.seed)
    device = resolve_device(train_cfg.device)
    dtype = resolve_dtype(train_cfg.dtype)
    use_amp = dtype != torch.float32 and device.type == "cuda"

    log.info(f"device={device} ({gpu_report()}) dtype={dtype}")

    # ---------------------------------------------------------------- model
    model_cfg = BlackHoleConfig.load(train_cfg.model_config)
    model = BlackHole(model_cfg).to(device)
    log.info(
        f"Black Hole params: {human_int(count_params(model))} non-emb "
        f"| dim={model_cfg.dim} layers={model_cfg.n_layers} heads={model_cfg.n_heads}/"
        f"{model_cfg.n_kv_heads} kv | vocab={model_cfg.vocab_size} | ctx={model_cfg.max_seq_len}"
        f" | tied={model_cfg.tie_weights}"
    )

    optimizer = build_optimizer(model, train_cfg)

    # ---------------------------------------------------------------- resume / warm start
    start_step = 0
    init_from = train_cfg.extra.get("init_from")
    if init_from:
        # weights only: the LR schedule and optimizer state start fresh, which is what
        # you want when switching to a new/bigger corpus (a resumed cosine schedule
        # would already be at min_lr and could not train further).
        ckpt = load_checkpoint(init_from, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in {init_from}: {unexpected}")
        missing = [k for k in missing if not k.endswith("lm_head.weight")]  # tied
        if missing:
            raise RuntimeError(f"missing keys in {init_from}: {missing}")
        log.info(f"warm start from {init_from} (weights only, fresh optimizer/schedule)")
    elif train_cfg.resume:
        ckpt = load_checkpoint(train_cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        log.info(f"resumed from {train_cfg.resume} at step {start_step}")

    # ---------------------------------------------------------------- data
    batcher = Batcher(
        train_cfg.train_data,
        train_cfg.val_data,
        batch_size=train_cfg.batch_size,
        seq_len=train_cfg.seq_len,
        device=device,
        seed=train_cfg.seed,
    )
    log.info(
        f"data: train={batcher.train.size:,} tok"
        f" val={0 if batcher.val is None else batcher.val.size:,} tok"
        f" | batch={train_cfg.batch_size}x{train_cfg.seq_len} "
        f"grad_accum={train_cfg.grad_accum_steps} "
        f"(global batch {train_cfg.batch_size * train_cfg.grad_accum_steps}x{train_cfg.seq_len})"
    )

    # ---------------------------------------------------------------- amp
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    scaler = torch.amp.GradScaler(enabled=(use_amp and dtype == torch.float16))

    @contextlib.contextmanager
    def autocast_ctx():
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=dtype):
                yield
        else:
            yield

    if train_cfg.compile:
        # NB: mode='default', NOT 'reduce-overhead'. CUDA graphs break with gradient
        # accumulation (several forwards per backward -> "tensor overwritten by a
        # subsequent run"). 'default' still gives ~1.4x on this GPU.
        log.info("torch.compile enabled (first steps will be slow)")
        model.forward = torch.compile(model.forward, mode="default")  # type: ignore[method-assign]

    # ---------------------------------------------------------------- wandb (optional)
    wandb_run = None
    if train_cfg.wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=train_cfg.wandb_project,
                name=train_cfg.wandb_run_name,
                config={**train_cfg.to_dict(), **model_cfg.to_dict()},
            )
            log.info("wandb logging enabled")
        except Exception as exc:  # never let logging kill a run
            log.warning(f"wandb disabled ({exc})")

    # ---------------------------------------------------------------- run loop
    out_dir = Path(train_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_config.json").write_text(json.dumps(model_cfg.to_dict(), indent=2))
    (out_dir / "train_config.json").write_text(json.dumps(train_cfg.to_dict(), indent=2))

    model.train()
    throughput = Throughput(device)
    ema_loss = None
    tokens_seen = 0
    step = start_step
    best_val = float("inf")

    micro_total = train_cfg.grad_accum_steps
    while step < train_cfg.max_steps:
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro in range(micro_total):
            x, y = batcher.next_train()
            with autocast_ctx():
                _, loss = model(x, y)
                loss = loss / micro_total
            scaler.scale(loss).backward()
            # `loss` is already divided by micro_total, so summing the scaled values
            # across micro-steps yields the mean raw loss (do NOT re-multiply).
            step_loss += float(loss.detach())
            tokens_seen += x.numel()
            throughput.update(x.numel())

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        lr = cosine_decay_lr(step, train_cfg.max_steps, train_cfg.lr, train_cfg.min_lr, train_cfg.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.step(optimizer)
        scaler.update()
        step += 1

        # ---- logging
        ema_loss = step_loss if ema_loss is None else 0.95 * ema_loss + 0.05 * step_loss
        if step % train_cfg.log_every == 0 or step == 1:
            tps = throughput.rate()
            ppl = torch.exp(torch.tensor(min(ema_loss, 20.0))).item()
            msg = (
                f"step {step}/{train_cfg.max_steps} | loss {ema_loss:.4f} | ppl {ppl:.2f} "
                f"| lr {lr:.2e} | grad_norm {float(grad_norm):.2f} "
                f"| {tps/1e3:.1f}k tok/s | {human_int(tokens_seen)} tok seen"
            )
            log.info(msg)
            if wandb_run is not None:
                wandb_run.log({"train/loss": ema_loss, "train/lr": lr,
                               "train/grad_norm": float(grad_norm),
                               "train/tok_s": tps, "step": step})
            throughput.reset()

        # ---- validation
        if step % train_cfg.eval_every == 0:
            val_loss = batcher.estimate_val_loss(model, train_cfg.eval_iters, autocast_ctx)
            if val_loss == val_loss:  # not NaN
                vppl = float(torch.exp(torch.tensor(min(val_loss, 20.0))))
                log.info(f"step {step} | val loss {val_loss:.4f} | val ppl {vppl:.2f}")
                if wandb_run is not None:
                    wandb_run.log({"val/loss": val_loss, "val/ppl": vppl, "step": step})
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(out_dir / "ckpt_best.pt", model, optimizer, step=step,
                                    config=model_cfg.to_dict(), tokenizer_path=train_cfg.tokenizer_path,
                                    extra={"val_loss": val_loss})
            model.train()

        # ---- checkpoint
        if step % train_cfg.save_every == 0:
            path = out_dir / f"ckpt_step{step}.pt"
            save_checkpoint(path, model, optimizer, step=step, config=model_cfg.to_dict(),
                            tokenizer_path=train_cfg.tokenizer_path)
            save_checkpoint(out_dir / "ckpt_last.pt", model, optimizer, step=step,
                            config=model_cfg.to_dict(), tokenizer_path=train_cfg.tokenizer_path)
            log.info(f"checkpoint -> {path}")

        if args.dry_run:
            break

    # ---------------------------------------------------------------- final save
    save_checkpoint(out_dir / "ckpt_final.pt", model, optimizer, step=step,
                    config=model_cfg.to_dict(), tokenizer_path=train_cfg.tokenizer_path,
                    extra={"tokens_seen": tokens_seen, "final_loss": ema_loss})
    elapsed = time.time() - t0
    log.info(f"done: {step} steps | {human_int(tokens_seen)} tokens | {elapsed/60:.1f} min "
             f"| final loss {ema_loss if ema_loss else float('nan'):.4f} -> {out_dir}/ckpt_final.pt")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
