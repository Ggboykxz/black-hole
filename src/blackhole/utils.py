"""Shared utilities: seeding, device selection, logging, checkpoint IO."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


# --------------------------------------------------------------------------- logging
def get_logger(name: str = "blackhole") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------- reproducibility
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- device / dtype
def resolve_device(device: str = "cuda") -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    table = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
             "fp16": torch.float16, "float16": torch.float16,
             "fp32": torch.float32, "float32": torch.float32}
    if name not in table:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(table)}")
    return table[name]


def count_params(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def human_int(n: float) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000 or unit == "T":
            return f"{n:.2f}{unit}" if unit else f"{int(n)}"
        n /= 1000.0
    return f"{n:.2f}T"


# --------------------------------------------------------------------------- LR schedule
def cosine_decay_lr(step: int, total: int, lr: float, min_lr: float, warmup: int) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


# --------------------------------------------------------------------------- throughput
class Throughput:
    """Tokens/second meter."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device
        self.start = time.time()
        self.tokens = 0
        self._cuda_sync = device is not None and device.type == "cuda"

    def update(self, n_tokens: int) -> None:
        self.tokens += n_tokens

    def rate(self) -> float:
        if self._cuda_sync:
            torch.cuda.synchronize()
        elapsed = max(1e-6, time.time() - self.start)
        return self.tokens / elapsed

    def reset(self) -> None:
        self.start = time.time()
        self.tokens = 0


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    t0 = time.time()
    try:
        yield
    finally:
        msg = f"{label}: {time.time() - t0:.2f}s"
        (logger or get_logger()).info(msg)


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    step: int = 0,
    config: dict[str, Any] | None = None,
    tokenizer_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "step": step,
        "config": config,
        "tokenizer_path": tokenizer_path,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic-ish write
    meta = path.with_suffix(".json")
    meta.write_text(json.dumps({"step": step, "config": config, "extra": extra or {}}, indent=2))
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def gpu_report() -> str:
    if not torch.cuda.is_available():
        return "cuda: unavailable"
    out = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        out.append(
            f"gpu{i} {props.name} {total // 1024**2}MiB "
            f"(free {free // 1024**2}MiB, sm_{props.major}{props.minor})"
        )
    return " | ".join(out)
