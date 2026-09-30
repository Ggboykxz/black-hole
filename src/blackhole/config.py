"""Configuration objects for Black Hole models and training runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BlackHoleConfig:
    """Hyperparameters of the Black Hole decoder-only Transformer."""

    # --- vocabulary / embeddings ---
    vocab_size: int = 32000
    dim: int = 768                      # residual stream width
    max_seq_len: int = 1024             # context length (RoPE is computed for this)
    tie_weights: bool = True            # share input embedding with LM head

    # --- transformer block ---
    n_layers: int = 12
    n_heads: int = 12                   # queries heads
    n_kv_heads: int = 4                 # < n_heads -> GQA, 1 -> MQA
    hidden_mult: int = 4                # SwiGLU MLP expansion (approx. 4x)
    mlp_dim: int | None = None          # explicit override; derived from dim if None
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.0

    # --- init ---
    init_std: float = 0.02
    resid_scale: float = 0.10 / (12 ** 0.5)  # GPT-2 style scaled residual init

    # --- generation defaults ---
    eos_token_id: int | None = None
    bos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.mlp_dim is None:
            # SwiGLU with hidden_mult=4 keeps param count close to a 4x FFN
            hidden = int(self.hidden_mult * self.dim * 2 / 3)
            self.mlp_dim = 8 * ((hidden + 7) // 8)  # round to multiple of 8 (tensor cores)

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlackHoleConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load(cls, path: str | Path) -> "BlackHoleConfig":
        data = json.loads(Path(path).read_text())
        # tolerate configs that nest the model under a "model" key
        if "model" in data and isinstance(data["model"], dict):
            data = data["model"]
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


@dataclass
class TrainConfig:
    """Hyperparameters of a pretraining run."""

    # --- data ---
    train_data: str = "data/tokenized/train.bin"
    val_data: str = "data/tokenized/val.bin"
    tokenizer_path: str = "data/tokenizer.json"

    # --- optimization ---
    batch_size: int = 16                # sequences per micro-step (per device)
    grad_accum_steps: int = 1
    seq_len: int = 1024
    max_steps: int = 1000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- system ---
    model_config: str = "configs/blackhole-small.json"
    out_dir: str = "runs/blackhole"
    seed: int = 1337
    device: str = "cuda"
    dtype: str = "bf16"                 # bf16 | fp16 | fp32
    compile: bool = False
    log_every: int = 10
    eval_every: int = 200
    eval_iters: int = 20
    save_every: int = 500
    resume: str | None = None

    # --- logging ---
    wandb_project: str | None = None    # set to enable Weights & Biases
    wandb_run_name: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        data = json.loads(Path(path).read_text())
        if "train" in data and isinstance(data["train"], dict):
            merged = {**data["train"]}
        else:
            merged = dict(data)
        known = set(cls.__dataclass_fields__)
        unknown = set(merged) - known
        extra = merged.pop("extra", {})
        if unknown:
            # keep unknown keys rather than failing (forward-compat)
            extra = {**extra, **{k: merged.pop(k) for k in unknown}}
        cfg = cls(**merged)
        cfg.extra = extra
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
