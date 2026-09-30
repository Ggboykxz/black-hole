"""Autoregressive text generation from a Black Hole checkpoint."""

from __future__ import annotations

import argparse

import torch

from .config import BlackHoleConfig
from .model import BlackHole
from .tokenizer import BOS, load_tokenizer
from .utils import get_logger, human_int, load_checkpoint, resolve_device, seed_everything

log = get_logger("blackhole.generate")


def load_model(
    checkpoint: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[BlackHole, dict]:
    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    cfg_dict = ckpt.get("config")
    if cfg_dict is None:
        raise ValueError("checkpoint missing model config; pass --model-config")
    cfg = BlackHoleConfig.from_dict(cfg_dict)
    model = BlackHole(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, ckpt


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate text with Black Hole")
    p.add_argument("--checkpoint", default="runs/blackhole/ckpt_final.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-config", default=None, help="only if checkpoint lacks config")
    args = p.parse_args(argv)

    seed = args.seed if args.seed is not None else 1337
    seed_everything(seed)
    device = resolve_device(args.device)

    if args.model_config:
        cfg = BlackHoleConfig.load(args.model_config)
        ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
        model = BlackHole(cfg)
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device).eval()
        tokenizer_path = ckpt.get("tokenizer_path") or args.tokenizer
    else:
        model, ckpt = load_model(args.checkpoint, device)
        tokenizer_path = ckpt.get("tokenizer_path") or args.tokenizer

    n = sum(p.numel() for p in model.parameters())
    log.info(f"loaded {human_int(n)} params from {args.checkpoint} on {device}")

    tok = load_tokenizer(tokenizer_path)
    eos_id = tok.token_to_id("<eos>")

    ids = tok.encode(args.prompt, add_special_tokens=False).ids
    ids = [BOS] + ids
    x = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(
        x,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
        eos_token_id=eos_id,
    )
    text = tok.decode(out[0].tolist())
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
