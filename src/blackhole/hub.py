"""Push a Black Hole checkpoint to the Hugging Face Hub.

Creates/updates a model repo and uploads:
* ``config.json``          — model hyperparameters (BlackHoleConfig)
* ``model.safetensors``    — weights
* ``tokenizer.json``       — BPE tokenizer
* ``generation_config.json``— sampling defaults
* ``README.md``            — model card

Usage:
    python -m blackhole.hub --checkpoint runs/blackhole/ckpt_final.pt \\
        --repo Ggboykxz/black-hole-tiny --private
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import BlackHoleConfig
from .generate import load_model
from .utils import get_logger, human_int, resolve_device

log = get_logger("blackhole.hub")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Upload a Black Hole checkpoint to the HF Hub")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--repo", required=True, help="e.g. Ggboykxz/black-hole-tiny")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--private", action="store_true")
    p.add_argument("--device", default="cpu", help="device for loading (cpu is fine)")
    args = p.parse_args(argv)

    from huggingface_hub import HfApi
    from safetensors.torch import save_file

    device = resolve_device(args.device)
    model, ckpt = load_model(args.checkpoint, device)
    cfg: BlackHoleConfig = model.cfg
    n = sum(x.numel() for x in model.parameters())
    log.info(f"loaded {human_int(n)} params, dim={cfg.dim} L={cfg.n_layers}")

    out = Path("hub_export")
    out.mkdir(exist_ok=True)

    # weights (cpu, fp32 for portability)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    # weight tying -> lm_head.weight and tok_emb.weight share memory; safetensors
    # forbids duplicate storage, so save one copy and drop the tied key (loading with
    # strict=False restores it, since both params point to the same tensor).
    tied_dropped = []
    if cfg.tie_weights and "lm_head.weight" in state and "tok_emb.weight" in state:
        if state["lm_head.weight"].data_ptr() == state["tok_emb.weight"].data_ptr():
            state.pop("lm_head.weight")
            tied_dropped.append("lm_head.weight")
    save_file(state, str(out / "model.safetensors"))
    if tied_dropped:
        log.info(f"weight tying: dropped {tied_dropped} from safetensors (shares tok_emb)")

    cfg.save(out / "config.json")
    (out / "generation_config.json").write_text(json.dumps({
        "bos_token_id": cfg.bos_token_id or 1,
        "eos_token_id": cfg.eos_token_id or 2,
        "pad_token_id": 0,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95,
    }, indent=2))

    tok_src = ckpt.get("tokenizer_path") or args.tokenizer
    if Path(tok_src).exists():
        Path(tok_src).replace(out / "tokenizer.json") if False else \
            (out / "tokenizer.json").write_bytes(Path(tok_src).read_bytes())

    (out / "README.md").write_text(f"""---
library_name: blackhole
tags: [llm, black-hole, from-scratch, gpt, rotary, gqa]
---

# Black Hole (tiny)

Decoder-only Transformer LM implemented from scratch (PyTorch).

| | |
|---|---|
| Layers | {cfg.n_layers} |
| Dim | {cfg.dim} |
| Heads | {cfg.n_heads} (KV {cfg.n_kv_heads}) |
| Context | {cfg.max_seq_len} |
| Vocab | {cfg.vocab_size} |
| Params | {human_int(n)} |

Load with the `blackhole` package:
```python
from blackhole.model import BlackHole
from blackhole.config import BlackHoleConfig
import json, torch
cfg = BlackHoleConfig.from_dict(json.load(open("config.json")))
model = BlackHole(cfg)
from safetensors.torch import load_file
sd = load_file("model.safetensors")
model.load_state_dict(sd, strict=False)  # lm_head tied to tok_emb -> omitted on purpose
```
""")

    api = HfApi()
    api.create_repo(args.repo, private=args.private, exist_ok=True, repo_type="model")
    url = api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="model")
    log.info(f"uploaded -> https://huggingface.co/{args.repo} ({url})")


if __name__ == "__main__":
    main()
