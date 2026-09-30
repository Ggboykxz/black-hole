"""Export/reload round-trip: a checkpoint must produce identical logits after save/load."""

import pytest
import torch

from blackhole.config import BlackHoleConfig
from blackhole.model import BlackHole


def _cfg(**o) -> BlackHoleConfig:
    base = dict(vocab_size=97, dim=64, max_seq_len=64, n_layers=2, n_heads=4, n_kv_heads=2)
    base.update(o)
    return BlackHoleConfig(**base)


def test_torch_checkpoint_roundtrip(tmp_path):
    from blackhole.utils import load_checkpoint, save_checkpoint

    torch.manual_seed(0)
    cfg = _cfg()
    model = BlackHole(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        ref, _ = model(x)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, None, step=7, config=cfg.to_dict())

    ck = load_checkpoint(path)
    assert ck["step"] == 7
    restored = BlackHole(BlackHoleConfig.from_dict(ck["config"]))
    restored.load_state_dict(ck["model"])
    with torch.no_grad():
        out, _ = restored(x)
    assert torch.allclose(ref, out, atol=1e-6), "checkpoint round-trip changed logits"


def test_safetensors_tied_weights_export(tmp_path):
    """The Hub export drops the tied lm_head; strict=False reload must match exactly."""
    safetensors = pytest.importorskip("safetensors.torch")
    from safetensors.torch import load_file, save_file

    torch.manual_seed(0)
    cfg = _cfg(tie_weights=True)
    model = BlackHole(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        ref, _ = model(x)

    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    # same rule as blackhole.hub: drop duplicated storage
    if state["lm_head.weight"].data_ptr() == state["tok_emb.weight"].data_ptr():
        state.pop("lm_head.weight")

    f = tmp_path / "model.safetensors"
    save_file(state, str(f))
    sd = load_file(str(f))

    restored = BlackHole(cfg)
    missing, unexpected = restored.load_state_dict(sd, strict=False)
    assert "lm_head.weight" in missing and not unexpected
    with torch.no_grad():
        out, _ = restored(x)
    assert torch.equal(ref, out), "tied-weight export must reproduce logits exactly"


def test_generate_deterministic_with_seed():
    torch.manual_seed(42)
    cfg = _cfg()
    model = BlackHole(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 6))

    torch.manual_seed(1)
    a = model.generate(x, max_new_tokens=8, temperature=1.0, top_k=None)
    torch.manual_seed(1)
    b = model.generate(x, max_new_tokens=8, temperature=1.0, top_k=None)
    assert torch.equal(a, b), "same seed must give identical generations"

    greedy = model.generate(x, max_new_tokens=8, greedy=True)
    torch.manual_seed(999)
    greedy2 = model.generate(x, max_new_tokens=8, greedy=True)
    assert torch.equal(greedy, greedy2), "greedy decoding must be deterministic"
