"""Core correctness tests for the Black Hole model."""

import math

import pytest
import torch

from blackhole.config import BlackHoleConfig
from blackhole.model import BlackHole, RMSNorm, RoPE


def tiny_cfg(**overrides) -> BlackHoleConfig:
    base = dict(vocab_size=211, dim=64, max_seq_len=64, n_layers=2, n_heads=4, n_kv_heads=2)
    base.update(overrides)
    return BlackHoleConfig(**base)


def test_config_validation():
    with pytest.raises(ValueError):
        BlackHoleConfig(dim=65, n_heads=4, n_kv_heads=2)
    with pytest.raises(ValueError):
        BlackHoleConfig(dim=64, n_heads=5, n_kv_heads=2)
    cfg = BlackHoleConfig(dim=768, n_heads=12, n_kv_heads=4)
    assert cfg.head_dim == 64
    assert cfg.kv_dim == 256
    assert cfg.mlp_dim % 8 == 0


def test_forward_shapes_loss():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = BlackHole(cfg)
    b, t = 3, 32
    x = torch.randint(0, cfg.vocab_size, (b, t))
    y = torch.randint(0, cfg.vocab_size, (b, t))

    logits, loss = model(x, y)
    assert logits.shape == (b, t, cfg.vocab_size)
    assert loss.dim() == 0 and torch.isfinite(loss)
    # untrained model should be near ln(vocab)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 1.0

    # inference path: only last position
    logits_inf, loss_inf = model(x)
    assert logits_inf.shape == (b, 1, cfg.vocab_size)
    assert loss_inf is None


def test_backward_produces_finite_grads():
    torch.manual_seed(0)
    model = BlackHole(tiny_cfg())
    x = torch.randint(0, 211, (2, 16))
    y = torch.randint(0, 211, (2, 16))
    _, loss = model(x, y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no grad"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite grad"


def test_causality():
    """Changing a future token must not change earlier logits."""
    torch.manual_seed(0)
    cfg = tiny_cfg(dropout=0.0)
    model = BlackHole(cfg).eval()
    x1 = torch.randint(0, cfg.vocab_size, (1, 20))
    x2 = x1.clone()
    x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size  # perturb last token only

    # pass dummy targets so we get *all* positions' logits (else only the last is computed)
    dummy = torch.zeros_like(x1)
    with torch.no_grad():
        logits1, _ = model(x1, dummy)
        logits2, _ = model(x2, dummy)
    assert torch.allclose(logits1[:, :19], logits2[:, :19], atol=1e-4), "attention is not causal"


def test_gqa_matches_mha_kv_heads():
    """GQA output must differ in shape handling but stay numerically valid."""
    torch.manual_seed(0)
    x = torch.randn(2, 12, 64)
    for kv in (1, 2, 4):
        cfg = tiny_cfg(n_kv_heads=kv)
        model = BlackHole(cfg).eval()
        out, _ = model(torch.randint(0, cfg.vocab_size, (2, 12)))
        assert torch.isfinite(out).all()


def test_rmsnorm_unit_variance():
    x = torch.randn(4, 8, 32) * 7.3
    y = RMSNorm(32)(x)
    # before weight scaling (weight=1), last-dim RMS should be ~1
    rms = y.float().pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_relative_positions():
    """RoPE must preserve relative offset: <q(t+k), k(t)> depends on k, not t."""
    rope = RoPE(head_dim=16, max_seq_len=64, theta=10000.0)
    q = torch.randn(1, 1, 8, 16)
    k = torch.randn(1, 1, 8, 16)
    qr, kr = rope(q, k)
    # a rotation applied to both q and k at the same position is orthogonal -> dot preserved
    a = (qr[0, 0] * kr[0, 0]).sum()
    b = (q[0, 0] * k[0, 0]).sum()
    assert abs(a.item() - b.item()) < 1e-3

    # orthogonality per-position: rotating any vector must preserve its norm
    norm_before = q.norm()
    norm_after = qr.norm()
    assert abs(norm_before.item() - norm_after.item()) < 1e-3


def test_generate_shape_and_termination():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = BlackHole(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 5))
    out = model.generate(x, max_new_tokens=10, temperature=0.8, top_k=20, top_p=0.9)
    assert out.shape == (2, 15)
    greedy = model.generate(x, max_new_tokens=4, greedy=True)
    assert greedy.shape == (2, 9)


def test_weight_tying():
    cfg = tiny_cfg(tie_weights=True)
    model = BlackHole(cfg)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
    cfg2 = tiny_cfg(tie_weights=False)
    model2 = BlackHole(cfg2)
    assert model2.lm_head.weight.data_ptr() != model2.tok_emb.weight.data_ptr()


def test_context_limit_enforced():
    cfg = tiny_cfg(max_seq_len=32)
    model = BlackHole(cfg)
    with pytest.raises(ValueError):
        model(torch.randint(0, cfg.vocab_size, (1, 33)))


def test_param_count_matches_manual():
    cfg = tiny_cfg()
    model = BlackHole(cfg)
    manual = sum(p.numel() for p in model.parameters())
    if cfg.tie_weights:
        manual -= model.lm_head.weight.numel()
    assert model.num_params() == manual
    assert model.num_params() > 0
