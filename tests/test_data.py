"""Tests for tokenizer training, data pipeline and training smoke path."""

import numpy as np
import pytest
import torch

from blackhole.data import get_batch, load_tokens, tokenize_corpus
from blackhole.tokenizer import train_tokenizer


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    p = tmp_path_factory.mktemp("corpus") / "corpus.txt"
    p.write_text(
        "The black hole at the center of the galaxy bends light and time.\n"
        "A language model predicts the next token given the previous ones.\n"
        "Machine learning, transformers, attention, and gradient descent.\n" * 400,
        encoding="utf-8",
    )
    return p


@pytest.fixture(scope="module")
def tokenizer(corpus, tmp_path_factory):
    out = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    return train_tokenizer([corpus], out, vocab_size=1000, min_frequency=1, show_progress=False)


def test_tokenizer_roundtrip(tokenizer):
    text = "The black hole bends light."
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    assert len(ids) > 0
    assert tokenizer.decode(ids) == text
    # BPE may stop early if the corpus runs out of merges, but we expect a real vocab
    assert tokenizer.get_vocab_size() >= 300


def test_tokenizer_covers_arbitrary_bytes(tokenizer):
    weird = "中文 ✓ 𝔘𝔫𝔦 coding() {} 0x1F"
    ids = tokenizer.encode(weird, add_special_tokens=False).ids
    assert tokenizer.decode(ids) == weird


def test_tokenize_corpus_and_load(corpus, tokenizer, tmp_path):
    tok_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tok_path))
    bin_path = tmp_path / "train.bin"
    n = tokenize_corpus(corpus, bin_path, tok_path, add_eos=True)
    assert n > 1000
    data = load_tokens(bin_path)
    assert data.dtype == np.uint16
    assert data.size == n


def test_get_batch_shapes(tmp_path):
    rng = np.random.default_rng(0)
    arr = np.arange(5000, dtype=np.uint16)
    x, y = get_batch(arr, batch_size=4, seq_len=64, device=torch.device("cpu"), rng=rng)
    assert x.shape == (4, 64) and y.shape == (4, 64)
    assert (y[:, :-1] == x[:, 1:]).all(), "y must be x shifted by one"
    assert x.dtype == torch.int64


def test_get_batch_too_small():
    arr = np.arange(10, dtype=np.uint16)
    with pytest.raises(ValueError):
        get_batch(arr, 1, 32, torch.device("cpu"), np.random.default_rng(0))


def test_train_smoke(tmp_path, corpus):
    """One optimizer step on a tiny model must reduce nothing crazy (just runs)."""
    from blackhole.config import BlackHoleConfig
    from blackhole.model import BlackHole

    tok_path = tmp_path / "tokenizer.json"
    train_tokenizer([corpus], tok_path, vocab_size=1000, min_frequency=1, show_progress=False)
    bin_path = tmp_path / "train.bin"
    tokenize_corpus(corpus, bin_path, tok_path)
    data = load_tokens(bin_path)

    cfg = BlackHoleConfig(vocab_size=1000, dim=64, max_seq_len=64, n_layers=2, n_heads=4, n_kv_heads=2)
    model = BlackHole(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    rng = np.random.default_rng(0)
    x, y = get_batch(data, 2, 64, torch.device("cpu"), rng)
    _, loss = model(x, y)
    before = float(loss.detach())
    loss.backward()
    opt.step()

    x, y = get_batch(data, 2, 64, torch.device("cpu"), rng)
    _, loss2 = model(x, y)
    assert np.isfinite(before) and np.isfinite(float(loss2.detach()))
