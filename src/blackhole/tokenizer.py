"""BPE tokenizer: train from a corpus, save/load, encode/decode.

Trains a byte-level BPE tokenizer with the ``tokenizers`` library (HF), which is
written in Rust and fast enough for multi-GB corpora. A tiny pure-Python fallback
is not provided on purpose — one implementation, well tested.

Byte-level BPE guarantees any input can be encoded (no <unk> holes), which matters
for code, math and non-Latin scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

__all__ = ["train_tokenizer", "load_tokenizer", "BH_SPECIAL_TOKENS"]


# special tokens shared across the project
BH_SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"]
PAD, BOS, EOS, UNK, MASK = range(5)


def train_tokenizer(
    files: Iterable[str | Path],
    out_path: str | Path,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    show_progress: bool = True,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer on one or more text files and save it."""
    files = [str(f) for f in files]
    for f in files:
        if not Path(f).exists():
            raise FileNotFoundError(f"corpus file not found: {f}")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=BH_SPECIAL_TOKENS,
        show_progress=show_progress,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train(files, trainer)

    # make bos/eos explicit post-processing so decoding round-trips
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A",
        pair="<bos> $A <bos> $B",
        special_tokens=[("<bos>", bos_id)],
    )
    # ensure eos at the end of single sequences too -> handled by callers
    _ = eos_id

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"tokenizer not found: {path}")
    return Tokenizer.from_file(str(path))


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer for Black Hole")
    p.add_argument("--input", nargs="+", required=True, help="corpus text file(s)")
    p.add_argument("--out", required=True, help="output tokenizer.json path")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--min-frequency", type=int, default=2)
    args = p.parse_args(argv)

    tok = train_tokenizer(args.input, args.out, args.vocab_size, args.min_frequency)
    print(f"tokenizer trained: vocab={tok.get_vocab_size()} -> {args.out}")

    # smoke round-trip
    ids = tok.encode("Black Hole is a language model.", add_special_tokens=False).ids
    print("round-trip:", ids[:12], "->", repr(tok.decode(ids)))


if __name__ == "__main__":
    main()
