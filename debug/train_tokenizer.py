"""
BPE tokenizer training script.
Usage:
  uv run python debug/train_tokenizer.py --dataset tinystories
  uv run python debug/train_tokenizer.py --dataset owt
"""

import argparse
import json
import os
import time
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "data"


def save_tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_prefix: str):
    """Save vocab and merges to files."""
    vocab_path = OUTPUT_DIR / f"{output_prefix}_vocab.json"
    merges_path = OUTPUT_DIR / f"{output_prefix}_merges.txt"

    # Save vocab: {token_id: latin-1 string} for bytes round-trip
    vocab_json = {str(k): v.decode("latin-1") for k, v in vocab.items()}
    with open(vocab_path, "w") as f:
        json.dump(vocab_json, f, ensure_ascii=False)

    # Save merges: one per line, "token1 token2"
    with open(merges_path, "w", encoding="utf-8") as f:
        for t1, t2 in merges:
            f.write(t1.decode("utf-8", errors="replace") + " " + t2.decode("utf-8", errors="replace") + "\n")

    print(f"  vocab → {vocab_path}")
    print(f"  merges → {merges_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tinystories", "owt"], default="tinystories")
    args = parser.parse_args()

    from cs336_basics.bpe import train_bpe

    if args.dataset == "tinystories":
        input_path = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
        vocab_size = 10_000
        special_tokens = ["<|endoftext|>"]
        output_prefix = "tinystories"
    else:
        input_path = DATA_DIR / "owt_train.txt"
        vocab_size = 32_000
        special_tokens = ["<|endoftext|>"]
        output_prefix = "owt"

    print(f"Training BPE tokenizer on {args.dataset}")
    print(f"  input:      {input_path}  ({input_path.stat().st_size / 1e9:.2f} GB)")
    print(f"  vocab_size: {vocab_size}")
    print(f"  special:    {special_tokens}")

    t0 = time.time()
    vocab, merges = train_bpe(str(input_path), vocab_size, special_tokens)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  vocab size: {len(vocab)}")
    print(f"  merges:     {len(merges)}")

    save_tokenizer(vocab, merges, output_prefix)


if __name__ == "__main__":
    main()
