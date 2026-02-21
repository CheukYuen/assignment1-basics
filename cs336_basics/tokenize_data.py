"""Tokenize raw text files and save as numpy uint16 arrays.

Usage:
    uv run python cs336_basics/tokenize_data.py \
        --vocab data/tinystories_vocab.json \
        --merges data/tinystories_merges.txt \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --output data/tinystories_train.npy

    uv run python cs336_basics/tokenize_data.py \
        --vocab data/tinystories_vocab.json \
        --merges data/tinystories_merges.txt \
        --input data/TinyStoriesV2-GPT4-valid.txt \
        --output data/tinystories_valid.npy
"""

from __future__ import annotations

import argparse

import numpy as np

from cs336_basics.bpe import Tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", required=True)
    p.add_argument("--merges", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--special_tokens", nargs="*", default=["<|endoftext|>"])
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=args.special_tokens)

    # Count total lines for progress bar
    print(f"Counting lines in {args.input} ...")
    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        total_lines = sum(1 for _ in f)
    print(f"Total lines: {total_lines:,}")

    # Stream tokens into a growing list, flushing to disk periodically
    CHUNK = 10_000_000  # flush every 10M tokens
    tmp_path = args.output + ".tmp"
    all_tokens: list[int] = []
    total_tokens = 0
    last_report = 0

    print(f"Tokenizing {args.input} ...")
    import time
    t0 = time.time()

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            all_tokens.extend(tokenizer.encode(line))

            # Progress report every 100k lines
            if line_no - last_report >= 100_000:
                elapsed = time.time() - t0
                pct = line_no / total_lines * 100
                rate = line_no / elapsed
                eta = (total_lines - line_no) / rate if rate > 0 else 0
                print(
                    f"  {pct:5.1f}% | {line_no:,}/{total_lines:,} lines"
                    f" | {len(all_tokens):,} tokens"
                    f" | {rate:.0f} lines/s"
                    f" | ETA {eta/60:.1f} min",
                    flush=True,
                )
                last_report = line_no

    arr = np.array(all_tokens, dtype=np.uint16)
    np.save(args.output, arr)
    elapsed = time.time() - t0
    print(f"Done! Saved {len(arr):,} tokens → {args.output}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
