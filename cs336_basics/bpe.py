"""
Byte-Pair Encoding (BPE) Tokenizer Implementation

This module implements BPE tokenization from scratch, including:
1. Pre-tokenization using GPT-2 regex pattern
2. BPE training algorithm with frequency-based merging
3. Tokenizer class for encoding and decoding text
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from typing import BinaryIO, Iterable, Iterator

# GPT-2 pre-tokenization pattern (requires `regex` package for Unicode properties)
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _get_pair_counts(words: dict[tuple[int, ...], int]) -> Counter:
    """
    Efficiently count all adjacent pairs in the word dictionary.

    Args:
        words: Dictionary mapping word (tuple of token IDs) to frequency

    Returns:
        Counter of pairs to their total frequency
    """
    pairs = Counter()
    for word, freq in words.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def _merge_word(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    """
    Merge a specific pair in a word.

    Args:
        word: Tuple of token IDs
        pair: Pair to merge (a, b)
        new_id: New token ID for merged pair

    Returns:
        New word with all occurrences of pair merged
    """
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            new_word.append(new_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def _pretokenize_chunk(args: tuple[str, str | None]) -> Counter:
    """
    Pre-tokenize a text chunk and count byte-level token frequencies.

    Args:
        args: (text_chunk, special_tokens_pattern)

    Returns:
        Counter of byte tokens to their frequencies
    """
    text, special_pattern = args
    counts = Counter()

    # If we have special tokens, split on them first
    if special_pattern:
        segments = re.split(f"({special_pattern})", text)
        for segment in segments:
            if segment and not re.fullmatch(special_pattern, segment):
                # Apply GPT-2 pre-tokenization to non-special segments
                import regex
                for match in regex.finditer(GPT2_SPLIT_PATTERN, segment):
                    token = match.group().encode("utf-8")
                    counts[token] += 1
    else:
        # No special tokens, just pre-tokenize
        import regex
        for match in regex.finditer(GPT2_SPLIT_PATTERN, text):
            token = match.group().encode("utf-8")
            counts[token] += 1

    return counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on a text corpus.

    Args:
        input_path: Path to training corpus (text file)
        vocab_size: Target vocabulary size (including special tokens)
        special_tokens: List of special tokens to add to vocabulary

    Returns:
        vocab: Mapping from token ID to bytes
        merges: List of merge operations in order
    """
    special_tokens = special_tokens or []

    # Step 1: Read corpus and pre-tokenize
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Build special token pattern for splitting
    # Sort by length (longest first) for greedy matching
    special_pattern = None
    if special_tokens:
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        escaped_tokens = [re.escape(token) for token in sorted_tokens]
        special_pattern = "|".join(escaped_tokens)

    token_counts = _pretokenize_chunk((text, special_pattern))

    # Step 2: Initialize vocabulary with 256 byte values
    vocab_bytes = {i: bytes([i]) for i in range(256)}
    next_token_id = 256

    # Add special tokens to vocab
    for special_token in special_tokens:
        vocab_bytes[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1

    # Calculate number of merges needed
    num_merges = vocab_size - next_token_id

    # Step 3: Convert to word dictionary with token IDs
    # Each byte becomes a token ID (0-255)
    words = {}
    for token_bytes, count in token_counts.items():
        word = tuple(token_bytes)  # Each element is a byte value (0-255)
        words[word] = count

    # Step 4: Perform BPE merges
    merges = []

    for _ in range(num_merges):
        # Count all pairs
        pair_counts = _get_pair_counts(words)

        if not pair_counts:
            break

        # Find most frequent pair (lexicographic tie-breaking on byte sequences)
        max_count = max(pair_counts.values())
        # For tie-breaking, compare pairs based on their byte representation
        candidates = [(pair, count) for pair, count in pair_counts.items() if count == max_count]
        best_pair = max(candidates, key=lambda x: (vocab_bytes[x[0][0]], vocab_bytes[x[0][1]]))[0]

        # Create new token ID
        new_token_id = next_token_id
        next_token_id += 1

        # Store merge (as bytes objects)
        vocab_bytes[new_token_id] = vocab_bytes[best_pair[0]] + vocab_bytes[best_pair[1]]
        merges.append((vocab_bytes[best_pair[0]], vocab_bytes[best_pair[1]]))

        # Update all words by merging the pair
        words = {_merge_word(word, best_pair, new_token_id): freq for word, freq in words.items()}

    return vocab_bytes, merges


class Tokenizer:
    """
    BPE Tokenizer for encoding and decoding text.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Initialize tokenizer from vocabulary and merges.

        Args:
            vocab: Mapping from token ID to bytes
            merges: List of BPE merge operations in order
            special_tokens: List of special tokens
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Build reverse vocab for decoding
        self.token_to_id = {token: id for id, token in vocab.items()}

        # Build merge priority (higher index = higher priority)
        self.merge_priority = {pair: i for i, pair in enumerate(merges)}

        # Build special token pattern
        # Sort by length (longest first) for greedy matching
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(token) for token in sorted_tokens]
            self.special_pattern = "|".join(escaped)
        else:
            self.special_pattern = None

    def _apply_bpe(self, token: bytes) -> list[int]:
        """
        Apply BPE merges to a single token to get token IDs.

        Args:
            token: Byte sequence to encode

        Returns:
            List of token IDs
        """
        # Start with individual bytes
        parts = [bytes([b]) for b in token]

        # Apply merges iteratively
        while len(parts) > 1:
            # Find the highest priority pair to merge
            best_pair = None
            best_priority = -1
            best_pos = -1

            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                if pair in self.merge_priority:
                    priority = self.merge_priority[pair]
                    if priority > best_priority:
                        best_pair = pair
                        best_priority = priority
                        best_pos = i

            # No more merges possible
            if best_pair is None:
                break

            # Merge the best pair
            parts = parts[:best_pos] + [best_pair[0] + best_pair[1]] + parts[best_pos + 2:]

        # Convert to token IDs
        return [self.token_to_id[part] for part in parts]

    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        token_ids = []

        # Handle special tokens
        if self.special_pattern:
            segments = re.split(f"({self.special_pattern})", text)
            for segment in segments:
                if not segment:
                    continue

                # Check if this segment is a special token
                if re.fullmatch(self.special_pattern, segment):
                    # Encode special token directly
                    token_bytes = segment.encode("utf-8")
                    token_ids.append(self.token_to_id[token_bytes])
                else:
                    # Pre-tokenize and apply BPE
                    import regex
                    for match in regex.finditer(GPT2_SPLIT_PATTERN, segment):
                        token = match.group().encode("utf-8")
                        token_ids.extend(self._apply_bpe(token))
        else:
            # No special tokens, just pre-tokenize and apply BPE
            import regex
            for match in regex.finditer(GPT2_SPLIT_PATTERN, text):
                token = match.group().encode("utf-8")
                token_ids.extend(self._apply_bpe(token))

        return token_ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text
        """
        # Concatenate all token bytes
        tokens = [self.vocab[id] for id in ids]
        text_bytes = b"".join(tokens)

        # Decode to string, replacing invalid UTF-8 with U+FFFD
        return text_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Memory-efficient encoding of an iterable of strings.

        Args:
            iterable: Iterable of text strings

        Yields:
            Token IDs one at a time
        """
        for text in iterable:
            yield from self.encode(text)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        Load tokenizer from vocabulary and merges files.

        Args:
            vocab_filepath: Path to vocab.json
            merges_filepath: Path to merges.txt
            special_tokens: List of special tokens

        Returns:
            Tokenizer instance
        """
        import json

        # Load vocab
        with open(vocab_filepath, "r") as f:
            vocab_json = json.load(f)
            vocab = {int(k): v.encode("latin-1") if isinstance(v, str) else v
                     for k, v in vocab_json.items()}

        # Load merges
        merges = []
        with open(merges_filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        token1 = parts[0].encode("utf-8")
                        token2 = parts[1].encode("utf-8")
                        merges.append((token1, token2))

        return cls(vocab, merges, special_tokens)
