"""
BPE 分词器最小验证脚本 — 无需下载任何数据集。
用手工构造的小数据来验证 BPE 训练、编码、解码的输入输出。

参考作业 PDF 中 bpe_example 的 stylized example (Sennrich et al. 2016)。
"""

# ============================================================
# Part 0: 理解预分词 (pre-tokenization)
# ============================================================
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

test_strings = [
    "some text that i'll pre-tokenize",
    "Hello, world! 123",
    "low low low lower widest newest",
    "This is a test<|endoftext|>Another doc",
]

print("=== Part 0: Pre-tokenization ===")
for s in test_strings:
    tokens = re.findall(PAT, s)
    print(f"  Input:  {s!r}")
    print(f"  Tokens: {tokens}")
    print()

# ============================================================
# Part 1: 手动模拟 BPE 训练 (来自作业 PDF 的 example)
# ============================================================
print("=== Part 1: BPE Training (手动模拟) ===")

# 模拟语料: "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
# 预分词后的频率表 (假设 split on whitespace)
pretokens = {
    (b"l", b"o", b"w"): 5,
    (b"l", b"o", b"w", b"e", b"r"): 2,
    (b"w", b"i", b"d", b"e", b"s", b"t"): 3,
    (b"n", b"e", b"w", b"e", b"s", b"t"): 6,
}

print("初始频率表:")
for tok, freq in pretokens.items():
    display = " ".join(t.decode() for t in tok)
    print(f"  ({display}): {freq}")

# 统计所有相邻 pair 的频率
from collections import Counter

def count_pairs(pretokens):
    pairs = Counter()
    for token_seq, freq in pretokens.items():
        for i in range(len(token_seq) - 1):
            pairs[(token_seq[i], token_seq[i + 1])] += freq
    return pairs

pairs = count_pairs(pretokens)
print("\n所有 pair 频率:")
for pair, freq in sorted(pairs.items(), key=lambda x: -x[1]):
    print(f"  ({pair[0]!r}, {pair[1]!r}): {freq}")

# 找到频率最高的 pair（tie-breaking: 取字典序最大的）
best_pair = max(pairs, key=lambda p: (pairs[p], p))
print(f"\n最佳合并: ({best_pair[0]!r}, {best_pair[1]!r}) 频率={pairs[best_pair]}")
print("(注意: 's' + 't' 和 'e' + 's' 频率都是 9，取字典序更大的 ('s','t'))")

# 执行合并
def apply_merge(pretokens, pair):
    new_pretokens = {}
    merged = pair[0] + pair[1]  # bytes 拼接
    for token_seq, freq in pretokens.items():
        new_seq = []
        i = 0
        while i < len(token_seq):
            if i < len(token_seq) - 1 and token_seq[i] == pair[0] and token_seq[i + 1] == pair[1]:
                new_seq.append(merged)
                i += 2
            else:
                new_seq.append(token_seq[i])
                i += 1
        new_pretokens[tuple(new_seq)] = freq
    return new_pretokens

# 模拟前 6 次合并
merges = []
print("\n--- 逐步合并 ---")
for step in range(6):
    pairs = count_pairs(pretokens)
    if not pairs:
        break
    best_pair = max(pairs, key=lambda p: (pairs[p], p))
    merges.append(best_pair)
    pretokens = apply_merge(pretokens, best_pair)
    merged_display = best_pair[0] + best_pair[1]
    print(f"  Merge {step + 1}: {best_pair[0]!r} + {best_pair[1]!r} → {merged_display!r}")

print(f"\n合并列表: {[f'{a!r} {b!r}' for a, b in merges]}")
print("(应该匹配: ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e'])")

print("\n合并后的频率表:")
for tok, freq in pretokens.items():
    display = " ".join(t.decode("utf-8") for t in tok)
    print(f"  ({display}): {freq}")

# ============================================================
# Part 2: 手动模拟编码 (Encoding)
# ============================================================
print("\n=== Part 2: BPE Encoding (手动模拟) ===")

# 用作业 PDF 的 encoding example
vocab = {
    0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t",
    6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at",
}
encode_merges = [
    (b"t", b"h"),
    (b" ", b"c"),
    (b" ", b"a"),
    (b"th", b"e"),
    (b" a", b"t"),
]

# 反向 vocab: bytes → id
bytes_to_id = {v: k for k, v in vocab.items()}

def encode_pretoken(token_bytes, merges):
    """对一个 pre-token 应用 merges"""
    seq = [bytes([b]) for b in token_bytes]  # 拆成单字节
    print(f"    初始: {[s.decode('utf-8', errors='replace') for s in seq]}")

    for pair in merges:
        new_seq = []
        i = 0
        applied = False
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                new_seq.append(pair[0] + pair[1])
                i += 2
                applied = True
            else:
                new_seq.append(seq[i])
                i += 1
        seq = new_seq
        if applied:
            print(f"    合并 {pair[0]!r}+{pair[1]!r}: {[s.decode('utf-8', errors='replace') for s in seq]}")

    return seq

text = "the cat ate"
pretokens_text = re.findall(PAT, text)
print(f"输入: {text!r}")
print(f"预分词: {pretokens_text}")

all_ids = []
for pt in pretokens_text:
    print(f"\n  处理 pre-token: {pt!r}")
    pt_bytes = pt.encode("utf-8")
    merged = encode_pretoken(pt_bytes, encode_merges)
    ids = [bytes_to_id[m] for m in merged]
    print(f"    → token IDs: {ids}")
    all_ids.extend(ids)

print(f"\n最终编码: {text!r} → {all_ids}")
print("(应该匹配: [9, 7, 1, 5, 10, 3])")

# ============================================================
# Part 3: 手动模拟解码 (Decoding)
# ============================================================
print("\n=== Part 3: BPE Decoding (手动模拟) ===")

ids_to_decode = [9, 7, 1, 5, 10, 3]
decoded_bytes = b"".join(vocab[i] for i in ids_to_decode)
decoded_str = decoded_bytes.decode("utf-8", errors="replace")
print(f"输入 IDs: {ids_to_decode}")
print(f"拼接 bytes: {decoded_bytes}")
print(f"解码结果: {decoded_str!r}")
print("(应该匹配: 'the cat ate')")

# ============================================================
# Part 4: UTF-8 编码基础验证
# ============================================================
print("\n=== Part 4: UTF-8 编码基础 ===")

examples = ["hello", "牛", "hello! こんにちは!", "🎉"]
for s in examples:
    encoded = s.encode("utf-8")
    print(f"  {s!r:20s} → {len(s)} chars, {len(encoded)} bytes, bytes={list(encoded)}")

print("\n=== 完成! 所有 BPE 核心逻辑已验证 ===")
