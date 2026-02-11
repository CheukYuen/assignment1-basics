"""
Byte-Pair Encoding (BPE) Tokenizer Implementation
==================================================

本模块实现了从零开始的 BPE 分词器，包含以下核心组件：
1. 使用 GPT-2 正则表达式模式的预分词 (Pre-tokenization)
2. 基于频率合并的 BPE 训练算法
3. 用于编码和解码文本的 Tokenizer 类

【BPE 算法背景知识】
===================
Byte-Pair Encoding (BPE) 最初是一种数据压缩算法，由 Philip Gage 在 1994 年提出。
2015 年，Sennrich 等人将其应用于神经机器翻译中的子词 (subword) 分割。
现代大语言模型（如 GPT 系列）广泛采用 BPE 或其变体作为分词方法。

BPE 的核心思想：
- 从字符级（或字节级）表示开始
- 迭代地将最频繁出现的相邻 token 对合并成新的 token
- 重复直到达到目标词汇表大小

BPE 的优势：
1. 开放词汇表：能够处理任何输入文本，不会出现 OOV (Out-of-Vocabulary) 问题
2. 数据驱动：通过训练语料自动学习最佳的子词切分方式
3. 平衡性：在字符级和词级之间取得平衡，既保留语义信息又控制词汇表大小
4. 可逆性：编码和解码过程完全可逆

【字节级 BPE vs 字符级 BPE】
===========================
本实现采用字节级 BPE (Byte-level BPE)，与 GPT-2/GPT-3 相同：
- 基础词汇表：256 个单字节 token (0x00-0xFF)
- 优势：天然支持任何 Unicode 字符，无需特殊的 UNK token
- 任何 UTF-8 编码的文本都可以被表示为字节序列

对比字符级 BPE：
- 基础词汇表：所有出现的 Unicode 字符
- 问题：词汇表可能很大，且可能遇到训练时未见过的字符
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from typing import BinaryIO, Iterable, Iterator

# =============================================================================
# GPT-2 预分词正则表达式模式
# =============================================================================
# 这个正则表达式来自 GPT-2 论文和开源实现，用于在 BPE 之前进行预分词。
#
# 【预分词的作用】
# 预分词将文本分割成较小的单元，确保 BPE 合并不会跨越这些边界。
# 这样做的好处：
# 1. 防止无意义的跨词合并（如 "end of" 不会合并成 "ndof"）
# 2. 保留词的边界信息（空格通常附加到下一个词的开头）
# 3. 提高分词的语言学合理性
#
# 【正则表达式详解】
# 模式按优先级从左到右匹配：
#
# '(?:[sdmt]|ll|ve|re)  - 英语缩写的后缀部分
#                         匹配 's, 'd, 'm, 't, 'll, 've, 're
#                         例如：don't → don + 't, I'll → I + 'll
#
# | ?\p{L}+             - 可选空格 + Unicode 字母序列
#                         \p{L} 是 Unicode 属性，匹配任何语言的字母
#                         这是匹配普通单词的主要模式
#
# | ?\p{N}+             - 可选空格 + Unicode 数字序列
#                         \p{N} 匹配任何 Unicode 数字（包括中文数字等）
#
# | ?[^\s\p{L}\p{N}]+   - 可选空格 + 非空白非字母非数字的序列
#                         匹配标点符号和特殊字符
#
# |\s+(?!\S)            - 行尾空白（不跟非空白字符的空白）
#                         使用负向前瞻 (?!\S) 确保是行尾空白
#
# |\s+                  - 其他空白序列
#                         作为最后的兜底匹配
#
# 【空格处理的特殊设计】
# 注意模式中的 " ?" 表示可选的前导空格。这意味着空格会附加到下一个词的开头，
# 而不是作为独立的 token。例如：
# "Hello World" → ["Hello", " World"] 而不是 ["Hello", " ", "World"]
#
# 这种设计的优势：
# 1. 减少 token 数量（空格不单独占用）
# 2. 保留了词首空格信息，解码时能正确恢复
# =============================================================================
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _get_pair_counts(words: dict[tuple[int, ...], int]) -> Counter:
    """
    统计词典中所有相邻 token 对的出现频率。

    【算法原理】
    这是 BPE 训练的核心步骤之一。我们需要找到整个语料库中出现频率最高的
    相邻 token 对，作为下一次合并的候选。

    【实现细节】
    - 遍历每个词（已经被转换为 token ID 序列）
    - 对于长度为 n 的词，有 n-1 个相邻对
    - 每个对的频率等于该词在语料中出现的次数

    【复杂度分析】
    - 时间复杂度：O(N)，其中 N 是所有词的总长度
    - 空间复杂度：O(V²)，其中 V 是当前词汇表大小（最坏情况）

    【优化说明】
    在大规模实现中，通常会维护一个优先队列来避免每次都重新统计所有对。
    但这个简单实现更易于理解和正确性验证。

    Args:
        words: 词典，key 是 token ID 元组，value 是该词的出现频率

    Returns:
        Counter 对象，记录每个 token 对的总出现频率
    """
    pairs = Counter()
    for word, freq in words.items():
        # 遍历词中的每个相邻位置
        for i in range(len(word) - 1):
            # 将该对的频率加上整个词的出现次数
            # 因为这个词每出现一次，这个对就出现一次
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def _merge_word(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    """
    在一个词中执行特定的 token 对合并操作。

    【算法原理】
    当我们决定将 token 对 (a, b) 合并为新 token c 时，需要更新语料库中
    所有包含这个对的词。这个函数处理单个词的更新。

    【实现细节】
    使用双指针扫描：
    - 当找到目标对 (pair[0], pair[1]) 时，输出 new_id 并跳过两个位置
    - 否则输出当前 token 并前进一个位置

    【示例】
    假设 word = (72, 101, 108, 108, 111)  # "Hello" 的字节表示
    如果 pair = (108, 108)  # "ll"
    new_id = 256  # 新分配的 token ID
    结果 = (72, 101, 256, 111)  # 两个 108 被合并为 256

    【边界情况】
    - 连续出现的对：(a, b, b) 中的 (b, b) 会被正确处理
    - 重叠的模式：(a, a, a) 只会合并前两个 a（贪婪匹配）

    Args:
        word: 原始词的 token ID 元组
        pair: 要合并的 token 对 (token_a, token_b)
        new_id: 合并后新 token 的 ID

    Returns:
        合并后的新词（token ID 元组）
    """
    new_word = []
    i = 0
    while i < len(word):
        # 检查当前位置和下一个位置是否匹配目标对
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            # 找到匹配，输出新 token 并跳过两个位置
            new_word.append(new_id)
            i += 2
        else:
            # 不匹配，保持原样
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def _pretokenize_chunk(args: tuple[str, str | None]) -> Counter:
    """
    对文本块进行预分词，并统计字节级 token 的频率。

    【预分词 (Pre-tokenization) 的重要性】
    预分词是 BPE 流程中的关键步骤，它决定了 BPE 合并的边界。

    主要目的：
    1. 将连续文本分割成有意义的单元（通常接近词的粒度）
    2. 确保 BPE 合并不会跨越这些边界
    3. 处理特殊 token（如 <|endoftext|>）使其保持完整

    【处理流程】
    1. 如果存在特殊 token，首先用正则表达式分割文本
    2. 对非特殊 token 的部分应用 GPT-2 预分词模式
    3. 将每个预分词结果编码为 UTF-8 字节序列
    4. 统计每个字节序列的出现频率

    【为什么要统计频率而不是直接存储？】
    - 内存效率：语料库中相同的词会重复出现，统计频率避免重复存储
    - 计算效率：BPE 训练需要频率信息，预先统计避免重复计算

    Args:
        args: 元组 (text_chunk, special_tokens_pattern)
              text_chunk: 要处理的文本
              special_tokens_pattern: 特殊 token 的正则模式（可选）

    Returns:
        Counter 对象，key 是 bytes 对象，value 是出现频率
    """
    text, special_pattern = args
    counts = Counter()

    # 如果有特殊 token，需要先按特殊 token 分割
    # 使用捕获组 () 确保分割后保留特殊 token 本身
    if special_pattern:
        segments = re.split(f"({special_pattern})", text)
        for segment in segments:
            if segment and not re.fullmatch(special_pattern, segment):
                # 对非特殊 token 的段落应用 GPT-2 预分词
                import regex  # 使用 regex 库支持 Unicode 属性 \p{L}
                for match in regex.finditer(GPT2_SPLIT_PATTERN, segment):
                    # 将匹配的文本编码为 UTF-8 字节
                    token = match.group().encode("utf-8")
                    counts[token] += 1
    else:
        # 没有特殊 token，直接预分词
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
    在文本语料库上训练 BPE 分词器。

    【BPE 训练算法详解】
    =====================

    BPE 训练是一个贪婪的迭代过程：

    1. 初始化：
       - 基础词汇表包含 256 个单字节 token (0x00-0xFF)
       - 可选地添加特殊 token（如 <|endoftext|>）

    2. 预处理：
       - 读取语料库并进行预分词
       - 将每个预分词结果转换为字节序列
       - 统计每个序列的频率

    3. 迭代合并：
       重复以下步骤直到达到目标词汇表大小：
       a) 统计所有相邻 token 对的频率
       b) 选择频率最高的对（平局时按字节序比较）
       c) 创建新 token = token_a + token_b
       d) 更新所有词，用新 token 替换所有出现的 (token_a, token_b)

    【为什么选择最频繁的对？】
    - 信息论角度：高频对意味着强共现关系，合并后压缩效果最好
    - 语言学角度：高频组合通常是有意义的子词单元（如 "th", "ing"）
    - 效率角度：优先合并高频对能最快减少序列长度

    【平局处理 (Tie-breaking)】
    当多个对有相同的最高频率时，我们按字节序（lexicographic order）选择。
    这确保了训练结果的确定性和可复现性。

    【词汇表大小的选择】
    - 太小：分词粒度过细，序列太长，训练/推理慢
    - 太大：词汇表稀疏，低频 token 学不好，内存占用大
    - 典型值：GPT-2 使用 50257，LLaMA 使用 32000

    【时间复杂度分析】
    假设语料大小为 N，目标词汇表大小为 V：
    - 预分词：O(N)
    - 每次合并：O(N)（需要遍历所有词）
    - 总合并次数：V - 256 - len(special_tokens)
    - 总复杂度：O(N * V)

    实际优化中可以使用更复杂的数据结构降低复杂度，但这里为了清晰性
    采用了直接的实现方式。

    Args:
        input_path: 训练语料文件路径（文本文件）
        vocab_size: 目标词汇表大小（包括特殊 token）
        special_tokens: 要添加到词汇表的特殊 token 列表

    Returns:
        vocab: 字典，映射 token ID -> bytes
        merges: 合并操作列表，按执行顺序排列，每项为 (bytes, bytes) 元组
    """
    special_tokens = special_tokens or []

    # ==========================================================================
    # 步骤 1：读取语料库并预分词
    # ==========================================================================
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # 构建特殊 token 的匹配模式
    # 按长度降序排序，确保长的特殊 token 优先匹配（贪婪匹配）
    # 例如：如果有 "<|end|>" 和 "<|endoftext|>"，应该先尝试匹配后者
    special_pattern = None
    if special_tokens:
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        # re.escape 转义特殊字符，防止 token 中的字符被误解为正则语法
        escaped_tokens = [re.escape(token) for token in sorted_tokens]
        special_pattern = "|".join(escaped_tokens)

    # 执行预分词并统计频率
    token_counts = _pretokenize_chunk((text, special_pattern))

    # ==========================================================================
    # 步骤 2：初始化词汇表
    # ==========================================================================
    # 基础词汇表：256 个单字节 token
    # 这确保任何 UTF-8 编码的文本都可以被表示
    vocab_bytes = {i: bytes([i]) for i in range(256)}
    next_token_id = 256

    # 添加特殊 token 到词汇表
    # 特殊 token 被整体作为一个 token，不会被 BPE 分割
    for special_token in special_tokens:
        vocab_bytes[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1

    # 计算需要执行的合并次数
    # 目标词汇表大小 - 当前词汇表大小 = 需要创建的新 token 数量
    num_merges = vocab_size - next_token_id

    # ==========================================================================
    # 步骤 3：将预分词结果转换为 token ID 序列
    # ==========================================================================
    # 每个字节值 (0-255) 就是对应的 token ID
    # 例如：b"Hello" = (72, 101, 108, 108, 111)
    words = {}
    for token_bytes, count in token_counts.items():
        # tuple(bytes) 会将 bytes 拆分为单个字节值的元组
        word = tuple(token_bytes)  # 每个元素是一个字节值 (0-255)
        words[word] = count

    # ==========================================================================
    # 步骤 4：执行 BPE 合并
    # ==========================================================================
    merges = []

    for _ in range(num_merges):
        # 4a. 统计所有相邻 token 对的频率
        pair_counts = _get_pair_counts(words)

        if not pair_counts:
            # 没有更多的对可以合并（所有词都已经是单个 token）
            break

        # 4b. 找到频率最高的对
        max_count = max(pair_counts.values())

        # 处理平局：按字节序比较
        # 这确保了确定性的结果，相同的输入总是产生相同的输出
        candidates = [(pair, count) for pair, count in pair_counts.items() if count == max_count]
        # 比较时使用 bytes 而不是 int，因为多字节 token 的比较需要考虑完整内容
        best_pair = max(candidates, key=lambda x: (vocab_bytes[x[0][0]], vocab_bytes[x[0][1]]))[0]

        # 4c. 创建新 token
        new_token_id = next_token_id
        next_token_id += 1

        # 新 token 的内容是两个被合并 token 的拼接
        vocab_bytes[new_token_id] = vocab_bytes[best_pair[0]] + vocab_bytes[best_pair[1]]
        # 记录这次合并操作（使用 bytes 表示，便于后续使用）
        merges.append((vocab_bytes[best_pair[0]], vocab_bytes[best_pair[1]]))

        # 4d. 更新所有词，执行合并
        # 注意：这里创建了新的字典而不是原地修改，
        # 因为合并可能改变词的哈希值（元组内容改变）
        words = {_merge_word(word, best_pair, new_token_id): freq for word, freq in words.items()}

    return vocab_bytes, merges


class Tokenizer:
    """
    BPE 分词器，用于文本的编码和解码。

    【Tokenizer 的作用】
    Tokenizer 是连接原始文本和模型输入的桥梁。它负责：
    1. 编码 (Encode)：将人类可读的文本转换为模型能处理的 token ID 序列
    2. 解码 (Decode)：将模型输出的 token ID 序列转换回人类可读的文本

    【编码流程】
    文本 → 预分词 → 字节序列 → 应用 BPE 合并 → token ID 序列

    【解码流程】
    token ID 序列 → 查表得到 bytes → 拼接 → UTF-8 解码 → 文本

    【重要属性】
    - vocab: token ID 到 bytes 的映射
    - merges: BPE 合并操作列表（按训练时的顺序）
    - token_to_id: bytes 到 token ID 的反向映射（用于编码）
    - merge_priority: 合并操作的优先级（后训练的优先级更高）
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        从词汇表和合并规则初始化分词器。

        【为什么需要 merge_priority？】
        在编码时，我们需要按照训练时相同的顺序应用合并操作。
        但与训练不同，编码时我们不是从最频繁的对开始，
        而是使用 merge_priority 来确定应该先应用哪个合并。

        【特殊 token 处理】
        特殊 token 在编码时会被识别并整体处理，不会被拆分。
        它们在文本中的匹配使用正则表达式，按长度降序匹配（贪婪）。

        Args:
            vocab: token ID 到 bytes 的映射字典
            merges: BPE 合并操作列表，按训练顺序
            special_tokens: 特殊 token 字符串列表
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # 构建反向词汇表：bytes -> token ID
        # 用于编码时快速查找
        self.token_to_id = {token: id for id, token in vocab.items()}

        # 构建合并优先级映射：(bytes, bytes) -> priority
        # 索引越大优先级越高（后训练的合并优先应用）
        # 这是因为后面的合并是基于前面合并结果的，所以要先应用
        self.merge_priority = {pair: i for i, pair in enumerate(merges)}

        # 构建特殊 token 匹配模式
        # 按长度降序排序确保贪婪匹配（先匹配长的）
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(token) for token in sorted_tokens]
            self.special_pattern = "|".join(escaped)
        else:
            self.special_pattern = None

    def _apply_bpe(self, token: bytes) -> list[int]:
        """
        对单个预分词结果应用 BPE 合并，得到 token ID 序列。

        【编码时的 BPE 应用算法】
        与训练不同，编码时我们不是按频率选择合并，而是按优先级。

        算法流程：
        1. 将输入字节序列拆分为单字节 parts
        2. 重复以下步骤直到无法继续：
           a) 扫描所有相邻对，找到优先级最高的可合并对
           b) 如果找到，执行合并
           c) 如果没找到，结束
        3. 将最终的 parts 转换为 token ID

        【为什么用最高优先级而不是最早出现的位置？】
        考虑序列 "abc"，假设我们有合并规则：
        1. a+b → ab (优先级 0)
        2. b+c → bc (优先级 1)

        如果按位置优先，会得到 [ab, c]
        如果按优先级，会得到 [a, bc]

        使用优先级是因为后训练的合并代表更强的共现关系，
        应该优先应用以获得最佳的压缩效果。

        【时间复杂度】
        最坏情况 O(n²)，其中 n 是输入字节数。
        实际中因为合并会减少序列长度，通常远低于最坏情况。

        Args:
            token: 要编码的字节序列（一个预分词结果）

        Returns:
            token ID 列表
        """
        # 从单字节开始
        parts = [bytes([b]) for b in token]

        # 迭代应用合并
        while len(parts) > 1:
            # 找优先级最高的可合并对
            best_pair = None
            best_priority = -1
            best_pos = -1

            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                if pair in self.merge_priority:
                    priority = self.merge_priority[pair]
                    # 注意：这里用 > 而不是 >=，确保选择第一个出现的最高优先级对
                    if priority > best_priority:
                        best_pair = pair
                        best_priority = priority
                        best_pos = i

            # 没有更多可合并的对
            if best_pair is None:
                break

            # 执行合并：用合并后的 bytes 替换原来的两个部分
            parts = parts[:best_pos] + [best_pair[0] + best_pair[1]] + parts[best_pos + 2:]

        # 将 bytes 转换为 token ID
        return [self.token_to_id[part] for part in parts]

    def encode(self, text: str) -> list[int]:
        """
        将文本编码为 token ID 序列。

        【完整编码流程】
        1. 特殊 token 分割：用正则表达式识别特殊 token
        2. 对非特殊部分：
           a) 预分词：用 GPT-2 模式分割
           b) UTF-8 编码：转换为字节序列
           c) BPE 应用：调用 _apply_bpe
        3. 对特殊 token：直接查表得到 ID
        4. 按顺序拼接所有 token ID

        【特殊 token 的处理】
        特殊 token（如 <|endoftext|>）需要被整体识别，不能被拆分。
        我们用正则表达式在预分词之前先识别它们，然后单独处理。

        Args:
            text: 输入文本字符串

        Returns:
            token ID 列表
        """
        token_ids = []

        # 处理特殊 token
        if self.special_pattern:
            # 使用捕获组分割，保留特殊 token
            segments = re.split(f"({self.special_pattern})", text)
            for segment in segments:
                if not segment:
                    continue

                # 检查是否是特殊 token
                if re.fullmatch(self.special_pattern, segment):
                    # 特殊 token：直接编码为单个 token
                    token_bytes = segment.encode("utf-8")
                    token_ids.append(self.token_to_id[token_bytes])
                else:
                    # 普通文本：预分词 + BPE
                    import regex
                    for match in regex.finditer(GPT2_SPLIT_PATTERN, segment):
                        token = match.group().encode("utf-8")
                        token_ids.extend(self._apply_bpe(token))
        else:
            # 没有特殊 token，直接预分词 + BPE
            import regex
            for match in regex.finditer(GPT2_SPLIT_PATTERN, text):
                token = match.group().encode("utf-8")
                token_ids.extend(self._apply_bpe(token))

        return token_ids

    def decode(self, ids: list[int]) -> str:
        """
        将 token ID 序列解码为文本。

        【解码流程】
        1. 对每个 token ID，从 vocab 中查找对应的 bytes
        2. 将所有 bytes 拼接成一个大的字节序列
        3. 用 UTF-8 解码为字符串

        【错误处理】
        使用 errors="replace" 处理无效的 UTF-8 序列。
        当某些字节组合不能形成有效的 UTF-8 字符时，
        用 U+FFFD (REPLACEMENT CHARACTER, �) 替代。

        这种情况可能发生在：
        - token 序列被截断在 UTF-8 多字节字符的中间
        - 模型生成了不合理的 token 组合

        【可逆性】
        在正常情况下，decode(encode(text)) == text
        但如果原文包含无效 UTF-8 或者 token 序列被截断，可能不完全相等。

        Args:
            ids: token ID 列表

        Returns:
            解码后的文本字符串
        """
        # 查表：token ID -> bytes
        tokens = [self.vocab[id] for id in ids]
        # 拼接所有字节
        text_bytes = b"".join(tokens)
        # UTF-8 解码，无效序列替换为 U+FFFD
        return text_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        内存高效的批量编码方法。

        【使用场景】
        当需要处理大量文本时（如整个训练集），
        不想一次性将所有 token ID 加载到内存中。

        【生成器的优势】
        使用 yield 返回生成器，具有以下优点：
        1. 惰性求值：只在需要时才计算
        2. 内存高效：不存储中间结果
        3. 可组合：可以用 itertools 等工具链式处理

        【示例用法】
        ```python
        # 处理大文件的每一行
        with open("large_file.txt") as f:
            for token_id in tokenizer.encode_iterable(f):
                process(token_id)

        # 批量写入
        token_ids = list(islice(tokenizer.encode_iterable(texts), 10000))
        ```

        Args:
            iterable: 可迭代的文本字符串

        Yields:
            token ID（一次一个）
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
        从文件加载分词器。

        【文件格式】
        vocab.json: JSON 对象，key 是 token ID (字符串)，value 是 token 内容
            示例：{"0": "\x00", "1": "\x01", ..., "256": "ab", ...}

        merges.txt: 每行一个合并规则，格式为 "token1 token2"
            示例：
            a b
            ab c
            abc d

        【编码处理】
        vocab 文件使用 latin-1 编码解析 token 内容，
        因为 JSON 中的字符串可能包含转义的字节值。
        latin-1 编码的特点是：字符码点 == 字节值（对于 0-255）

        merges 文件使用 UTF-8 编码，token 按空格分割。

        Args:
            vocab_filepath: vocab.json 文件路径
            merges_filepath: merges.txt 文件路径
            special_tokens: 特殊 token 列表

        Returns:
            初始化好的 Tokenizer 实例
        """
        import json

        # 加载词汇表
        with open(vocab_filepath, "r") as f:
            vocab_json = json.load(f)
            # 将 JSON 中的字符串转换为 bytes
            # 使用 latin-1 因为它保持字节值不变
            vocab = {int(k): v.encode("latin-1") if isinstance(v, str) else v
                     for k, v in vocab_json.items()}

        # 加载合并规则
        merges = []
        with open(merges_filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        # 将 token 字符串编码为 bytes
                        token1 = parts[0].encode("utf-8")
                        token2 = parts[1].encode("utf-8")
                        merges.append((token1, token2))

        return cls(vocab, merges, special_tokens)
