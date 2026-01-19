# CS336 Assignment 1 · 理解驱动版项目规划

> **目标导向**: 不是"写完一个 Transformer"，而是**每一层都知道它为什么存在、拿掉会如何失败**。
> **方法论**: Implementation ✔ + Unit Test ✔ + Conceptual Checkpoint ✔

## 项目元信息
- **课程**: Stanford CS336 - Language Modeling from Scratch
- **作业**: Assignment 1 - 从零构建Transformer语言模型
- **分支**: `assignment-implementation`
- **目标**: 通过**最小可验证实验（MVT）**，建立对 Transformer 的可解释理解

## 全局原则（必须遵守）
1. **训练目标一致性**: 训练阶段与生成阶段等价（依赖 causal mask）
2. **自回归闭环**: 每一步预测下一个 token，并回馈为下一步输入
3. **概率建模优先**: logits ≠ 概率；softmax 与 loss/decoding 解耦
4. **理解可审计**: 每个模块都必须通过一个 Conceptual Checkpoint

## 学习者背景
- 前端开发工程师
- 有Prompt工程经验
- 会基础Python
- 目标是智能体开发

## 作业概览

### 需要实现的组件
1. **BPE Tokenizer** (§2) - 字节对编码分词器
2. **Transformer LM** (§3) - Transformer语言模型
3. **Loss & Optimizer** (§4) - 交叉熵损失和AdamW优化器
4. **Training Loop** (§5) - 训练循环和checkpoint
5. **Text Generation** (§6) - 文本生成
6. **Experiments** (§7) - 实验和调优

### 代码结构
```
cs336_basics/          # 你的实现代码(从零开始)
tests/adapters.py      # 连接测试的胶水代码
tests/test_*.py        # 单元测试(不要修改)
data/                  # 数据集
conceptual_tests/      # 🧠 最小可验证实验（MVT - 你自己写）
```

---

## 📋 详细实施计划

### 阶段1: Tokenizer (2-3天) 🔴 HIGH PRIORITY

> **核心理解**: Tokenizer 决定了**模型能看到的世界边界**。模型永远无法理解 tokenizer 切不出来的结构。

#### 🎯 Conceptual Checkpoints
- [ ] 我能解释 BPE 是如何用频率合并子串的
- [ ] 我能解释 tokenizer 改变会如何影响模型表现（不是"变好/变坏"，而是"看到不同结构"）
- [ ] 我知道 token id 本身没有语义顺序
- [ ] 我理解 compression ratio 的实际意义

#### 📝 Standard Answers (for self-check)
<details>
<summary>Q1: 为什么 BPE 要基于频率合并？</summary>
A: 频率高的子串合并成单个 token 可以：(1) 减少序列长度，降低计算成本；(2) 让常见词/短语成为原子单元，模型更容易学习其语义；(3) 平衡 vocab 大小与序列长度的 trade-off。
</details>

<details>
<summary>Q2: Vocab size 从 10k 变到 32k，会发生什么？</summary>
A: (1) 每个 token 包含更多信息（更长子串）；(2) 序列变短，但 vocab 变大；(3) 罕见词可能被完整保留，而不是切成字符；(4) 对于形态学丰富的语言（如德语），可能更好地保留词根-词缀结构。
</details>

<details>
<summary>Q3: Token 123 和 token 456，哪个"更大"？</summary>
A: 没有"更大"。Token ID 只是查表索引，没有大小关系。这不是数字，而是离散符号。
</details>

#### 任务1.1: Unicode基础 (unicode1, unicode2)
- [ ] 回答chr(0)相关问题
- [ ] 理解UTF-8编码
- [ ] 完成writeup问题
- **测试**: 概念题,无自动化测试
- **文件**: 在writeup.pdf中回答

#### 任务1.2: BPE训练 (train_bpe)
- [ ] 实现pre-tokenization (使用GPT-2 regex)
- [ ] 实现BPE merge算法
- [ ] 支持special tokens
- [ ] 优化: 使用multiprocessing并行化
- [ ] 在TinyStories上训练 (vocab_size=10000, ≤30min, ≤30GB RAM)
- [ ] 在OpenWebText上训练 (vocab_size=32000, ≤12h, ≤100GB RAM)
- **测试**: `uv run pytest tests/test_train_bpe.py`
- **Adapter**: `tests/adapters.py::run_train_bpe`
- **输出**:
  - `vocab.json` (dict[int, bytes])
  - `merges.txt` (list of merge operations)

#### 任务1.3: Tokenizer编解码 (tokenizer)
- [ ] 实现`Tokenizer.__init__(vocab, merges, special_tokens)`
- [ ] 实现`Tokenizer.encode(text) -> list[int]`
- [ ] 实现`Tokenizer.encode_iterable()` (memory-efficient)
- [ ] 实现`Tokenizer.decode(ids) -> str`
- [ ] 处理invalid Unicode (U+FFFD replacement)
- **测试**: `uv run pytest tests/test_tokenizer.py`
- **Adapter**: `tests/adapters.py::get_tokenizer`

#### 任务1.4: Tokenizer实验
- [ ] 对比TinyStories和OWT的compression ratio
- [ ] 估算throughput (bytes/second)
- [ ] 使用tokenizer处理数据集,保存为uint16 numpy array

#### 🧪 MVT-Tokenizer-01: Vocab Size Impact
**目标**: 验证 vocab size 如何影响 tokenization

**实验**:
```python
# conceptual_tests/test_tokenizer.py
def test_vocab_size_impact():
    text = "The quick brown fox jumps over the lazy dog" * 100

    # Train tokenizers with different vocab sizes
    tokenizer_10k = train_bpe(text, vocab_size=10000)
    tokenizer_1k = train_bpe(text, vocab_size=1000)

    tokens_10k = tokenizer_10k.encode(text)
    tokens_1k = tokenizer_1k.encode(text)

    # 验证：larger vocab → shorter sequence
    assert len(tokens_10k) < len(tokens_1k)

    # 验证：compression ratio
    compression_10k = len(text) / len(tokens_10k)
    compression_1k = len(text) / len(tokens_1k)
    assert compression_10k > compression_1k

    print(f"10k vocab: {len(tokens_10k)} tokens, compression {compression_10k:.2f}")
    print(f"1k vocab: {len(tokens_1k)} tokens, compression {compression_1k:.2f}")
```

**Expected Understanding**: Vocab size 是信息密度与模型复杂度的权衡。

---

### 阶段2: 模型基础组件 (3-4天) 🔴 HIGH PRIORITY

> **核心理解**: Embedding 是唯一把离散 token 变成连续语义空间的地方。之后所有模块只是在变换这个空间。

#### 🎯 Conceptual Checkpoints
- [ ] 我确认 token id 本身没有任何语义
- [ ] 我知道 embedding 是"可学习查表"
- [ ] 我理解 Linear 层为什么不需要 bias（在 Transformer 中）
- [ ] 我知道 RMSNorm 为什么比 LayerNorm 更高效

#### 📝 Standard Answers
<details>
<summary>Q1: Embedding 层做了什么？</summary>
A: Embedding 层把离散 token ID 映射到连续向量空间。这个映射是可学习的，训练过程中会调整，使得语义相似的 token 在向量空间中靠近。本质上是一个大的查找表（vocab_size × d_model）。
</details>

<details>
<summary>Q2: 如果打乱 Embedding 矩阵的行，会发生什么？</summary>
A: 语义会完全被破坏。每个 token 会被映射到错误的向量，模型需要重新学习整个语义空间。这证明了 embedding 是唯一赋予 token 语义的地方。
</details>

<details>
<summary>Q3: 为什么 Transformer 的 Linear 层不需要 bias？</summary>
A: 因为后面有 LayerNorm/RMSNorm，它们会中心化激活值。Bias 的作用会被 norm 抵消，所以是冗余的。省略 bias 可以减少参数量和计算。
</details>

#### 任务2.1: Linear (linear)
- [ ] 实现`Linear(in_features, out_features)`
- [ ] 无bias参数
- [ ] 权重初始化: N(0, 2/(d_in+d_out)) truncated at [-3σ, 3σ]
- **测试**: `uv run pytest -k test_linear`
- **Adapter**: `tests/adapters.py::run_linear`

#### 任务2.2: Embedding (embedding)
- [ ] 实现`Embedding(num_embeddings, embedding_dim)`
- [ ] 权重初始化: N(0, 1) truncated at [-3, 3]
- **测试**: `uv run pytest -k test_embedding`
- **Adapter**: `tests/adapters.py::run_embedding`

#### 任务2.3: RMSNorm (rmsnorm)
- [ ] 实现`RMSNorm(d_model, eps=1e-5)`
- [ ] 公式: x / RMS(x) * g, where RMS = sqrt(mean(x²) + ε)
- [ ] Upcast到float32防止overflow
- [ ] gain参数初始化为1
- **测试**: `uv run pytest -k test_rmsnorm`
- **Adapter**: `tests/adapters.py::run_rmsnorm`

#### 任务2.4: Softmax (softmax)
- [ ] 实现数值稳定的softmax
- [ ] 减去max value避免overflow
- **测试**: `uv run pytest -k test_softmax_matches_pytorch`
- **Adapter**: `tests/adapters.py::run_softmax`

#### 任务2.5: SwiGLU (positionwise_feedforward)
- [ ] 实现SiLU(x) = x * σ(x)
- [ ] 实现SwiGLU: W2(SiLU(W1*x) ⊙ W3*x)
- [ ] d_ff = ceil(8/3 * d_model / 64) * 64
- [ ] 可以使用torch.sigmoid
- **测试**: `uv run pytest -k test_swiglu`
- **Adapter**: `tests/adapters.py::run_swiglu`

#### 🧪 MVT-Embedding-01: Semantic Destruction Test
**目标**: 验证 embedding 是唯一的语义来源

**实验**:
```python
# conceptual_tests/test_embedding.py
def test_embedding_semantic_source():
    vocab_size, d_model = 1000, 128
    embedding = Embedding(vocab_size, d_model)

    # Original embeddings
    tokens = torch.tensor([10, 20, 30, 40, 50])
    original_vecs = embedding(tokens)

    # Shuffle embedding matrix rows (破坏语义)
    import random
    perm = list(range(vocab_size))
    random.shuffle(perm)
    embedding.weight.data = embedding.weight.data[perm]

    shuffled_vecs = embedding(tokens)

    # 验证：向量完全不同
    similarity = torch.cosine_similarity(
        original_vecs.flatten(),
        shuffled_vecs.flatten(),
        dim=0
    )
    assert abs(similarity) < 0.1  # 几乎正交

    print(f"Semantic destroyed: cosine similarity = {similarity:.4f}")
```

**Expected Understanding**: Embedding 是语义的唯一来源，打乱它会破坏所有语义。

---

### 阶段3: 注意力机制 (2-3天) 🔴 HIGH PRIORITY

> **核心理解**:
> - **Q×K 决定"向谁要、要多少"，V 决定"拿什么内容"**
> - **Mask 不是优化技巧，而是训练目标成立的前提**
> - **多头不是"多算几次"，而是在不同子空间并行建模关系**

#### 🎯 Conceptual Checkpoints

**Attention 机制**:
- [ ] 我能解释 attention 输出一定是 Value 的线性组合
- [ ] 我知道 attention 是软选择（不是 argmax）
- [ ] 我理解 scaling factor √d_k 的作用

**Causal Mask**:
- [ ] 我能解释 train–inference mismatch 如何被 mask 解决
- [ ] 我能画出 causal mask 矩阵
- [ ] 我知道没有 mask 会导致信息泄漏

**Multi-Head**:
- [ ] 我能解释为什么一个 head 不够
- [ ] 我理解 concat + WO 是信息再融合
- [ ] 我知道不同 head 会学到不同的关注模式

#### 📝 Standard Answers
<details>
<summary>Q1: Attention 的输出是什么？</summary>
A: Attention 输出是 V 的加权平均，权重由 Q·K 决定。数学上：output = Σ(softmax(Q·K/√d_k) · V)。这意味着输出永远在 V 的线性空间内，不会产生"新"信息，只是重新组合。
</details>

<details>
<summary>Q2: 为什么训练时必须用 causal mask？</summary>
A: 因为生成时，第 t 步只能看到前 t-1 个 token。如果训练时看到了未来信息，会产生 train-inference mismatch，导致生成时性能崩溃。Mask 保证训练和推理的信息流一致。
</details>

<details>
<summary>Q3: Multi-head 的本质是什么？</summary>
A: 把 d_model 维空间切成 h 个子空间，每个 head 在自己的子空间独立计算 attention，学习不同的依赖模式（如：一个 head 关注语法，另一个关注语义）。最后 concat + 线性变换融合所有 head 的信息。
</details>

<details>
<summary>Q4: 为什么需要 √d_k scaling？</summary>
A: Q·K 的点积方差会随 d_k 线性增长。不 scale 的话，d_k 很大时 softmax 会饱和（接近 one-hot），梯度消失。除以 √d_k 使方差稳定在 O(1)。
</details>

#### 任务3.1: RoPE (rope)
- [ ] 实现`RotaryPositionalEmbedding(theta, d_k, max_seq_len)`
- [ ] 预计算cos和sin buffers
- [ ] 支持任意batch维度
- [ ] 使用token_positions进行索引
- [ ] 无可学习参数
- **测试**: `uv run pytest -k test_rope`
- **Adapter**: `tests/adapters.py::run_rope`
- **公式**: Ri rotates pairs by angle θi,k = i/Θ^((2k-2)/d)

#### 任务3.2: Scaled Dot-Product Attention (scaled_dot_product_attention)
- [ ] 实现Attention(Q,K,V) = softmax(Q^T K / sqrt(d_k)) V
- [ ] 支持可选mask (boolean tensor)
- [ ] mask=False位置设为-inf (before softmax)
- [ ] 支持任意batch维度
- **测试**: `uv run pytest -k test_scaled_dot_product_attention`
- **Adapter**: `tests/adapters.py::run_scaled_dot_product_attention`

#### 任务3.3: Multi-Head Self-Attention (multihead_self_attention)
- [ ] 实现`MultiHeadSelfAttention(d_model, num_heads)`
- [ ] WQ, WK, WV 投影 (d_model -> h*d_k)
- [ ] 应用RoPE到Q和K (不是V!)
- [ ] Causal masking (下三角mask)
- [ ] WO输出投影
- [ ] d_k = d_v = d_model / num_heads
- **测试**: `uv run pytest -k test_multihead_self_attention`
- **Adapter**: `tests/adapters.py::run_multihead_self_attention`

#### 🧪 MVT-Attention-01: Value Averaging Test
**目标**: 验证 attention 输出是 V 的线性组合

**实验**:
```python
# conceptual_tests/test_attention.py
def test_attention_value_averaging():
    d_k, seq_len = 64, 10

    # 构造均匀的 Q 和 K（所有位置注意力权重相等）
    Q = torch.ones(1, seq_len, d_k)
    K = torch.ones(1, seq_len, d_k)
    V = torch.randn(1, seq_len, d_k)

    # Attention without mask
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    attn_weights = torch.softmax(scores, dim=-1)
    output = attn_weights @ V

    # 验证：输出应该约等于 V 的均值
    expected = V.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
    assert torch.allclose(output, expected, atol=1e-5)

    print("✓ Attention output = weighted average of V")
```

#### 🧪 MVT-Mask-01: Causality Verification
**目标**: 验证 causal mask 阻止未来信息泄漏

**实验**:
```python
def test_causal_mask_prevents_leakage():
    d_k, seq_len = 64, 5
    Q = torch.randn(1, seq_len, d_k)
    K = torch.randn(1, seq_len, d_k)
    V = torch.randn(1, seq_len, d_k)

    # Create causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores.masked_fill(~mask, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)

    # 验证：上三角的权重应该为 0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert attn_weights[0, i, j] < 1e-6

    print("✓ Causal mask prevents future attention")

    # 可视化 attention weights
    import matplotlib.pyplot as plt
    plt.imshow(attn_weights[0].detach().numpy())
    plt.title("Causal Attention Weights")
    plt.colorbar()
    plt.savefig("causal_mask_weights.png")
```

#### 🧪 MVT-MultiHead-01: Head Diversity Test
**目标**: 验证不同 head 学到不同模式

**实验**:
```python
def test_multihead_diversity():
    d_model, num_heads, seq_len = 512, 8, 20
    mha = MultiHeadSelfAttention(d_model, num_heads)

    x = torch.randn(1, seq_len, d_model)

    # 手动计算每个 head 的 attention weights
    # (需要修改 MHA 实现以返回中间 attention weights)
    # 这里假设你实现了 return_attention_weights=True 参数

    _, attn_per_head = mha(x, return_attention_weights=True)
    # attn_per_head: (batch, num_heads, seq_len, seq_len)

    # 计算 head 之间的相似度
    similarities = []
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            sim = torch.cosine_similarity(
                attn_per_head[0, i].flatten(),
                attn_per_head[0, j].flatten(),
                dim=0
            )
            similarities.append(sim.item())

    avg_similarity = sum(similarities) / len(similarities)
    print(f"Average head similarity: {avg_similarity:.4f}")

    # 期望：不同 head 有一定差异（不是完全相同）
    assert avg_similarity < 0.95
```

**Expected Understanding**: Multi-head 让模型在不同表示子空间学习不同的依赖关系。

---

### 阶段4: 完整模型 (1-2天) 🔴 HIGH PRIORITY

> **核心理解**:
> - **Attention = 信息交换，FFN = 信息变换**
> - **Residual = 信息高速公路，Pre-norm = 梯度稳定性**
> - **LM Head 把隐藏表示映射回 token 空间，logits 本身不是概率**

#### 🎯 Conceptual Checkpoints
- [ ] 我能解释 pre-norm 为什么比 post-norm 稳定
- [ ] 我知道移除 residual 会发生什么
- [ ] 我理解 softmax 不应该写在模型里
- [ ] 我理解 logits 的几何意义

#### 📝 Standard Answers
<details>
<summary>Q1: Pre-norm vs Post-norm 的区别？</summary>
A: Pre-norm 在 attention/FFN 之前做归一化，梯度流更稳定，可以训练更深的模型。Post-norm 在之后归一化，容易梯度爆炸/消失。公式：
- Pre-norm: x + F(Norm(x))
- Post-norm: Norm(x + F(x))
现代 Transformer（GPT, Llama）都用 Pre-norm。
</details>

<details>
<summary>Q2: Residual connection 的作用？</summary>
A: (1) 梯度高速公路：梯度可以直接跳过多层传播；(2) 保留原始信息：防止信息在多层变换中丢失；(3) 让模型学习"增量"而非"全量"变换。移除 residual 会导致训练困难，深层网络退化。
</details>

<details>
<summary>Q3: 为什么 softmax 不应该在模型 forward 里？</summary>
A: 因为：(1) 训练时用 cross-entropy，可以和 softmax 融合计算，数值更稳定；(2) 生成时需要 temperature/top-p，要在 softmax 前修改 logits；(3) 模型应该输出"未归一化的分数"，由外部决定如何转为概率。
</details>

#### 任务4.1: Transformer Block (transformer_block)
- [ ] Pre-norm架构:
  - z = x + MHA(RMSNorm(x))
  - y = z + FFN(RMSNorm(z))
- [ ] 包含2个RMSNorm, 1个MHA, 1个FFN
- **测试**: `uv run pytest -k test_transformer_block`
- **Adapter**: `tests/adapters.py::run_transformer_block`

#### 任务4.2: Transformer LM (transformer_lm)
- [ ] 实现完整模型:
  - Token Embedding
  - N个Transformer Blocks
  - Final RMSNorm
  - Output Linear (LM head)
  - Softmax (在loss中,不在模型里)
- [ ] 参数:
  - vocab_size, context_length, num_layers
  - d_model, num_heads, d_ff
- **测试**: `uv run pytest -k test_transformer_lm`
- **Adapter**: `tests/adapters.py::run_transformer_lm`

#### 任务4.3: 资源核算 (transformer_accounting)
- [ ] 计算GPT-2 XL的参数量
- [ ] 识别forward pass的所有矩阵乘法
- [ ] 计算总FLOPs
- [ ] 分析不同模型大小的FLOPs分布
- [ ] MFU计算
- **写在writeup.pdf**

#### 🧪 MVT-Block-01: Residual Bypass Test
**目标**: 验证 residual connection 的作用

**实验**:
```python
# conceptual_tests/test_transformer_block.py
def test_residual_importance():
    d_model, num_heads = 512, 8
    block = TransformerBlock(d_model, num_heads, d_ff=2048)

    x = torch.randn(1, 10, d_model)

    # 正常输出
    y_normal = block(x)

    # 强制 MHA 输出为 0（模拟极端情况）
    with torch.no_grad():
        # 这需要你在实现时支持 debug 模式
        # 或者直接修改 block 的中间状态
        pass

    # 验证：即使 MHA 失效，输出仍然包含输入信息
    # y ≈ x + FFN(RMSNorm(x))

    print("✓ Residual connection preserves input information")
```

#### 🧪 MVT-LMHead-01: Logits Shape Test
**目标**: 验证 LM Head 输出形状

**实验**:
```python
def test_lm_head_output():
    vocab_size, d_model = 10000, 512
    batch, seq_len = 4, 20

    hidden = torch.randn(batch, seq_len, d_model)
    lm_head = Linear(d_model, vocab_size)

    logits = lm_head(hidden)

    # 验证形状
    assert logits.shape == (batch, seq_len, vocab_size)

    # 验证：logits 不是概率（可以是负数，和不为1）
    assert (logits < 0).any()
    assert not torch.allclose(logits.sum(dim=-1), torch.ones(batch, seq_len))

    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Logits are NOT probabilities (min={logits.min():.2f})")
```

---

### 阶段5: 训练基础设施 (2-3天) 🔴 HIGH PRIORITY

> **核心理解**: **Loss 只惩罚一件事：真实 token 的概率不够高。**

#### 🎯 Conceptual Checkpoints
- [ ] 我知道 loss 与采样无关
- [ ] 我理解 teacher forcing
- [ ] 我能解释为什么要用 log-softmax 而不是 softmax + log
- [ ] 我理解梯度裁剪的物理意义

#### 📝 Standard Answers
<details>
<summary>Q1: Cross-Entropy Loss 的本质？</summary>
A: CE Loss = -log P(真实token | context)。只看真实 token 的概率，其他 vocab 的概率通过 softmax 归一化间接影响。Loss 越小，模型给真实 token 的概率越高。
</details>

<details>
<summary>Q2: 什么是 Teacher Forcing？</summary>
A: 训练时，每一步的输入是真实序列（不是模型生成的），这叫 teacher forcing。好处：训练快，梯度稳定；坏处：train-inference mismatch（训练时看真实数据，推理时看自己生成的）。
</details>

<details>
<summary>Q3: 为什么要 gradient clipping？</summary>
A: 防止梯度爆炸。当 ||g|| 太大时，按比例缩小，限制单步更新幅度。这不改变梯度方向，只改变步长，让训练更稳定。
</details>

#### 任务5.1: Cross-Entropy Loss (cross_entropy)
- [ ] 实现 -log(softmax(logits)[target])
- [ ] 数值稳定: 减去max, 消除log和exp
- [ ] 支持batch维度
- **测试**: `uv run pytest -k test_cross_entropy`
- **Adapter**: `tests/adapters.py::run_cross_entropy`

#### 任务5.2: AdamW (adamw)
- [ ] 实现AdamW optimizer (Algorithm 1 from paper)
- [ ] 保持m, v状态 (first/second moments)
- [ ] Bias correction: α_t = α * sqrt(1-β2^t) / (1-β1^t)
- [ ] Weight decay: θ = θ - αλθ
- [ ] 默认: β1=0.9, β2=0.999, ε=1e-8
- **测试**: `uv run pytest -k test_adamw`
- **Adapter**: `tests/adapters.py::get_adamw_cls`

#### 任务5.3: 资源核算 - AdamW (adamwAccounting)
- [ ] 计算peak memory (参数 + 激活 + 梯度 + optimizer state)
- [ ] GPT-2 XL最大batch size (80GB memory)
- [ ] AdamW的FLOPs
- [ ] 训练时间估算 (400K steps, batch=1024, 50% MFU, A100)
- **写在writeup.pdf**

#### 任务5.4: Learning Rate Schedule (learning_rate_schedule)
- [ ] 实现cosine annealing with warmup
- [ ] Warmup: α_t = (t/T_w) * α_max
- [ ] Cosine: α_t = α_min + 0.5(1 + cos(...))*(α_max - α_min)
- [ ] Post-annealing: α_t = α_min
- **测试**: `uv run pytest -k test_get_lr_cosine_schedule`
- **Adapter**: `tests/adapters.py::get_lr_cosine_schedule`

#### 任务5.5: Gradient Clipping (gradient_clipping)
- [ ] 计算全局梯度范数 ||g||_2
- [ ] 如果 ||g|| > M, scale by M/(||g||+ε)
- [ ] ε = 1e-6
- [ ] 原地修改梯度
- **测试**: `uv run pytest -k test_gradient_clipping`
- **Adapter**: `tests/adapters.py::run_gradient_clipping`

#### 任务5.6: Data Loading (data_loading)
- [ ] 实现get_batch(data, batch_size, context_length, device)
- [ ] 随机采样batch_size个起始位置
- [ ] 返回(inputs, targets), 都是(batch_size, context_length)
- [ ] 使用np.memmap加载大文件
- **测试**: `uv run pytest -k test_get_batch`
- **Adapter**: `tests/adapters.py::run_get_batch`

#### 任务5.7: Checkpointing (checkpointing)
- [ ] save_checkpoint(model, optimizer, iteration, out)
- [ ] load_checkpoint(src, model, optimizer) -> iteration
- [ ] 使用model.state_dict() 和 optimizer.state_dict()
- [ ] torch.save() 和 torch.load()
- **测试**: `uv run pytest -k test_checkpointing`
- **Adapter**: `tests/adapters.py::run_save_checkpoint`, `run_load_checkpoint`

#### 任务5.8: Training Loop (training_together)
- [ ] 创建训练脚本
- [ ] 支持配置所有超参数
- [ ] Memory-efficient data loading (np.memmap)
- [ ] 周期性checkpoint
- [ ] 记录train/val loss
- [ ] 可选: Weights & Biases集成

#### 🧪 MVT-Loss-01: Loss Behavior Test
**目标**: 验证 cross-entropy 的行为

**实验**:
```python
# conceptual_tests/test_loss.py
def test_cross_entropy_behavior():
    vocab_size = 100
    batch, seq_len = 2, 5

    # Case 1: 正确 token 概率很高
    logits_confident = torch.zeros(batch, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch, seq_len))
    for b in range(batch):
        for s in range(seq_len):
            logits_confident[b, s, targets[b, s]] = 10.0  # 很高的 logit

    loss_confident = cross_entropy(logits_confident, targets)

    # Case 2: 正确 token 概率很低
    logits_uncertain = torch.randn(batch, seq_len, vocab_size)
    loss_uncertain = cross_entropy(logits_uncertain, targets)

    # 验证：confident case loss 更小
    assert loss_confident < loss_uncertain
    assert loss_confident < 0.5  # 接近 0
    assert loss_uncertain > 3.0  # 接近 log(vocab_size)

    print(f"✓ Confident loss: {loss_confident:.4f}")
    print(f"✓ Uncertain loss: {loss_uncertain:.4f}")

    # Case 3: 改变非target位置的 logits，loss 应该变化不大
    logits_noise = logits_confident.clone()
    for b in range(batch):
        for s in range(seq_len):
            # 增加其他位置的 logit
            mask = torch.ones(vocab_size, dtype=torch.bool)
            mask[targets[b, s]] = False
            logits_noise[b, s, mask] += 5.0

    loss_noise = cross_entropy(logits_noise, targets)

    # Loss 会增加（因为 softmax 归一化），但不会增加很多
    print(f"✓ Loss with noise: {loss_noise:.4f}")
```

---

### 阶段6: 文本生成 (1天) 🟡 MEDIUM PRIORITY

> **核心理解**: **生成 = 用自己的输出当下一步输入（自回归闭环）**

#### 🎯 Conceptual Checkpoints
- [ ] 我能解释为什么丢掉上一步输出会导致失败
- [ ] 我理解 temperature / top-p 改变的是分布形状
- [ ] 我知道 greedy decoding vs sampling 的区别

#### 📝 Standard Answers
<details>
<summary>Q1: 自回归生成的流程？</summary>
A: (1) 给定 prompt，模型预测下一个 token；(2) 把预测的 token 添加到序列；(3) 用新序列再次预测；(4) 重复直到 EOS 或达到最大长度。关键：每一步的输入包含之前所有生成的 token。
</details>

<details>
<summary>Q2: Temperature 的作用？</summary>
A: Temperature τ 控制分布的"尖锐度"：
- τ → 0: 接近 greedy（总是选最大概率）
- τ = 1: 原始分布
- τ > 1: 更平滑（更随机）
通过 logits/τ 实现。高温→更多样化但可能不连贯，低温→更确定但可能重复。
</details>

<details>
<summary>Q3: Top-p (nucleus) sampling 是什么？</summary>
A: 只从累积概率达到 p 的最小 token 集合中采样。例如 p=0.9，选概率最高的 token，直到它们的总概率 ≥ 0.9。这动态调整候选集大小，比 top-k 更灵活。
</details>

#### 任务6.1: Decoding (decoding)
- [ ] 实现基础采样 (从softmax分布采样)
- [ ] Temperature scaling: softmax(v/τ)
- [ ] Top-p (nucleus) sampling
- [ ] 生成直到<|endoftext|>或max_tokens
- [ ] 支持batch生成
- **示例输出**: 至少256 tokens

#### 🧪 MVT-Generation-01: Autoregressive Loop Test
**目标**: 验证自回归闭环

**实验**:
```python
# conceptual_tests/test_generation.py
def test_autoregressive_loop():
    model = TransformerLM(vocab_size=1000, context_length=128, ...)
    tokenizer = get_tokenizer()

    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt)

    # 手动实现一步生成
    for _ in range(10):
        # Forward pass
        logits = model(torch.tensor([input_ids]))  # (1, seq_len, vocab)

        # 只看最后一个位置的 logits
        next_token_logits = logits[0, -1, :]

        # 采样
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        # 🔑 关键：把生成的 token 加入序列
        input_ids.append(next_token)

    generated_text = tokenizer.decode(input_ids)
    print(f"Generated: {generated_text}")

    # 验证：如果不把 next_token 加入序列，后续生成会失败
```

#### 🧪 MVT-Generation-02: Temperature Effect Test
**目标**: 验证 temperature 的效果

**实验**:
```python
def test_temperature_effect():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    # Low temperature (greedy)
    probs_low = torch.softmax(logits / 0.1, dim=-1)
    print(f"T=0.1: {probs_low}")  # 几乎所有概率在最大位置

    # Normal temperature
    probs_normal = torch.softmax(logits / 1.0, dim=-1)
    print(f"T=1.0: {probs_normal}")

    # High temperature (random)
    probs_high = torch.softmax(logits / 2.0, dim=-1)
    print(f"T=2.0: {probs_high}")  # 更均匀

    # 验证：低温更尖锐
    assert probs_low.max() > probs_high.max()
    assert probs_low.std() > probs_high.std()
```

---

### 阶段7: 实验 (5-7天) 🟢 AS NEEDED

#### 任务7.1: 实验日志 (experiment_log)
- [ ] 创建实验跟踪基础设施
- [ ] 记录每个实验的超参数
- [ ] 保存learning curves (steps & wallclock time)
- [ ] 建议: Weights & Biases

#### 任务7.2: TinyStories训练
**基础配置**:
- vocab_size: 10000
- context_length: 256
- d_model: 512
- d_ff: 1344
- num_layers: 4
- num_heads: 16
- total_tokens: 327,680,000
- RoPE theta: 10000
- 目标: val loss ≤ 1.45

**实验**:
- [ ] (learning_rate) 学习率扫描
- [ ] (batch_size_experiment) Batch size实验
- [ ] (generate) 生成文本样本

#### 任务7.3: 消融实验
- [ ] (layer_norm_ablation) 移除RMSNorm
- [ ] (pre_norm_ablation) Post-norm vs Pre-norm
- [ ] (no_pos_emb) NoPE vs RoPE
- [ ] (swiglu_ablation) SwiGLU vs SiLU

#### 任务7.4: OpenWebText
- [ ] (main_experiment) 在OWT上训练
- [ ] 生成样本
- [ ] 分析与TinyStories的差异

#### 任务7.5: Leaderboard (可选)
- [ ] 优化模型 (1.5 H100-hour limit)
- [ ] 目标: val loss < 5.0
- [ ] 提交到: github.com/stanford-cs336/assignment1-basics-leaderboard
- [ ] 可能尝试: weight tying, 架构修改等

---

## 🎯 重要提示

### 不能使用的PyTorch组件
❌ **禁止使用**:
- `torch.nn.functional.*` (除了container classes)
- `torch.nn.Linear`
- `torch.nn.Embedding`
- `torch.nn.RMSNorm` / `LayerNorm`
- `torch.nn.*` 中的任何layer
- `torch.optim.*` (除了Optimizer基类)

✅ **可以使用**:
- `torch.nn.Parameter`
- `torch.nn.Module`, `ModuleList`, `Sequential`
- `torch.optim.Optimizer` (基类)
- 所有其他PyTorch功能 (matmul, sigmoid, etc.)
- `torch.sigmoid` (仅在SwiGLU中,为了数值稳定性)

### 关键技术
- **Einsum notation**: 强烈推荐使用einops/einx
- **数值稳定性**: Softmax, cross-entropy需要特别注意
- **内存优化**: np.memmap, gradient checkpointing
- **性能**: 并行化, GPU优化

### Debug技巧
1. 先在单个batch上过拟合 (loss → 0)
2. 检查中间tensor的shape
3. 监控激活/梯度范数
4. 使用IDE debugger (不是print)
5. Profiling: cProfile, scalene

### Low-Resource Tips (CPU/MPS)
如果在CPU或Apple Silicon上:
- 减少total_tokens到40M
- 目标val loss: 2.0 (而不是1.45)
- 使用torch.compile优化
- MPS: 不要用TF32, 用`backend="aot_eager"`

---

## 📊 进度追踪

### 完成情况
- [ ] 阶段1: Tokenizer (0/4)
- [ ] 阶段2: 基础组件 (0/5)
- [ ] 阶段3: 注意力 (0/3)
- [ ] 阶段4: 完整模型 (0/3)
- [ ] 阶段5: 训练 (0/8)
- [ ] 阶段6: 生成 (0/1)
- [ ] 阶段7: 实验 (0/5)

### 测试状态
运行 `uv run pytest` 查看当前状态

---

## 📚 参考资源

### 论文
- Vaswani et al. 2017: Attention Is All You Need
- Sennrich et al. 2016: BPE
- Su et al. 2021: RoPE
- Shazeer 2020: GLU Variants
- Loshchilov & Hutter 2019: AdamW

### 代码示例
- NanoGPT speedrun: github.com/KellerJordan/modded-nanogpt
- Llama 3: Model card
- Qwen 2.5: Model card

### 重要公式
- **RMSNorm**: x / sqrt(mean(x²) + ε) * g
- **SwiGLU**: W2(SiLU(W1*x) ⊙ W3*x)
- **Attention**: softmax(Q^T K / sqrt(d_k)) V
- **RoPE**: Rotate pairs by θi,k = i/10000^((2k-2)/d)

---

## 下一步行动

1. ✅ 创建git分支 `assignment-implementation`
2. ⏭️ 运行 `uv run pytest` 查看测试状态
3. ⏭️ 下载数据: TinyStories + OpenWebText
4. ⏭️ 开始任务1.1: Unicode问题
5. ⏭️ 实现任务1.2: BPE训练

---

**创建时间**: 2026-01-19
**Git分支**: assignment-implementation
**预计完成**: 3-5周

---

## 🎓 最终毕业条件（终极测试）

完成这份规划，不只是完成 CS336 作业，而是获得"设计 LLM 系统的资格"。

### 理解层（Understanding Layer）
你应该能够回答以下问题，不需要查资料：

#### 🧠 核心概念
- [ ] 我能画出完整 Transformer 信息流图（从 token → embedding → blocks → logits → loss）
- [ ] 我能解释任一模块"如果删除会如何失败"
- [ ] 我能用一句话解释 Transformer 为什么"只能这样设计"

#### 🔍 深度理解测试
<details>
<summary>Q1: 为什么 Transformer 必须有 residual connection？</summary>
标准答案：没有 residual，梯度无法有效传播到浅层，深层网络退化。Residual 提供梯度高速公路，让模型学习"增量变换"而非"全量变换"。移除 residual 会导致深层网络训练困难，甚至比浅层网络效果更差。
</details>

<details>
<summary>Q2: 为什么训练和生成必须用同样的 causal mask？</summary>
标准答案：生成时第 t 步只能看到前 t-1 个 token（因果性）。如果训练时看到了未来信息，会产生 train-inference mismatch。模型训练时依赖的信息在推理时不可用，导致性能崩溃。Mask 保证信息流一致性。
</details>

<details>
<summary>Q3: 为什么 softmax 不应该在模型 forward 中？</summary>
标准答案：(1) 训练时 CE loss 可以与 softmax 融合，数值更稳定；(2) 生成时需要在 softmax 前调整 logits（temperature/top-p）；(3) 模型应该输出"未归一化的分数"，由调用者决定如何转为概率。模型只负责评分，不负责决策。
</details>

<details>
<summary>Q4: Embedding 和 LM Head 是否应该共享权重（weight tying）？</summary>
标准答案：可以共享。Embedding 把 token → vector，LM Head 把 vector → logits（token 空间）。它们操作同一个语义空间的双向映射。共享权重可以：(1) 减少参数量；(2) 强制语义一致性；(3) 但可能限制表达能力。现代模型（GPT-2/3）不共享，但小模型可以共享以节省参数。
</details>

<details>
<summary>Q5: 如果不用 teacher forcing，会怎样？</summary>
标准答案：每步用模型自己生成的 token 作为下一步输入。优点：消除 train-inference mismatch；缺点：(1) 训练慢（需要序列生成）；(2) 早期模型很差，生成噪声会累积；(3) 梯度方差大，训练不稳定。现代 LLM 用 teacher forcing + causal mask 平衡效率和一致性。
</details>

### 实践层（Implementation Layer）

#### ✅ 所有单元测试通过
```bash
uv run pytest  # 100% pass
```

#### ✅ 所有概念测试通过
```bash
uv run python conceptual_tests/test_tokenizer.py
uv run python conceptual_tests/test_embedding.py
uv run python conceptual_tests/test_attention.py
uv run python conceptual_tests/test_transformer_block.py
uv run python conceptual_tests/test_loss.py
uv run python conceptual_tests/test_generation.py
```

#### ✅ 模型训练成功
- [ ] TinyStories val loss ≤ 1.45
- [ ] 能够生成连贯文本（至少 256 tokens）
- [ ] 训练曲线平滑下降（无梯度爆炸/消失）

### 应用层（Application Layer）

你应该能够：
- [ ] 独立设计一个新的 Transformer 变体（如：改变 attention 机制）
- [ ] 解释你的设计选择（不是"试试看"，而是"因为 X 所以 Y"）
- [ ] 预测你的设计的性能特征（速度、内存、表达能力）
- [ ] Debug 训练失败的模型（loss 不降、loss 爆炸、生成重复等）

### 🏆 终极挑战（可选）

完成以下任一项，证明你真正掌握了 Transformer：

1. **架构创新**: 设计并实现一个 Transformer 改进（如：新的 positional encoding、attention 变体），在 TinyStories 上证明它有效。

2. **效率优化**: 在不改变架构的前提下，通过系统优化（如：FlashAttention、混合精度、算子融合）让训练速度提升 2x。

3. **深度分析**: 可视化训练好的模型的 attention patterns，解释不同 head 学到了什么语言学模式（如：语法依赖、共指消解等）。

4. **消融研究**: 系统地测试每个设计选择的影响（RMSNorm vs LayerNorm、RoPE vs Sinusoidal、SwiGLU vs GELU），用实验数据支持你的结论。

---

## 📖 学习资源

### 必读论文
1. **Attention Is All You Need** (Vaswani et al., 2017) - Transformer 原始论文
2. **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021) - RoPE
3. **GLU Variants Improve Transformer** (Shazeer, 2020) - SwiGLU
4. **Root Mean Square Layer Normalization** (Zhang & Sennrich, 2019) - RMSNorm
5. **Decoupled Weight Decay Regularization** (Loshchilov & Hutter, 2019) - AdamW

### 推荐阅读
- **The Illustrated Transformer** (Jay Alammar) - 可视化教程
- **Llama 3 Model Card** - 现代架构实例
- **GPT-2 Paper** (Radford et al., 2019) - 语言模型训练
- **Scaling Laws for Neural Language Models** (Kaplan et al., 2020) - 模型规模研究

### 代码参考（仅供理解，不要抄袭）
- **nanoGPT** (Andrej Karpathy) - 简洁的 GPT 实现
- **minGPT** - 教学用最小实现
- **Hugging Face Transformers** - 生产级实现

---

> **最终目标**: 当你完成这个项目，你应该能够自信地说："我不只是写了一个 Transformer，我理解了为什么现代 LLM 必须这样设计。"

**Good luck! 🚀**
