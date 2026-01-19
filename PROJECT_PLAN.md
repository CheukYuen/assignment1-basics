# CS336 Assignment 1 项目规划

## 项目背景
- **课程**: Stanford CS336 - Language Modeling from Scratch
- **作业**: Assignment 1 - 从零构建Transformer语言模型
- **分支**: `assignment-implementation`
- **目标**: 学习智能体开发,通过实现完整的LM训练pipeline

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
```

---

## 📋 详细实施计划

### 阶段1: Tokenizer (2-3天) 🔴 HIGH PRIORITY

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

---

### 阶段2: 模型基础组件 (3-4天) 🔴 HIGH PRIORITY

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

---

### 阶段3: 注意力机制 (2-3天) 🔴 HIGH PRIORITY

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

---

### 阶段4: 完整模型 (1-2天) 🔴 HIGH PRIORITY

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

---

### 阶段5: 训练基础设施 (2-3天) 🔴 HIGH PRIORITY

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

---

### 阶段6: 文本生成 (1天) 🟡 MEDIUM PRIORITY

#### 任务6.1: Decoding (decoding)
- [ ] 实现基础采样 (从softmax分布采样)
- [ ] Temperature scaling: softmax(v/τ)
- [ ] Top-p (nucleus) sampling
- [ ] 生成直到<|endoftext|>或max_tokens
- [ ] 支持batch生成
- **示例输出**: 至少256 tokens

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
