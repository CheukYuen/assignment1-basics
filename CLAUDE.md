# CLAUDE.md

## Project Overview

Stanford CS336 (Spring 2025) Assignment 1: Basics. Build all components needed to train a Transformer language model from scratch in PyTorch.

**Constraint**: Cannot use `torch.nn`, `torch.nn.functional`, or `torch.optim` except for:
- `torch.nn.Parameter`
- Container classes (`Module`, `ModuleList`, `Sequential`, etc.)
- `torch.optim.Optimizer` base class
- `torch.sigmoid` (allowed for numerical stability in SiLU)

## Repository Structure

- `cs336_basics/` — Main package (implementation code goes here)
- `tests/` — Unit tests; `tests/adapters.py` bridges implementations to tests
- `conceptual_tests/` — Conceptual understanding tests
- `docs/` — Documentation
- `data/` — Training data + serialized BPE artifacts (see below)
- `debug/` — Scratch/debug scripts

### tests/ 目录结构

| 文件 | 作用 | 对应 adapter |
|------|------|-------------|
| `adapters.py` | 胶水层，将测试与实现解耦；每个 `run_*` / `get_*` 需要填写实现调用 | — |
| `common.py` | 共享工具：`FIXTURES_PATH`、`gpt2_bytes_to_unicode()` | — |
| `conftest.py` | pytest fixtures（snapshot、ts_state_dict、参数化输入等）| — |
| `test_train_bpe.py` | BPE 训练：速度(<1.5s)、正确性、特殊 token | `run_train_bpe` |
| `test_tokenizer.py` | encode/decode 正确性、与 tiktoken 对比、内存用量 | `get_tokenizer` |
| `test_model.py` | Linear/Embedding/RMSNorm/SwiGLU/RoPE/Attention/MHA/Block/LM | `run_linear` 等 |
| `test_nn_utils.py` | softmax、cross_entropy、gradient_clipping | 对应 adapter |
| `test_optimizer.py` | AdamW、cosine LR schedule | `get_adamw_cls` 等 |
| `test_data.py` | get_batch 数据加载 | `run_get_batch` |
| `test_serialization.py` | model/optimizer checkpoint save+load | `run_save_checkpoint` 等 |
| `fixtures/` | 小型测试语料（`corpus.en`、GPT-2 vocab/merges、TinyStories 样本）、BPE 参考输出 | — |
| `_snapshots/` | 预计算的参考数值输出（`.npz`/`.pkl`），用于正确性验证 | — |

### data/ 目录 — 所有文件均仅本地，不上传

| 文件 | 来源 | 对应作业问题 |
|------|------|------------|
| `TinyStoriesV2-GPT4-train/valid.txt` | 课程提供 | BPE 训练 + LM 训练输入 |
| `owt_train.txt` / `owt_valid.txt` | 课程提供 | OWT 实验输入 |
| `tinystories_vocab.json` | **已生成** | Problem: train_bpe_tinystories（§2.5）|
| `tinystories_merges.txt` | **已生成** | Problem: train_bpe_tinystories（§2.5）|
| `tinystories_train.npy` | **已生成** ~1.33B tokens | Problem: tokenizer_experiments（§2.7）|
| `tinystories_valid.npy` | **已生成** ~13.4M tokens | Problem: tokenizer_experiments（§2.7）|
| `owt_train.npy` *(待生成)* | tokenizer_experiments 产物 | Problem: tokenizer_experiments（§2.7）|
| `checkpoints/ckpt_0005000.pt` | **已生成** step 5000 | Problem: checkpointing（§5.2）|
| `checkpoints/ckpt_0020000.pt` | **已生成** step 20000 | Problem: checkpointing（§5.2）|
| `checkpoints/ckpt_0020000_final.pt` | **已生成** 最终模型 val_loss=0.7534 | Problem: training_together（§5.3）|

### 提交物（需要上传的只有三项）

| 产物 | 目标 | 内容 |
|------|------|------|
| `writeup.pdf` | Gradescope | 所有书面回答（unicode、BPE 分析、FLOPs 等）|
| `code.zip` | Gradescope | `cs336_basics/` + `tests/adapters.py` 实现代码 |
| leaderboard 结果 | GitHub PR（assignment1-basics-leaderboard）| 最终 val loss + 学习曲线 |

## 作业进度

> 依据 `tests/adapters.py` 中是否还有 `raise NotImplementedError` 判断代码完成状态。
> ✅ 已完成 | ⬜ 待做

| Problem | pts | Code | Written |
|---------|-----|------|---------|
| **Part 2: BPE Tokenizer** | | | |
| unicode1, unicode2 | 4pt | — | ⬜ |
| train_bpe | 15pt | ✅ | — |
| tokenizer (encode/decode) | 15pt | ✅ | — |
| train_bpe_tinystories | 2pt | ✅ (data/已生成) | ⬜ 写耗时/最长token |
| train_bpe_expts_owt | 2pt | ⬜ 需跑OWT | ⬜ |
| tokenizer_experiments | 4pt | 🔶 train/valid.npy已生成，owt.npy待生成，writeup待写 | ⬜ |
| **Part 3: Transformer LM** | | | |
| linear | 1pt | ✅ | — |
| embedding | 1pt | ✅ | — |
| rmsnorm | 1pt | ✅ | — |
| positionwise_feedforward (SwiGLU) | 2pt | ✅ | — |
| rope | 2pt | ✅ | — |
| softmax | 1pt | ✅ | — |
| scaled_dot_product_attention | 5pt | ✅ | — |
| multihead_self_attention | 5pt | ✅ | — |
| transformer_block | 3pt | ✅ | — |
| transformer_lm | 3pt | ✅ | — |
| transformer_accounting (FLOPs) | 5pt | — | ⬜ |
| **Part 4: Training** | | | |
| cross_entropy | 1pt | ✅ | — |
| gradient_clipping | 1pt | ✅ | — |
| adamw | 2pt | ✅ | — |
| learning_rate_schedule | 1pt | ✅ | — |
| **Part 5: Training Loop** | | | |
| data_loading (get_batch) | 2pt | ✅ | — |
| checkpointing | 1pt | ✅ | — |
| training_together | 4pt | ✅ (cs336_basics/train.py) 训练完成 val_loss=0.7534 | — |
| **Part 6: Text Generation** | | | |
| decoding | 3pt | ✅ (transformer.generate + cs336_basics/generate.py) | — |
| **Part 7: Experiments** | | | |
| learning_rate sweep | 3pt | ⬜ | ⬜ |
| batch_size_experiment | 1pt | ⬜ | ⬜ |
| generate | 1pt | ✅ (generate.py 已验证生成>256 tokens) | ⬜ writeup |
| layer_norm_ablation | 1pt | ⬜ | ⬜ |
| pre_norm_ablation | 1pt | ⬜ | ⬜ |
| no_pos_emb | 1pt | ⬜ | ⬜ |
| swiglu_ablation | 1pt | ⬜ | ⬜ |
| main_experiment (OWT) | 2pt | ⬜ | ⬜ |
| leaderboard | 6pt | ⬜ | ⬜ |

## Build & Run

- **Package manager**: `uv` (not pip)
- **Python**: >=3.11
- **Run any script**: `uv run <file>`
- **Run all tests**: `uv run pytest`
- **Run a specific test**: `uv run pytest tests/test_model.py::test_linear -v`
- **Run conceptual tests**: `uv run python conceptual_tests/test_attention.py`
- **BPE 轻量验证**: `uv run python debug/bpe_mini_test.py` (无需下载数据集，手工数据验证 BPE 训练/编码/解码)
- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`

## Code Style & Conventions

- Line length: 120 characters (ruff)
- Type hints use `jaxtyping` for tensor shapes (e.g., `Float[Tensor, "batch seq d_model"]`)
- Tensor operations use `einops` and `einx` (strongly recommended over raw reshape/transpose)
- Regex uses the `regex` library (not stdlib `re`)
- Ruff ignores: `F722` (jaxtyping annotations), plus relaxed rules in `__init__.py`

## Key Implementation Pattern

Implementations live in `cs336_basics/`. Tests call through adapter functions in `tests/adapters.py`. When implementing a new component:
1. Write the implementation in `cs336_basics/`
2. Wire it up in `tests/adapters.py` (replace `raise NotImplementedError`)
3. Run the corresponding test to verify

## Git Workflow

- Main branch: `main`
- Working branch: `assignment-implementation`
- Remote `private` for pushing work; `origin` is the upstream course repo

---

## Assignment Components & Problems

### Part 2: BPE Tokenizer

#### 2.1-2.2 Unicode Basics (Problems: unicode1, unicode2)
- `chr(0)` → NULL character (U+0000); `repr` shows `'\x00'`, `print` shows nothing
- UTF-8 encoding: `str.encode("utf-8")` → bytes; `bytes.decode("utf-8")` → str
- UTF-8 preferred over UTF-16/UTF-32 (compact for ASCII, dominant on web)

#### 2.3-2.4 BPE Training (Problem: train_bpe — 15 pts)
- Pre-tokenization regex (GPT-2 style): `r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""`
- Use `regex.finditer` (not `findall`) for memory efficiency
- Strip special tokens before pre-tokenization; split on them so no merging across boundaries
- Initial vocab: 256 byte values; merges add to vocab until `vocab_size` reached
- Tie-breaking: prefer lexicographically greater pair (`max()` in Python)
- Optimization: incrementally update pair counts instead of re-scanning all pairs each merge
- Parallelization: chunk corpus at `<|endoftext|>` boundaries, use `multiprocessing`
- Adapter: `adapters.run_train_bpe` → `uv run pytest tests/test_train_bpe.py`

#### 2.5 BPE Training Experiments (Problems: train_bpe_tinystories, train_bpe_expts_owt)
- TinyStories: vocab_size=10,000, special token `<|endoftext|>`. Target: ≤30 min, ≤30GB RAM
- OpenWebText: vocab_size=32,000. Target: ≤12 hrs, ≤100GB RAM

#### 2.6 BPE Encoding/Decoding (Problem: tokenizer — 15 pts)
- Tokenizer class with `__init__`, `from_files`, `encode`, `encode_iterable`, `decode`
- Encoding: pre-tokenize → apply merges in order → map to IDs
- Decoding: ID → bytes lookup → concatenate → `bytes.decode("utf-8", errors="replace")`
- `encode_iterable`: lazy generator for memory-efficient tokenization of large files
- Special tokens must be preserved as single tokens during encoding
- Adapter: `adapters.get_tokenizer` → `uv run pytest tests/test_tokenizer.py`

#### 2.7 Tokenizer Experiments (Problem: tokenizer_experiments — 4 pts)
- Measure compression ratio (bytes/token) on TinyStories and OWT samples
- Cross-tokenizer evaluation, throughput estimation
- Serialize tokenized data as `np.uint16` arrays

### Part 3: Transformer Language Model

#### 3.4.1 Parameter Initialization
- Linear weights: `trunc_normal_(mean=0, std=sqrt(2/(d_in+d_out)))`, truncated at [-3σ, 3σ]
- Embedding: `trunc_normal_(mean=0, std=1)`, truncated at [-3, 3]
- RMSNorm gain: initialized to 1

#### 3.4.2 Linear Module (Problem: linear — 1 pt)
- Custom `nn.Module`, no bias, store weight as W (not W^T)
- Adapter: `adapters.run_linear` → `uv run pytest -k test_linear`

#### 3.4.3 Embedding Module (Problem: embedding — 1 pt)
- Custom `nn.Module`, no `nn.Embedding`
- Adapter: `adapters.run_embedding` → `uv run pytest -k test_embedding`

#### 3.5.1 RMSNorm (Problem: rmsnorm — 1 pt)
- `RMSNorm(a_i) = (a_i / RMS(a)) * g_i` where `RMS(a) = sqrt(mean(a^2) + eps)`
- Upcast to float32 before computation, downcast back afterward
- Adapter: `adapters.run_rmsnorm` → `uv run pytest -k test_rmsnorm`

#### 3.5.2 Feed-Forward / SwiGLU (Problem: positionwise_feedforward — 2 pts)
- `FFN(x) = W2 * (SiLU(W1 * x) ⊙ W3 * x)`
- `SiLU(x) = x * sigmoid(x)` — may use `torch.sigmoid`
- `d_ff ≈ (8/3) * d_model`, rounded to multiple of 64
- Adapter: `adapters.run_swiglu` → `uv run pytest -k test_swiglu`

#### 3.5.3 RoPE (Problem: rope — 2 pts)
- Rotary Position Embeddings: rotate pairs of query/key dimensions
- `θ_{i,k} = i / Θ^((2k-2)/d)`, Θ=10000 by default
- Block-diagonal rotation matrix; implement efficiently without constructing full matrix
- Can precompute sin/cos buffers with `register_buffer(persistent=False)`
- Apply to Q and K only, not V; head dimension is a batch dimension
- Adapter: `adapters.run_rope` → `uv run pytest -k test_rope`

#### 3.5.4 Scaled Dot-Product Attention (Problem: scaled_dot_product_attention — 5 pts)
- `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V`
- Implement softmax from scratch with max-subtraction for numerical stability
- Support boolean mask: True = attend, False = -inf before softmax
- Handle arbitrary batch dimensions
- Softmax adapter: `adapters.run_softmax` → `uv run pytest -k test_softmax_matches_pytorch`
- Attention adapter: `adapters.run_scaled_dot_product_attention` → `uv run pytest -k test_scaled_dot_product_attention`

#### 3.5.5 Causal Multi-Head Self-Attention (Problem: multihead_self_attention — 5 pts)
- `d_k = d_v = d_model / num_heads`
- Q, K, V projections → split into heads → attention per head → concat → output projection
- Causal mask: token i attends only to positions j ≤ i
- RoPE applied to Q and K after projection, before attention
- Adapter: `adapters.run_multihead_self_attention` → `uv run pytest -k test_multihead_self_attention`

#### 3.6 Transformer Block (Problem: transformer_block — 3 pts)
- Pre-norm: `y = x + SubLayer(RMSNorm(x))` for both attention and FFN sublayers
- Adapter: `adapters.run_transformer_block` → `uv run pytest -k test_transformer_block`

#### 3.6 Full Transformer LM (Problem: transformer_lm — 3 pts)
- Token embedding → num_layers Transformer blocks → RMSNorm → linear output head → logits
- Input: (batch_size, seq_len) int tensor → Output: (batch_size, seq_len, vocab_size) logits
- Adapter: `adapters.run_transformer_lm` → `uv run pytest -k test_transformer_lm`

#### 3.6 Resource Accounting (Problem: transformer_accounting — 5 pts, written)
- Count parameters and FLOPs for matrix multiplies
- Rule: A(m×n) × B(n×p) = 2mnp FLOPs
- GPT-2 XL config: vocab=50257, ctx=1024, layers=48, d_model=1600, heads=25, d_ff=6400

### Part 4: Training

#### 4.1 Cross-Entropy Loss (Problem: cross_entropy)
- `loss = -log(softmax(logits)[target])`, with max-subtraction and log-exp cancellation
- Perplexity: `exp(mean(losses))`
- Adapter: `adapters.run_cross_entropy` → `uv run pytest -k test_cross_entropy`

#### 4.3 AdamW Optimizer (Problem: adamw — 2 pts)
- Subclass `torch.optim.Optimizer`
- State: first moment m, second moment v per parameter
- Includes bias correction and decoupled weight decay
- Typical: β1=0.9, β2=0.95 (for LLMs), eps=1e-8
- Adapter: `adapters.get_adamw_cls` → `uv run pytest -k test_adamw`

#### 4.4 Cosine LR Schedule (Problem: learning_rate_schedule)
- Warmup: linear from 0 to α_max over T_w steps
- Cosine decay: α_max → α_min over T_w to T_c steps
- Post-annealing: constant α_min after T_c
- Adapter: `adapters.get_lr_cosine_schedule` → `uv run pytest -k test_get_lr_cosine_schedule`

#### 4.5 Gradient Clipping (Problem: gradient_clipping — 1 pt)
- Compute global L2 norm of all gradients; scale down if > max_norm
- eps = 1e-6 for stability
- Adapter: `adapters.run_gradient_clipping` → `uv run pytest -k test_gradient_clipping`

### Part 5: Training Loop

#### 5.1 Data Loading (Problem: data_loading — 2 pts)
- Input: numpy array of token IDs (use `np.memmap` for large files)
- Random sample batch_size starting positions; extract (input, target) pairs of context_length
- Adapter: `adapters.run_get_batch` → `uv run pytest -k test_get_batch`

#### 5.2 Checkpointing (Problem: checkpointing — 1 pt)
- Save/load: model state_dict, optimizer state_dict, iteration number
- Use `torch.save` / `torch.load`
- Adapter: `adapters.run_save_checkpoint`, `adapters.run_load_checkpoint` → `uv run pytest -k test_checkpointing`

#### 5.3 Training Script (Problem: training_together — 4 pts)
- Configurable hyperparameters, memory-efficient data loading, checkpointing, logging (W&B)

### Part 6: Text Generation (Problem: decoding — 3 pts)
- Autoregressive decoding: feed prefix, sample next token, append, repeat
- Temperature scaling: `softmax(logits / τ)`
- Top-p (nucleus) sampling: truncate to smallest set of tokens with cumulative prob ≥ p

### Part 7: Experiments

#### Base Model Hyperparameters (TinyStories)
- vocab_size=10000, context_length=256, d_model=512, d_ff=1344
- num_layers=4, num_heads=16, RoPE θ=10000
- Total tokens: 327,680,000
- Target validation loss: ≤1.45 (or ≤2.00 on CPU/MPS with 40M tokens)

#### Experiment Problems
- **learning_rate** (3 pts): LR sweep, find optimal, "edge of stability"
- **batch_size_experiment** (1 pt): Vary batch size 1 to GPU limit
- **generate** (1 pt): Generate ≥256 tokens from trained model
- **layer_norm_ablation** (1 pt): Remove RMSNorm, observe instability
- **pre_norm_ablation** (1 pt): Switch to post-norm, compare
- **no_pos_emb** (1 pt): Remove RoPE (NoPE), compare to RoPE
- **swiglu_ablation** (1 pt): SwiGLU vs SiLU (no gating, d_ff=4*d_model)
- **main_experiment** (2 pts): Train on OpenWebText, compare to TinyStories
- **leaderboard** (6 pts): Minimize OWT validation loss in ≤1.5 H100 hours

#### Low-Resource Tips
- Train on TinyStories validation set first as debug dataset
- On CPU: `torch.compile(model)`
- On MPS (Apple Silicon): `torch.compile(model, backend="aot_eager")`, do NOT use TF32
- Cosine schedule should decay to min LR at exactly the final training step

---

## 运行环境

- CPU: Intel Core i9-12700K
- GPU: NVIDIA GeForce RTX 5070 Ti (16GB VRAM, sm_120/Blackwell)
  - ✅ CUDA 正常工作，`torch.cuda.is_available() == True`
  - ✅ sm_120 已在 `arch_list` 中：`['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']`
  - **PyTorch**: `2.12.0.dev20260220+cu128`（nightly，内置 CUDA 12.8 运行时）
  - **解决方案**: `pyproject.toml` 已配置 `pytorch-nightly-cu128` 索引，`uv sync` 锁定 nightly 不降级
- 系统 nvcc: 12.0（无需升级，torch 使用内置 CUDA 12.8 库）
- 单线程 BPE 分词速度：约 7,500 lines/s

---

## 数据文件状态

| 文件 | 状态 | 说明 |
|------|------|------|
| `data/tinystories_vocab.json` | ✅ 已生成 | vocab_size=10000 |
| `data/tinystories_merges.txt` | ✅ 已生成 | 9734 有效 merge 规则 |
| `data/tinystories_valid.npy` | ✅ 已生成 | 13,396,709 tokens, uint16 |
| `data/tinystories_train.npy` | ✅ 已生成 | ~1.33B tokens, uint16 |
| `data/owt_train.npy` | ⬜ 需生成 | — |
| `data/owt_valid.npy` | ⬜ 需生成 | — |
| `checkpoints/ckpt_0005000.pt` | ✅ 已生成 | step 5000 中间 checkpoint |
| `checkpoints/ckpt_0020000.pt` | ✅ 已生成 | step 20000 checkpoint |
| `checkpoints/ckpt_0020000_final.pt` | ✅ 已生成 | 最终模型，val_loss=0.7534 |

---

## 新增文件（本次实现）

| 文件 | 说明 |
|------|------|
| `cs336_basics/train.py` | 完整训练脚本，支持 CLI 配置、memmap、W&B、断点续训、bfloat16、TF32 |
| `cs336_basics/tokenize_data.py` | 将原始文本分词并保存为 .npy（含进度报告） |
| `cs336_basics/generate.py` | 文本生成脚本，支持单次/交互模式，temperature + top-p 采样 |

### transformer.py 新增函数/类

| 名称 | 说明 |
|------|------|
| `AdamW` | AdamW 优化器（含偏差修正、解耦权重衰减） |
| `get_lr_cosine_schedule` | 带线性预热的余弦退火学习率调度 |
| `get_batch` | 从 numpy token 数组采样 (x, y) 批次 |
| `save_checkpoint` / `load_checkpoint` | 保存/恢复模型+优化器+迭代次数 |
| `generate` | 自回归文本生成（支持温度缩放、top-p 采样） |

---

## 关键 Bug 修复记录

### bpe.py `from_files` merges 解析 Bug

**问题**：`line.strip().split()` 丢失行首空格，导致 6255 条以空格字符开头的 merge 规则被跳过。
例：`  t`（空格+t 两个 token）被 `split()` 解析为 `['t']`（1个元素），不满足 `len==2` 直接跳过。

**现象**：`tokenizer.encode_iterable()` 抛出 `KeyError: b'li'`（merge 产生的 token 不在 vocab 中）。

**修复**（`bpe.py` `from_files` 方法）：
```python
# 旧：
line = line.strip()
parts = line.split()
token1 = parts[0].encode("utf-8")

# 新：
line = line.rstrip("\n")
if not line or "\ufffd" in line:
    continue
parts = line.rsplit(" ", 1)          # rsplit 保留行首空格
token1 = parts[0].encode("utf-8")   # merges 文件以 raw bytes 写入后 UTF-8 读回
```

**附注**：
- merges 文件以 `errors='replace'` 打开，跳过含 `\ufffd` 的行（10 行非 ASCII byte merge，英文训练集不影响）
- vocab JSON 仍用 `encode("latin-1")`，merges 用 `encode("utf-8")`（文件编码不同）

---

## 训练命令（✅ 已完成训练）

### 实际训练结果
- **耗时**：48.9 分钟（RTX 5070 Ti，bfloat16 + TF32，~111k tok/s）
- **最终 val loss**：0.7534（远超作业目标 ≤1.45）
- **checkpoint**：`checkpoints/ckpt_0020000_final.pt`
- **日志**：`logs/train_tinystories.log`

### 复现命令（nohup 后台运行）
```bash
mkdir -p checkpoints logs
nohup uv run python cs336_basics/train.py \
    --train_data data/tinystories_train.npy \
    --val_data   data/tinystories_valid.npy \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 512 \
    --d_ff 1344 \
    --num_layers 4 \
    --num_heads 16 \
    --max_lr 3e-4 \
    --min_lr 3e-5 \
    --warmup_iters 2000 \
    --total_iters 20000 \
    --batch_size 64 \
    --use_bf16 \
    --checkpoint_dir checkpoints/ \
    --log_interval 100 \
    --val_interval 500 \
    --save_interval 5000 \
    > logs/train_tinystories.log 2>&1 &
# 总 tokens = 64 * 20000 * 256 = 327,680,000 (满足作业要求)
```

### 生成推理命令
```bash
# 单次生成
uv run python cs336_basics/generate.py \
    --checkpoint checkpoints/ckpt_0020000_final.pt \
    --prompt "Once upon a time" \
    --max_new_tokens 300 --temperature 0.8 --top_p 0.95

# 交互模式
uv run python cs336_basics/generate.py \
    --checkpoint checkpoints/ckpt_0020000_final.pt
```
