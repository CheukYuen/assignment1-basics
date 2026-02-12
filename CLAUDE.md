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
- `data/` — Training data (not committed)
- `debug/` — Scratch/debug scripts

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
