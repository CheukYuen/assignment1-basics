"""Transformer Language Model — built from scratch for CS336 Assignment 1."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable
from typing import IO, BinaryIO, Optional

import numpy as np
import torch
from einops import einsum, rearrange
from torch import Tensor


# ---------------------------------------------------------------------------
# Utility: softmax (from scratch, numerically stable)
# ---------------------------------------------------------------------------

def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax along *dim*."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


# ---------------------------------------------------------------------------
# Utility: SiLU activation
# ---------------------------------------------------------------------------

def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


# ---------------------------------------------------------------------------
# Utility: cross-entropy loss (from scratch)
# ---------------------------------------------------------------------------

def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """Average cross-entropy loss.

    Args:
        inputs: (batch_size, vocab_size) unnormalized logits.
        targets: (batch_size,) integer class indices.
    """
    # max-subtraction for stability
    x_max = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - x_max
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))
    # gather the logit for the correct class
    target_logits = shifted[torch.arange(inputs.size(0), device=inputs.device), targets]
    loss = -target_logits + log_sum_exp
    return loss.mean()


# ---------------------------------------------------------------------------
# Linear (no bias)
# ---------------------------------------------------------------------------

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Store weight as W (d_out, d_in)
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d_in)  W: (d_out, d_in)  → (..., d_out)
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.weight.float()
        return result.to(in_dtype)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            # d_ff ≈ 8/3 * d_model, rounded up to multiple of 64
            d_ff = int(math.ceil((8 / 3) * d_model / 64) * 64)
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        # FFN(x) = W2 * (SiLU(W1 * x) ⊙ W3 * x)
        return self.w2(silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        # Precompute cos/sin buffers: shape (max_seq_len, d_k)
        # k indices: 1-based pair index → 0-based: k = 0, 1, ..., d_k/2 - 1
        k = torch.arange(0, d_k // 2, device=device, dtype=torch.float32)
        # θ_{i,k} = i / Θ^{2k/d}
        freqs = 1.0 / (theta ** (2 * k / d_k))  # (d_k/2,)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)  # (max_seq_len,)
        angles = einsum(positions, freqs, "seq, half -> seq half")  # (max_seq_len, d_k/2)
        self.register_buffer("cos_buf", torch.cos(angles), persistent=False)
        self.register_buffer("sin_buf", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        Args:
            x: (..., seq_len, d_k)
            token_positions: (..., seq_len)  integer positions
        Returns:
            rotated x with same shape
        """
        # Gather cos/sin for the requested positions
        cos = self.cos_buf[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_buf[token_positions]  # (..., seq_len, d_k/2)

        # Split x into pairs
        x1 = x[..., 0::2]  # (..., seq_len, d_k/2)
        x2 = x[..., 1::2]  # (..., seq_len, d_k/2)

        # Apply 2D rotation to each pair
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # Interleave back
        out = torch.stack([out1, out2], dim=-1)  # (..., seq_len, d_k/2, 2)
        return out.reshape(x.shape)


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Args:
        Q: (..., n, d_k)
        K: (..., m, d_k)
        V: (..., m, d_v)
        mask: (..., n, m) boolean — True means attend, False means mask out
    Returns:
        (..., n, d_v)
    """
    d_k = Q.size(-1)
    # QK^T / sqrt(d_k)  →  (..., n, m)
    scores = einsum(Q, K, "... n d, ... m d -> ... n m") / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = softmax(scores, dim=-1)
    # (..., n, m) @ (..., m, d_v) → (..., n, d_v)
    return einsum(attn_weights, V, "... n m, ... m d -> ... n d")


# ---------------------------------------------------------------------------
# Causal Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        device=None,
        dtype=None,
        use_rope: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            token_positions: (batch, seq_len) or None (defaults to 0..seq_len-1)
        """
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to (batch, num_heads, seq_len, d_k)
        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)

        # Apply RoPE to Q and K
        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
            # rope expects (..., seq_len, d_k) — head dim is a batch dim
            Q = self.rope(Q, token_positions.unsqueeze(1).expand(-1, self.num_heads, -1))
            K = self.rope(K, token_positions.unsqueeze(1).expand(-1, self.num_heads, -1))

        # Causal mask: (seq_len, seq_len)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        # Attention
        attn_out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        # (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model)
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")

        return self.output_proj(attn_out)


# ---------------------------------------------------------------------------
# Transformer Block (pre-norm)
# ---------------------------------------------------------------------------

class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        # Sub-layer 1: attention
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        # Sub-layer 2: FFN
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Transformer Language Model
# ---------------------------------------------------------------------------

class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Args:
            token_ids: (batch_size, seq_len) integer tensor
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch, seq_len = token_ids.shape
        token_positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch, -1)

        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

def gradient_clipping(parameters, max_l2_norm: float) -> None:
    """Clip combined gradients to have L2 norm at most max_l2_norm."""
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    total_norm_sq = sum(p.grad.pow(2).sum() for p in params_with_grad)
    total_norm = torch.sqrt(total_norm_sq)
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params_with_grad:
            p.grad.mul_(clip_coef)


# ---------------------------------------------------------------------------
# AdamW Optimizer
# ---------------------------------------------------------------------------

class AdamW(torch.optim.Optimizer):
    """AdamW optimizer following Loshchilov and Hutter (2019) Algorithm 2."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected learning rate
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # Update parameters
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

                # Apply decoupled weight decay
                p.data.mul_(1 - lr * weight_decay)

        return loss


# ---------------------------------------------------------------------------
# Cosine Learning Rate Schedule with Linear Warmup
# ---------------------------------------------------------------------------

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine annealing LR schedule with linear warmup."""
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


# ---------------------------------------------------------------------------
# Data Loading (get_batch)
# ---------------------------------------------------------------------------

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """Sample a batch of (input, target) pairs from a token array.

    Args:
        dataset: 1D integer numpy array of token IDs.
        batch_size: Number of sequences in the batch.
        context_length: Length of each sequence.
        device: PyTorch device string.

    Returns:
        Tuple of LongTensors (x, y) each of shape (batch_size, context_length).
        y[i] = x[i] shifted by one position (next-token targets).
    """
    n = len(dataset)
    # Valid start indices: 0 .. n - context_length - 1  (need context_length+1 tokens)
    starts = np.random.randint(0, n - context_length, size=batch_size)
    x = np.stack([dataset[i : i + context_length] for i in starts])
    y = np.stack([dataset[i + 1 : i + context_length + 1] for i in starts])
    x_t = torch.tensor(x, dtype=torch.long, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    return x_t, y_t


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Save model, optimizer state and iteration count to a file."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load model/optimizer state from a checkpoint file. Returns saved iteration."""
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


# ---------------------------------------------------------------------------
# Text Generation (Decoding)
# ---------------------------------------------------------------------------

def generate(
    model: torch.nn.Module,
    token_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> Tensor:
    """Autoregressively generate tokens from a language model.

    Args:
        model: TransformerLM (or similar) that takes (batch, seq) → (batch, seq, vocab).
        token_ids: (batch, prefix_len) integer prompt tensor.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Softmax temperature (τ). Values < 1 sharpen, > 1 flatten.
        top_p: Nucleus sampling threshold p ∈ (0, 1]. Use 1.0 to disable.
        eos_token_id: If provided, stop generation when this token is sampled.

    Returns:
        (batch, prefix_len + generated_len) integer tensor.
    """
    model.eval()
    generated = token_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate to context_length if needed
            context = generated
            if hasattr(model, "context_length"):
                context = generated[:, -model.context_length :]

            logits = model(context)  # (batch, seq, vocab)
            next_logits = logits[:, -1, :]  # (batch, vocab)

            # Temperature scaling
            if temperature != 1.0:
                next_logits = next_logits / temperature

            probs = softmax(next_logits, dim=-1)  # (batch, vocab)

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # Remove tokens where cumulative prob exceeds top_p
                # Keep at least one token (the most probable)
                sorted_probs_to_remove = cumulative_probs - sorted_probs > top_p
                sorted_probs[sorted_probs_to_remove] = 0.0
                # Renormalize
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                # Scatter back to original ordering
                probs = torch.zeros_like(sorted_probs).scatter_(1, sorted_indices, sorted_probs)

            next_tokens = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            generated = torch.cat([generated, next_tokens], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break

    return generated
