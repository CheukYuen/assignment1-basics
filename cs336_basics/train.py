"""Training script for CS336 Assignment 1 — Transformer Language Model.

Usage examples
--------------
# Train on TinyStories (default config):
uv run python cs336_basics/train.py \
    --train_data data/tinystories_train.npy \
    --val_data data/tinystories_valid.npy \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 512 \
    --d_ff 1344 \
    --num_layers 4 \
    --num_heads 16 \
    --max_lr 3e-4 \
    --min_lr 3e-5 \
    --warmup_iters 2000 \
    --batch_size 64 \
    --total_iters 20000 \
    --checkpoint_dir checkpoints/ \
    --log_interval 100 \
    --val_interval 500

# Resume from a checkpoint:
uv run python cs336_basics/train.py ... --resume checkpoints/ckpt_10000.pt
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.transformer import (
    AdamW,
    TransformerLM,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train a Transformer LM from scratch.")

    # Data
    p.add_argument("--train_data", type=str, required=True, help="Path to tokenized train .npy file.")
    p.add_argument("--val_data", type=str, required=True, help="Path to tokenized val .npy file.")
    p.add_argument("--data_dtype", type=str, default="uint16", help="numpy dtype of token arrays.")

    # Model
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    # Optimizer
    p.add_argument("--max_lr", type=float, default=3e-4, help="Peak learning rate.")
    p.add_argument("--min_lr", type=float, default=3e-5, help="Minimum learning rate after cosine decay.")
    p.add_argument("--warmup_iters", type=int, default=2000)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient L2 norm (0 to disable).")

    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--total_iters", type=int, default=20000, help="Total training iterations.")
    p.add_argument("--device", type=str, default="", help="Device string, e.g. 'cuda:0'. Auto-detected if empty.")

    # Logging & checkpointing
    p.add_argument("--log_interval", type=int, default=100, help="Log train loss every N steps.")
    p.add_argument("--val_interval", type=int, default=500, help="Evaluate val loss every N steps.")
    p.add_argument("--val_batches", type=int, default=50, help="Number of val batches to average.")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    p.add_argument("--save_interval", type=int, default=5000, help="Save checkpoint every N steps.")
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from.")

    # Mixed precision
    p.add_argument("--use_bf16", action="store_true", help="Use bfloat16 autocast for faster training on CUDA.")

    # W&B (optional)
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb_project", type=str, default="cs336-lm")
    p.add_argument("--wandb_run_name", type=str, default="")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device(device_str: str) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, val_data, batch_size, context_length, device, num_batches, use_bf16: bool = False):
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = get_batch(val_data, batch_size, context_length, str(device))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(x)
            # logits: (B, T, V)  y: (B, T)
            B, T, V = logits.shape
            loss = cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = get_device(args.device)
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")  # TF32 on Ampere/Ada/Blackwell

    # W&B setup
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            config=vars(args),
        )

    # Load tokenized data with memory-mapping (copy=False avoids loading the full file into RAM)
    dtype = np.dtype(args.data_dtype)
    train_data = np.load(args.train_data, mmap_mode="r").astype(dtype, copy=False)
    val_data = np.load(args.val_data, mmap_mode="r").astype(dtype, copy=False)
    print(f"Train tokens: {len(train_data):,}   Val tokens: {len(val_data):,}")

    # Build model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=device,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # Resume from checkpoint if requested
    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from checkpoint at iteration {start_iter}")

    # Checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    model.train()
    start_time = time.time()
    log_loss_acc = 0.0
    log_loss_steps = 0

    for step in range(start_iter, args.total_iters):
        # Learning rate schedule
        lr = get_lr_cosine_schedule(step, args.max_lr, args.min_lr, args.warmup_iters, args.total_iters)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Sample batch
        x, y = get_batch(train_data, args.batch_size, args.context_length, str(device))

        # Forward pass
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.use_bf16):
            logits = model(x)
            B, T, V = logits.shape
            loss = cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))

        # Backward pass
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()

        log_loss_acc += loss.item()
        log_loss_steps += 1

        # Logging
        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            avg_loss = log_loss_acc / log_loss_steps
            tokens_per_sec = (args.log_interval * args.batch_size * args.context_length) / (
                elapsed / ((step - start_iter + 1) / args.log_interval)
            )
            print(
                f"step {step+1:6d} | lr {lr:.2e} | loss {avg_loss:.4f} | "
                f"tok/s {tokens_per_sec:.0f} | elapsed {elapsed:.0f}s"
            )
            if wandb_run:
                wandb_run.log({"train/loss": avg_loss, "train/lr": lr, "train/step": step + 1, "train/time": elapsed})
            log_loss_acc = 0.0
            log_loss_steps = 0

        # Validation
        if (step + 1) % args.val_interval == 0:
            val_loss = evaluate(model, val_data, args.batch_size, args.context_length, device, args.val_batches, args.use_bf16)
            elapsed = time.time() - start_time
            print(f"  [val] step {step+1:6d} | val_loss {val_loss:.4f} | elapsed {elapsed:.0f}s")
            if wandb_run:
                wandb_run.log({"val/loss": val_loss, "train/step": step + 1, "train/time": elapsed})

        # Checkpointing
        if (step + 1) % args.save_interval == 0:
            ckpt_path = ckpt_dir / f"ckpt_{step+1:07d}.pt"
            save_checkpoint(model, optimizer, step + 1, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    ckpt_path = ckpt_dir / f"ckpt_{args.total_iters:07d}_final.pt"
    save_checkpoint(model, optimizer, args.total_iters, ckpt_path)
    print(f"Training complete. Final checkpoint: {ckpt_path}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
