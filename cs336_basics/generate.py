"""Text generation script for the trained TinyStories Transformer LM.

Usage
-----
# Interactive mode (default):
uv run python cs336_basics/generate.py \
    --checkpoint checkpoints/ckpt_0020000_final.pt \
    --vocab data/tinystories_vocab.json \
    --merges data/tinystories_merges.txt

# Single prompt:
uv run python cs336_basics/generate.py \
    --checkpoint checkpoints/ckpt_0020000_final.pt \
    --vocab data/tinystories_vocab.json \
    --merges data/tinystories_merges.txt \
    --prompt "Once upon a time" \
    --max_new_tokens 200 \
    --temperature 0.8 \
    --top_p 0.95
"""

from __future__ import annotations

import argparse

import torch

from cs336_basics.bpe import Tokenizer
from cs336_basics.transformer import TransformerLM, generate, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file.")
    p.add_argument("--vocab", type=str, default="data/tinystories_vocab.json")
    p.add_argument("--merges", type=str, default="data/tinystories_merges.txt")
    p.add_argument("--special_tokens", type=str, nargs="*", default=["<|endoftext|>"])

    # Model (must match training config)
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    # Generation
    p.add_argument("--prompt", type=str, default="", help="Prompt text. Empty = interactive mode.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def get_device(s: str) -> torch.device:
    if s:
        return torch.device(s)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_generate(model, tokenizer, prompt: str, args, device: torch.device) -> str:
    ids = tokenizer.encode(prompt) if prompt else []
    if not ids:
        # Start from <|endoftext|> to let the model free-generate a story
        ids = tokenizer.encode("<|endoftext|>")

    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    output = generate(
        model,
        input_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    generated_ids = output[0, len(ids):].tolist()
    return tokenizer.decode(generated_ids)


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=args.special_tokens,
    )

    # Build model (must match training hyperparameters)
    print("Building model...")
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

    # Load checkpoint (weights only, ignore optimizer state)
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    saved_iter = ckpt.get("iteration", "?")
    print(f"Loaded checkpoint (iteration {saved_iter})\n")

    if args.prompt:
        # Single-shot generation
        print(f"Prompt: {args.prompt!r}\n")
        text = run_generate(model, tokenizer, args.prompt, args, device)
        print(text)
    else:
        # Interactive loop
        print("Interactive mode — type a prompt and press Enter. Empty input = free generate. Ctrl-C to quit.\n")
        while True:
            try:
                prompt = input("Prompt> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break
            text = run_generate(model, tokenizer, prompt, args, device)
            print("\n" + "─" * 60)
            print(prompt + text)
            print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
