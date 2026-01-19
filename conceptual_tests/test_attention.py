"""
Conceptual Tests for Attention Mechanism

These tests verify understanding of:
1. Attention as weighted averaging of Values
2. Causal mask preventing future information leakage
3. Multi-head attention learning diverse patterns

Run: uv run python conceptual_tests/test_attention.py
"""

import torch
import math
import sys
from pathlib import Path

# Add parent directory to path to import your implementation
sys.path.insert(0, str(Path(__file__).parent.parent))

# TODO: Import your implementation
# from cs336_basics.attention import scaled_dot_product_attention, MultiHeadSelfAttention


def test_attention_value_averaging():
    """
    MVT-Attention-01: Value Averaging Test

    Goal: Verify that attention output is always a linear combination of V

    Key Insight: Attention doesn't create new information, it only mixes existing information
    """
    print("\n" + "="*70)
    print("MVT-Attention-01: Value Averaging Test")
    print("="*70)

    d_k, seq_len = 64, 10

    # Create uniform Q and K (all positions get equal attention)
    Q = torch.ones(1, seq_len, d_k)
    K = torch.ones(1, seq_len, d_k)
    V = torch.randn(1, seq_len, d_k)

    # Compute attention manually
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    attn_weights = torch.softmax(scores, dim=-1)
    output = attn_weights @ V

    # Expected: output should be the mean of V (since all weights are equal)
    expected = V.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)

    # Verify
    assert torch.allclose(output, expected, atol=1e-5), \
        f"Output should equal V.mean(), but got max diff: {(output - expected).abs().max()}"

    print("✓ PASSED: Attention output = weighted average of V")
    print(f"  → Attention weights shape: {attn_weights.shape}")
    print(f"  → All weights ≈ 1/{seq_len} = {1/seq_len:.4f}")
    print(f"  → Attention weight sample: {attn_weights[0, 0, :5]}")
    print("\n  Key Understanding:")
    print("  - Attention output is ALWAYS a convex combination of V")
    print("  - Q·K determines the mixing weights")
    print("  - Attention cannot create information not present in V")
    print("  - This is why we need FFN layers to add new information!\n")


def test_causal_mask_prevents_leakage():
    """
    MVT-Mask-01: Causality Verification

    Goal: Verify that causal mask prevents future information leakage

    Key Insight: This is not an optimization—it's necessary for autoregressive consistency
    """
    print("\n" + "="*70)
    print("MVT-Mask-01: Causality Verification")
    print("="*70)

    d_k, seq_len = 64, 5
    Q = torch.randn(1, seq_len, d_k)
    K = torch.randn(1, seq_len, d_k)
    V = torch.randn(1, seq_len, d_k)

    # Create causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

    # Compute attention with mask
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores.masked_fill(~mask, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)

    # Verify: upper triangle should be zero
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert attn_weights[0, i, j] < 1e-6, \
                f"Position {i} should not attend to future position {j}"

    print("✓ PASSED: Causal mask prevents future attention")
    print(f"\n  Attention weights matrix (should be lower triangular):")
    print(f"  {attn_weights[0].detach().numpy()}")
    print("\n  Key Understanding:")
    print("  - Position t can only attend to positions ≤ t")
    print("  - This matches generation: at step t, we only see tokens 0...t-1")
    print("  - Without mask, training sees future → inference fails (train-test mismatch)")
    print("  - Mask is not optional—it's required for autoregressive models!\n")

    # Optional: visualize
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        plt.imshow(attn_weights[0].detach().numpy(), cmap='viridis')
        plt.title("Causal Attention Weights")
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        plt.colorbar(label="Attention Weight")

        # Create plots directory if it doesn't exist
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(exist_ok=True)

        plt.savefig(plots_dir / "causal_mask_weights.png", dpi=150, bbox_inches='tight')
        print(f"  Visualization saved to: {plots_dir / 'causal_mask_weights.png'}\n")
        plt.close()
    except ImportError:
        print("  (matplotlib not installed, skipping visualization)\n")


def test_scaling_factor_importance():
    """
    MVT-Attention-02: Scaling Factor Test

    Goal: Understand why we divide by √d_k

    Key Insight: Prevents softmax saturation in high dimensions
    """
    print("\n" + "="*70)
    print("MVT-Attention-02: Scaling Factor Test")
    print("="*70)

    seq_len = 10

    for d_k in [8, 64, 512]:
        Q = torch.randn(1, seq_len, d_k)
        K = torch.randn(1, seq_len, d_k)

        # Without scaling
        scores_unscaled = Q @ K.transpose(-2, -1)
        attn_unscaled = torch.softmax(scores_unscaled, dim=-1)

        # With scaling
        scores_scaled = scores_unscaled / math.sqrt(d_k)
        attn_scaled = torch.softmax(scores_scaled, dim=-1)

        # Measure entropy (higher = more uniform distribution)
        def entropy(p):
            return -(p * torch.log(p + 1e-9)).sum(dim=-1).mean()

        entropy_unscaled = entropy(attn_unscaled)
        entropy_scaled = entropy(attn_scaled)

        print(f"\n  d_k = {d_k}:")
        print(f"    Score variance (unscaled): {scores_unscaled.var():.4f}")
        print(f"    Score variance (scaled):   {scores_scaled.var():.4f}")
        print(f"    Attention entropy (unscaled): {entropy_unscaled:.4f}")
        print(f"    Attention entropy (scaled):   {entropy_scaled:.4f}")
        print(f"    Max attention weight (unscaled): {attn_unscaled.max():.4f}")
        print(f"    Max attention weight (scaled):   {attn_scaled.max():.4f}")

    print("\n  Key Understanding:")
    print("  - Without scaling: as d_k grows, scores grow → softmax saturates")
    print("  - Saturated softmax ≈ one-hot → gradient vanishing")
    print("  - Scaling by √d_k keeps score variance ≈ O(1)")
    print("  - This is mathematically derived, not empirically tuned!\n")


def main():
    """Run all conceptual tests"""
    print("\n" + "="*70)
    print("ATTENTION MECHANISM: CONCEPTUAL UNDERSTANDING TESTS")
    print("="*70)
    print("\nThese tests verify understanding, not implementation.")
    print("If you haven't implemented attention yet, some tests may fail.")
    print("That's OK—come back after implementation!\n")

    try:
        test_attention_value_averaging()
    except Exception as e:
        print(f"❌ FAILED: {e}\n")

    try:
        test_causal_mask_prevents_leakage()
    except Exception as e:
        print(f"❌ FAILED: {e}\n")

    try:
        test_scaling_factor_importance()
    except Exception as e:
        print(f"❌ FAILED: {e}\n")

    print("="*70)
    print("CONCEPTUAL TESTS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the 'Key Understanding' sections above")
    print("2. Can you explain each concept without looking?")
    print("3. Try modifying the tests to explore edge cases")
    print("4. Move on to test_transformer_block.py\n")


if __name__ == "__main__":
    main()
