# Conceptual Tests

This directory contains **Minimum Viable Tests (MVT)** to verify conceptual understanding of Transformer components.

## Purpose

Unlike unit tests that verify implementation correctness, these tests verify that you understand:
- **Why** each component exists
- **What happens** when you modify or remove components
- **How** components interact with each other

## Philosophy

> "I understand a component when I can predict how it fails." — Understanding-Driven Development

## Test Structure

Each test file corresponds to a major component:

- `test_tokenizer.py` - Tokenization and vocab size impact
- `test_embedding.py` - Semantic space and embedding behavior
- `test_attention.py` - Attention mechanism, mask, and multi-head
- `test_transformer_block.py` - Residual, normalization, and block structure
- `test_loss.py` - Cross-entropy and loss behavior
- `test_generation.py` - Autoregressive generation and sampling

## Running Tests

Run individual test files:
```bash
uv run python conceptual_tests/test_tokenizer.py
uv run python conceptual_tests/test_embedding.py
# ... etc
```

Or run all at once:
```bash
for test in conceptual_tests/test_*.py; do
    echo "Running $test..."
    uv run python "$test"
done
```

## Writing Your Own Tests

Each test should:
1. **Have a clear hypothesis** - What are you testing?
2. **Use minimal code** - Focus on the concept, not engineering
3. **Print explanations** - Help future-you understand what you learned
4. **Include assertions** - Make it fail if understanding is wrong

Example template:
```python
"""
MVT-Component-NN: Test Name
Goal: What conceptual understanding are we verifying?
"""

def test_concept():
    # Setup: Create minimal example
    ...

    # Action: Perform the key operation
    ...

    # Verify: Check expected behavior
    assert ...

    # Explain: Print what this proves
    print("✓ Concept verified: ...")

if __name__ == "__main__":
    test_concept()
```

## Guidelines

**DO**:
- Focus on conceptual insights
- Test edge cases that reveal understanding
- Compare expected vs actual behavior
- Visualize results when helpful (save plots to `conceptual_tests/plots/`)

**DON'T**:
- Duplicate unit tests
- Test implementation details
- Write production-quality code (hacky is fine!)
- Worry about performance

## Example: What Makes a Good Conceptual Test?

**Bad** (just duplicates unit test):
```python
def test_attention():
    # Just checks attention output shape
    assert attention(Q, K, V).shape == expected_shape
```

**Good** (reveals understanding):
```python
def test_attention_is_value_averaging():
    # Create uniform Q and K (all positions equal attention)
    Q = K = torch.ones(...)
    V = torch.randn(...)

    output = attention(Q, K, V)

    # Key insight: Output should be mean of V
    assert torch.allclose(output, V.mean(dim=1, keepdim=True).expand(...))

    print("✓ Verified: Attention output is always a weighted average of V")
    print("  → This means attention can only *mix* information, not create new info")
```

## Resources

See [PROJECT_PLAN.md](../PROJECT_PLAN.md) for:
- Detailed MVT specifications for each component
- Standard answers to conceptual questions
- Expected understanding after each test

---

**Remember**: The goal is not to pass tests, but to build mental models that let you design and debug Transformers with confidence.
