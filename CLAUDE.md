# CLAUDE.md

## Project Overview

Stanford CS336 (Spring 2025) Assignment 1: Basics. Implement core Transformer components from scratch in PyTorch — including BPE tokenizer, model layers, optimizer, data loading, and training utilities.

## Repository Structure

- `cs336_basics/` — Main package (implementation code goes here)
- `tests/` — Unit tests; `tests/adapters.py` bridges implementations to tests
- `conceptual_tests/` — Conceptual understanding tests
- `docs/` — Documentation
- `data/` — Training data (not committed)

## Build & Run

- **Package manager**: `uv` (not pip)
- **Python**: >=3.11
- **Run any script**: `uv run <file>`
- **Run all tests**: `uv run pytest`
- **Run a specific test**: `uv run pytest tests/test_model.py::test_linear -v`
- **Run conceptual tests**: `uv run python conceptual_tests/test_attention.py`
- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`

## Code Style & Conventions

- Line length: 120 characters (ruff)
- Type hints use `jaxtyping` for tensor shapes (e.g., `Float[Tensor, "batch seq d_model"]`)
- Tensor operations use `einops` and `einx`
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
