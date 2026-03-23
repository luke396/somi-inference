# CLAUDE.md

Repository-specific guidance for coding agents working in this repo. Keep this file semantically aligned with `AGENTS.md`; if a rule is not tool-specific, update both files in the same change.

A minimal, self-contained LLM inference framework focused on implementing the smallest complete set of features needed for real inference, from PyTorch to CUDA/Triton.

## Environment

- Use `uv` for dependency and environment management
- Run tests: `uv run pytest`
- Lint: `uv run ruff check`, type check: `uv run ty check`

## Architecture

- Design follows vLLM/SGLang patterns (ForwardContext, unified forward signature)
- Model adapter pattern: `ModelAdapter` protocol in `models/base.py`, concrete adapters (e.g. `QwenAdapter`) handle prefill/decode
- Core components: paged attention, KV cache management, continuous batching

## Documentation

- AI-generated design docs and implementation plans are NOT committed to the repo; keep them local (gitignored in `docs/plans/`) or in PR descriptions
- Long-lived architectural decisions should be written as ADRs in `docs/adr/`
