# somi-inference

Minimal LLM inference engine — from PyTorch to CUDA/Triton.

A learning-oriented project that implements core LLM inference components from scratch, following vLLM/SGLang design patterns.

## Features

- **Paged Attention** — online softmax, GQA support
- **Continuous Batching** — scheduler + batching engine
- **Qwen2.5 Model** — hand-written forward pass (RMSNorm, RoPE, MHA, SwiGLU MLP)
- **HF Weight Loading** — load pretrained weights from Hugging Face (0.5B / 1.5B)
- **End-to-end Greedy Decode** — validated against HF reference output

## Roadmap

- [x] **Phase 1**: PyTorch baseline — paged attention, Qwen2.5 model, e2e greedy decode
- [ ] **Phase 2**: End-to-end inference pipeline — ModelRunner, Tokenizer, text-in/text-out API
- [ ] **Phase 3**: Triton/CUDA optimization — flash attention, fused kernels, quantization
- [ ] **Phase 4**: Serving — HTTP API, concurrent requests, streaming

## Installation

```bash
uv sync
```

## Pre-commit

This repo uses `pre-commit` to run `uv run ruff check`, `uv run ty check`,
and the committed fast test suite (`uv run pytest -m "not slow"`) before
every commit.

Enable it in a local clone with:

```bash
uv sync
uv run pre-commit install
```

## Testing

```bash
uv run pytest                      # default committed suite
uv run pytest -m integration       # integration tests (requires GPU)
uv run pytest -m "not slow"        # fast local / pre-commit suite
uv run pytest tests_tdd/phase2     # Phase 2 TDD / not-yet-green tests
```

`tests_tdd/` is intentionally excluded from the default `ruff`, `ty`, and
`pytest` checks used by `pre-commit`.

## Project Structure

```
somi_inference/
├── core/
│   ├── paged_attention.py     # Paged attention with online softmax, GQA
│   └── continuous_batching.py # Scheduler + batching engine
└── models/
    ├── base.py                # ModelAdapter protocol, ForwardContext
    ├── qwen2.py               # RMSNorm, RotaryEmbedding, Attention, MLP, DecoderLayer, Model
    └── qwen2_adapter.py       # QwenAdapter (prefill/decode) + HF weight loading
```
