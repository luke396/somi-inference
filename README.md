# somi-inference

Minimal LLM inference engine — from PyTorch to CUDA/Triton

## Current Status

Phase 1: PyTorch Baseline for Qwen2.5 (branch: `phase-1`)
- [x] Phase 1.1: Paged attention + continuous batching
- [x] Phase 1.2: Qwen2.5 support (GQA, ForwardContext, HF weight loading, e2e greedy decode)
- Currently supports Qwen2.5-0.5B and 1.5B

## Roadmap

### Phase 2: CUDA/Triton Optimization
- [ ] FlashAttention kernel
- [ ] Qwen3.5 hybrid attention + mrope

## Installation

```bash
uv sync
```

## Testing

```bash
uv run pytest tests/
```

## Project Structure

```
somi_inference/
├── core/
│   ├── paged_attention.py     # Paged attention with online softmax, GQA
│   └── continuous_batching.py # Scheduler + batching engine
└── models/
    ├── base.py                # ModelAdapter protocol
    ├── qwen2.py               # RMSNorm, RotaryEmbedding, Attention, MLP, DecoderLayer, Model
    └── qwen2_adapter.py       # QwenAdapter (prefill/decode) + HF weight loading
```
