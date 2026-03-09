# somi-inference

Minimal LLM inference engine - from PyTorch to CUDA/Triton

## Roadmap

### Phase 1: PyTorch Baseline
- [x] Phase 1.1: Migrate paged attention + continuous batching
- [ ] Phase 1.2: Qwen2.5-1.5B support with GQA

### Phase 2: CUDA/Triton Optimization
- [ ] FlashAttention kernel
- [ ] Qwen3.5 hybrid attention + mrope

## Installation

```bash
pip install -e ".[dev]"
```

## Testing

```bash
pytest tests/
```

## Project Structure

```
somi_inference/
├── core/              # Core inference algorithms
│   ├── paged_attention.py
│   └── continuous_batching.py
└── models/            # Model adapters
    └── base.py
```
