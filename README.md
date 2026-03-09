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
uv sync
```

## Testing

```bash
uv run pytest tests/
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

## Current Status

Phase 1.1 complete - core inference algorithms migrated:
- BlockAllocator: Physical block allocator (ref counting + COW)
- KVCache: KV tensor storage
- KVCacheManager: Coordinates allocator and cache
- paged_attention_decode: Paged attention with online softmax
- Scheduler: FCFS scheduler
- ContinuousBatchingEngine: Inference engine
- ModelAdapter: Abstract interface (QwenAdapter in Phase 1.2)

## Next Steps (Phase 1.2)

1. Implement QwenAdapter for HuggingFace Qwen2.5-1.5B
2. Modify KVCache and paged_attention_decode for GQA support
3. Implement end-to-end inference loop
4. Verify output correctness
