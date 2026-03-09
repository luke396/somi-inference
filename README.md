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

## Current Status

Phase 1.1 完成 - 核心推理算法已迁移：
- ✅ BlockAllocator: 物理块分配器（引用计数 + COW）
- ✅ KVCache: KV tensor 存储
- ✅ KVCacheManager: 协调分配器和缓存
- ✅ paged_attention_decode: 在线 softmax 的分页注意力
- ✅ Scheduler: FCFS 调度器
- ✅ ContinuousBatchingEngine: 推理引擎
- ✅ ModelAdapter: 抽象接口（Phase 1.2 实现 QwenAdapter）

## Next Steps (Phase 1.2)

1. 实现 QwenAdapter 对接 HuggingFace Qwen2.5-1.5B
2. 修改 KVCache 和 paged_attention_decode 支持 GQA
3. 实现端到端推理循环
4. 验证输出正确性
