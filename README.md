# somi-inference

Minimal LLM inference engine — from PyTorch to CUDA/Triton.

A learning-oriented project that implements core LLM inference components from scratch, following vLLM/SGLang design patterns.

## Features

- **Paged Attention** — online softmax, GQA support
- **Continuous Batching** — scheduler + batching engine
- **Text-in / Text-out API** — `LLM.generate()` over tokenizer + runner + scheduler
- **Qwen2.5 Model** — hand-written forward pass (RMSNorm, RoPE, MHA, SwiGLU MLP)
- **HF Weight Loading** — load pretrained weights from Hugging Face (0.5B / 1.5B)
- **End-to-end Greedy Decode** — validated against HF reference output

## Roadmap

- [x] **Phase 1**: PyTorch baseline — paged attention, Qwen2.5 model, e2e greedy decode
- [x] **Phase 2**: End-to-end inference pipeline — ModelRunner, Tokenizer, text-in/text-out API
- [ ] **Phase 3**: Triton/CUDA optimization — flash attention, fused kernels, quantization
- [ ] **Phase 4**: Serving — HTTP API, concurrent requests, streaming

## Installation

```bash
uv sync
uv sync --extra triton  # optional CUDA-only Triton backend
```

## Pre-commit

This repo uses `pre-commit` to run `uv run --no-sync ruff check`,
`uv run --no-sync ty check`, and the committed fast test suite
(`uv run --no-sync pytest -m "not slow"`) before every commit.

If dependencies or the lockfile changed, run `uv sync` before relying on the
`--no-sync` commands.

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
uv run pytest tests/entrypoints/test_llm_e2e.py -m slow  # LLM slow e2e tests
```

## Benchmarking

The `triton` backend is optional and CUDA-only. Install it with
`uv sync --extra triton` and run `--backend triton` benchmarks on a CUDA
machine.

```bash
uv run python -m benchmarks.bench_prefill --model-name Qwen/Qwen2.5-0.5B --prompt-lens 32 128 512
uv run python -m benchmarks.bench_decode --model-name Qwen/Qwen2.5-0.5B --batch-sizes 1 4 --context-lens 128 512
uv run python -m benchmarks.bench_paged_attention --batch-sizes 1 4 --seq-lens 512 2048
uv run python -m benchmarks.bench_paged_attention --backend torch_ref --batch-sizes 1 4 --seq-lens 128 512
uv run python -m benchmarks.bench_paged_attention --backend triton --batch-sizes 1 4 --seq-lens 128 512
uv run python -m benchmarks.bench_engine --model-name Qwen/Qwen2.5-0.5B --num-prompts 32 --prompt-len 128 --output-len 32
scripts/run_cuda_benchmarks.sh             # lighter local preset
MODE=server scripts/run_cuda_benchmarks.sh # fuller server preset
```

See `benchmarks/README.md` for more examples and JSONL output support.

Recent local raw paged-attention results from `bench_paged_attention.py`,
collected on an `NVIDIA GeForce GTX 1650 Ti` with:

```bash
uv run python -m benchmarks.bench_paged_attention --backend torch_ref --batch-sizes 1 4 --seq-lens 128 512 --num-kv-heads 2 --head-dim 64 --warmup-iters 3 --measure-iters 20
uv run python -m benchmarks.bench_paged_attention --backend triton --batch-sizes 1 4 --seq-lens 128 512 --num-kv-heads 2 --head-dim 64 --warmup-iters 3 --measure-iters 20
```

Both runs used `float16`, `block_size=16`, and `num_q_heads=8`.

| Batch | Seq Len | `torch_ref` mean | `triton` mean | Speedup |
| --- | --- | ---: | ---: | ---: |
| 1 | 128 | 30.620 ms | 0.676 ms | 45.27x |
| 1 | 512 | 82.625 ms | 0.475 ms | 173.89x |
| 4 | 128 | 22.107 ms | 0.529 ms | 41.77x |
| 4 | 512 | 81.450 ms | 0.282 ms | 288.93x |

These are raw decode-kernel microbenchmarks, not end-to-end model decode
numbers, but they do confirm that the Triton paged-attention path removes the
old Python-side gather / materialization hot path.

## Project Structure

```
somi_inference/
├── core/
│   ├── paged_attention.py     # Paged attention with online softmax, GQA
│   ├── continuous_batching.py # Scheduler + batching engine
│   ├── model_runner.py        # Adapter + sampler execution layer
│   └── sampler.py             # Greedy / temperature / top-k / top-p / repetition penalty
├── benchmarks/
│   ├── bench_prefill.py       # Prefill latency and tok/s
│   ├── bench_decode.py        # Decode latency and tok/s
│   ├── bench_paged_attention.py # Raw paged attention microbenchmark
│   └── bench_engine.py        # Continuous batching throughput benchmark
├── entrypoints/
│   └── llm.py                 # High-level text-in / text-out API
└── models/
    ├── base.py                # ModelAdapter protocol, ForwardContext
    ├── loader.py              # Model-family dispatch from HF config
    ├── qwen2.py               # RMSNorm, RotaryEmbedding, Attention, MLP, DecoderLayer, Model
    └── qwen2_adapter.py       # QwenAdapter (prefill/decode) + HF weight loading
├── tokenizer.py               # HF tokenizer wrapper
```
