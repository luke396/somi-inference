# Benchmarks

This directory contains the first offline benchmark suite for `somi-inference`.

The structure follows the same high-level split used by `vLLM` and `SGLang`:

- `bench_prefill.py`: prefill latency / input tok/s
- `bench_decode.py`: fixed-context single-step decode latency / output tok/s
- `bench_paged_attention.py`: raw `paged_attention_decode()` microbenchmark
- `bench_engine.py`: continuous batching engine throughput with synthetic arrivals
- `bench_e2e.py`: single-request `LLM.generate()` TTFT / end-to-end tok/s

## Usage

Run the model-backed benchmarks:

```bash
uv run python -m benchmarks.bench_prefill \
  --model-name Qwen/Qwen2.5-0.5B \
  --prompt-lens 32 128
uv run python -m benchmarks.bench_decode \
  --model-name Qwen/Qwen2.5-0.5B \
  --batch-sizes 1 4 --context-lens 128 512
uv run python -m benchmarks.bench_e2e \
  --model-name Qwen/Qwen2.5-0.5B \
  --prompt "Explain paged attention simply." --max-new-tokens 32
uv run python -m benchmarks.bench_engine \
  --model-name Qwen/Qwen2.5-0.5B \
  --num-prompts 32 --prompt-len 128 --output-len 32
```

Run the raw kernel microbenchmark:

The `triton` backend is optional and CUDA-only. Install it with
`uv sync --extra triton` and run `--backend triton` only on a CUDA machine.

```bash
uv run python -m benchmarks.bench_paged_attention \
  --batch-sizes 1 4 --seq-lens 512 2048
uv run python -m benchmarks.bench_paged_attention \
  --backend torch_ref --batch-sizes 1 4 --seq-lens 128 512
uv run python -m benchmarks.bench_paged_attention \
  --backend triton --batch-sizes 1 4 --seq-lens 128 512
```

Append results to JSONL:

```bash
uv run python -m benchmarks.bench_decode \
  --model-name Qwen/Qwen2.5-0.5B \
  --output-file benchmark_results.jsonl
```

Run the convenience script with local or server presets:

```bash
scripts/run_cuda_benchmarks.sh
MODE=server scripts/run_cuda_benchmarks.sh
```

## Notes

- `--model-name` uses the existing HF loader and then moves the model to the chosen `--device` / `--dtype`.
- `bench_decode.py` rebuilds KV state before each timed iteration, so it measures one decode step at a fixed context length and excludes prefill time.
- `bench_e2e.py` exercises the public `LLM.generate()` path, including tokenization, scheduling, sampling, and decode.
- JSONL payloads now include an `environment` object with `git_sha`, `git_dirty`, Python / PyTorch versions, and the resolved device name.
- `bench_engine.py` benchmarks the current in-process scheduler / engine path, not an HTTP serving stack.
- `scripts/run_cuda_benchmarks.sh` defaults to a lighter `MODE=local` preset for small GPUs and uses `MODE=server` for the fuller sweep.
