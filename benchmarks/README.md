# Benchmarks

This directory contains the first offline benchmark suite for `somi-inference`.

The structure follows the same high-level split used by `vLLM` and `SGLang`:

- `bench_prefill.py`: prefill latency / input tok/s
- `bench_decode.py`: fixed-context single-step decode latency / output tok/s
- `bench_paged_attention.py`: raw `paged_attention_decode()` microbenchmark
- `bench_engine.py`: continuous batching engine throughput with workload arrivals
- `bench_e2e.py`: deterministic workload `LLM.generate()` TTFT / end-to-end tok/s
- `profile_prefill.py`: semantic prefill stage profiler (`project_qkv` / attention / `write_kv` / `mlp` / `lm_head`)

Current split:

- fixed-shape:
  - `bench_paged_attention.py`
  - `bench_prefill.py`
  - `bench_decode.py`
  - `profile_prefill.py`
- workload:
  - `bench_e2e.py`
  - `bench_engine.py`

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
  --device cuda --dtype float16 \
  --attention-backend torch_ref \
  --decode-attention-backend torch_ref \
  --mlp-backend torch_ref \
  --workload agent-session --preset mid
uv run python -m benchmarks.bench_e2e \
  --model-name Qwen/Qwen2.5-0.5B \
  --device cuda --dtype float16 \
  --attention-backend triton \
  --decode-attention-backend triton \
  --mlp-backend triton \
  --workload chat-serving --preset long \
  --base-prompt "Explain paged attention simply."
uv run python -m benchmarks.bench_engine \
  --model-name Qwen/Qwen2.5-0.5B \
  --workload agent-session --preset mid \
  --arrival-pattern burst --max-concurrent 4
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

Run the semantic prefill profiler before tuning kernels or cache writes:

```bash
uv run python -m benchmarks.profile_prefill \
  --model-name Qwen/Qwen2.5-0.5B \
  --device cuda \
  --dtype float16 \
  --attention-backends torch_ref triton \
  --synthetic-prompt-len 512
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
scripts/run_resume_benchmarks.sh
```

The convenience script fixes `dtype=float16` and runs a backend comparison sweep
for both `torch_ref` and `triton`, appending both variants to the same JSONL so
the results can be compared directly.

`scripts/run_resume_benchmarks.sh` is the focused evaluation path used by the
top-level `README.md`: it runs a small, resume-friendly comparison matrix,
writes raw JSONL under `.local/resume/`, refreshes the committed charts in
`docs/images/benchmarks/`, and updates the generated evaluation block in
`README.md`.

## Notes

- `--model-name` uses the existing HF loader and then moves the model to the chosen `--device` / `--dtype`.
- The benchmark suite is intentionally split into `fixed-shape` and `workload` layers: fixed-shape scripts answer kernel / operator questions, while workload scripts answer user-facing interaction and scheduler questions.
- `bench_decode.py` rebuilds KV state before each timed iteration, so it measures one decode step at a fixed context length and excludes prefill time.
- `bench_e2e.py` is workload-only now: it benchmarks deterministic multi-turn `agent-session` and `chat-serving` presets through the public `LLM.generate()` path.
- `bench_e2e.py` requires explicit `--device`, `--dtype`, `--attention-backend`, `--decode-attention-backend`, and `--mlp-backend`; `cuda` is the device, while `torch_ref` / `triton` are kernel backend choices.
- `bench_e2e.py` appends one JSONL row per turn with `mode`, `scenario`, `session_id`, `turn_idx`, `requested_prompt_tokens`, `actual_prompt_tokens`, and `requested_output_tokens`.
- `bench_e2e.py` and `bench_engine.py` accept `--output-tokens` so targeted traces such as one-token TTFT comparisons can reuse the same workload definitions without editing code.
- `agent-session` runs `1` and `32` token decode variants, while `chat-serving` runs `64`, `128`, and `256` token decode variants.
- `--base-prompt` controls the seed text used to synthesize the base system prompt for both workload families.
- `bench_engine.py` now supports the same `--workload` / `--preset` families as `bench_e2e.py`; it tokenizes those prompts before timing, then benchmarks only the in-process scheduler / engine path.
- Triton dtype support is currently split by kernel: prefill attention supports `float16` / `bfloat16`, decode paged attention supports `float32` / `float16` / `bfloat16`, and Triton MLP supports `float16` only.
- `profile_prefill.py` is for attribution, not acceptance: use it to find the current bottleneck before optimization, then use `bench_prefill.py` / `bench_e2e.py` to verify gains.
- JSONL payloads now include an `environment` object with `git_sha`, `git_dirty`, Python / PyTorch versions, and the resolved device name.
- `bench_engine.py` benchmarks the current in-process scheduler / engine path, not an HTTP serving stack; it preserves the same prompt growth and output-length semantics as `bench_e2e.py`, but tokenizes prompts before timing so scheduler / engine measurements stay isolated.
- `scripts/run_cuda_benchmarks.sh` defaults to a lighter `MODE=local` preset for small GPUs and uses `MODE=server` for the fuller sweep.
- `scripts/run_cuda_benchmarks.sh` now runs the same sweep twice—once with `torch_ref` and once with `triton`—while keeping `dtype=float16` fixed for apples-to-apples comparisons.
