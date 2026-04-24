#!/usr/bin/env bash
set -euo pipefail

# Local-only resume benchmark artifacts default to .local/resume/runs/<run-id>.
RUN_ID="${RUN_ID:-$(date +%F-%H%M%S)}"
OUT_DIR="${OUT_DIR:-.local/resume/runs/${RUN_ID}}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${OUT_DIR}/.uv-cache}"
README_PATH="${README_PATH:-README.md}"
CHARTS_DIR="${CHARTS_DIR:-docs/images/benchmarks}"
LOCAL_NOTES_PATH="${LOCAL_NOTES_PATH:-.local/resume/resume-bullets.md}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B}"
DEVICE="${DEVICE:-cuda}"
DTYPE="float16"
BASE_PROMPT="${BASE_PROMPT:-人工智能的发展历程可以追溯到上世纪五十年代。}"

PREFILL_PROMPT_LENS="${PREFILL_PROMPT_LENS:-128 512}"
PAGED_ATTENTION_BATCH_SIZES="${PAGED_ATTENTION_BATCH_SIZES:-1 4}"
PAGED_ATTENTION_SEQ_LENS="${PAGED_ATTENTION_SEQ_LENS:-128 512}"

PREFILL_WARMUP_ITERS="${PREFILL_WARMUP_ITERS:-1}"
PREFILL_MEASURE_ITERS="${PREFILL_MEASURE_ITERS:-5}"
PAGED_ATTENTION_WARMUP_ITERS="${PAGED_ATTENTION_WARMUP_ITERS:-1}"
PAGED_ATTENTION_MEASURE_ITERS="${PAGED_ATTENTION_MEASURE_ITERS:-5}"
E2E_WARMUP_ITERS="${E2E_WARMUP_ITERS:-1}"
E2E_MEASURE_ITERS="${E2E_MEASURE_ITERS:-3}"
ENGINE_WARMUP_REQUESTS="${ENGINE_WARMUP_REQUESTS:-1}"

E2E_WORKLOAD="${E2E_WORKLOAD:-agent-session}"
E2E_PRESET="${E2E_PRESET:-mid}"
E2E_OUTPUT_TOKENS="${E2E_OUTPUT_TOKENS:-1}"
E2E_NUM_BLOCKS="${E2E_NUM_BLOCKS:-96}"
E2E_MAX_CONCURRENT="${E2E_MAX_CONCURRENT:-2}"

ENGINE_WORKLOAD="${ENGINE_WORKLOAD:-agent-session}"
ENGINE_PRESET="${ENGINE_PRESET:-mid}"
ENGINE_OUTPUT_TOKENS="${ENGINE_OUTPUT_TOKENS:-1}"
ENGINE_MAX_CONCURRENT="${ENGINE_MAX_CONCURRENT:-2}"
ENGINE_ARRIVAL_PATTERN="${ENGINE_ARRIVAL_PATTERN:-burst}"
ENGINE_ARRIVAL_RATE="${ENGINE_ARRIVAL_RATE:-1.0}"

mkdir -p "$OUT_DIR" "$UV_CACHE_DIR" "$CHARTS_DIR" "$(dirname "$LOCAL_NOTES_PATH")"

read -r -a prefill_prompt_lens <<<"$PREFILL_PROMPT_LENS"
read -r -a paged_attention_batch_sizes <<<"$PAGED_ATTENTION_BATCH_SIZES"
read -r -a paged_attention_seq_lens <<<"$PAGED_ATTENTION_SEQ_LENS"

run_python() {
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync python "$@"
}

run_jsonl() {
  local stem="$1"
  shift
  local output_file="${OUT_DIR}/${stem}.jsonl"
  : > "$output_file"
  run_python "$@" --output-file "$output_file"
}

printf 'resume benchmark run -> %s\n' "$OUT_DIR"
printf 'model=%s device=%s dtype=%s\n' "$MODEL_NAME" "$DEVICE" "$DTYPE"
printf 'primary workload=%s/%s output_tokens=%s\n' \
  "$E2E_WORKLOAD" "$E2E_PRESET" "$E2E_OUTPUT_TOKENS"

run_jsonl prefill__baseline \
  -m benchmarks.bench_prefill \
  --model-name "$MODEL_NAME" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attention-backend torch_ref \
  --mlp-backend torch_ref \
  --warmup-iters "$PREFILL_WARMUP_ITERS" \
  --measure-iters "$PREFILL_MEASURE_ITERS" \
  --prompt-lens "${prefill_prompt_lens[@]}"

run_jsonl prefill__prefill_bundle \
  -m benchmarks.bench_prefill \
  --model-name "$MODEL_NAME" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attention-backend triton \
  --mlp-backend triton \
  --warmup-iters "$PREFILL_WARMUP_ITERS" \
  --measure-iters "$PREFILL_MEASURE_ITERS" \
  --prompt-lens "${prefill_prompt_lens[@]}"

run_jsonl paged_attention__baseline \
  -m benchmarks.bench_paged_attention \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --backend torch_ref \
  --warmup-iters "$PAGED_ATTENTION_WARMUP_ITERS" \
  --measure-iters "$PAGED_ATTENTION_MEASURE_ITERS" \
  --batch-sizes "${paged_attention_batch_sizes[@]}" \
  --seq-lens "${paged_attention_seq_lens[@]}"

run_jsonl paged_attention__decode_bundle \
  -m benchmarks.bench_paged_attention \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --backend triton \
  --warmup-iters "$PAGED_ATTENTION_WARMUP_ITERS" \
  --measure-iters "$PAGED_ATTENTION_MEASURE_ITERS" \
  --batch-sizes "${paged_attention_batch_sizes[@]}" \
  --seq-lens "${paged_attention_seq_lens[@]}"

run_jsonl e2e__baseline \
  -m benchmarks.bench_e2e \
  --model-name "$MODEL_NAME" \
  --num-blocks "$E2E_NUM_BLOCKS" \
  --max-concurrent "$E2E_MAX_CONCURRENT" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attention-backend torch_ref \
  --decode-attention-backend torch_ref \
  --mlp-backend torch_ref \
  --warmup-iters "$E2E_WARMUP_ITERS" \
  --measure-iters "$E2E_MEASURE_ITERS" \
  --workload "$E2E_WORKLOAD" \
  --preset "$E2E_PRESET" \
  --output-tokens "$E2E_OUTPUT_TOKENS" \
  --base-prompt "$BASE_PROMPT"

run_jsonl e2e__prefill_bundle \
  -m benchmarks.bench_e2e \
  --model-name "$MODEL_NAME" \
  --num-blocks "$E2E_NUM_BLOCKS" \
  --max-concurrent "$E2E_MAX_CONCURRENT" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attention-backend triton \
  --decode-attention-backend torch_ref \
  --mlp-backend triton \
  --warmup-iters "$E2E_WARMUP_ITERS" \
  --measure-iters "$E2E_MEASURE_ITERS" \
  --workload "$E2E_WORKLOAD" \
  --preset "$E2E_PRESET" \
  --output-tokens "$E2E_OUTPUT_TOKENS" \
  --base-prompt "$BASE_PROMPT"

run_jsonl e2e__decode_bundle \
  -m benchmarks.bench_e2e \
  --model-name "$MODEL_NAME" \
  --num-blocks "$E2E_NUM_BLOCKS" \
  --max-concurrent "$E2E_MAX_CONCURRENT" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attention-backend torch_ref \
  --decode-attention-backend triton \
  --mlp-backend torch_ref \
  --warmup-iters "$E2E_WARMUP_ITERS" \
  --measure-iters "$E2E_MEASURE_ITERS" \
  --workload "$E2E_WORKLOAD" \
  --preset "$E2E_PRESET" \
  --output-tokens "$E2E_OUTPUT_TOKENS" \
  --base-prompt "$BASE_PROMPT"

run_jsonl e2e__full_triton \
  -m benchmarks.bench_e2e \
  --model-name "$MODEL_NAME" \
  --num-blocks "$E2E_NUM_BLOCKS" \
  --max-concurrent "$E2E_MAX_CONCURRENT" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attention-backend triton \
  --decode-attention-backend triton \
  --mlp-backend triton \
  --warmup-iters "$E2E_WARMUP_ITERS" \
  --measure-iters "$E2E_MEASURE_ITERS" \
  --workload "$E2E_WORKLOAD" \
  --preset "$E2E_PRESET" \
  --output-tokens "$E2E_OUTPUT_TOKENS" \
  --base-prompt "$BASE_PROMPT"

run_jsonl engine__baseline \
  -m benchmarks.bench_engine \
  --model-name "$MODEL_NAME" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attention-backend torch_ref \
  --decode-attention-backend torch_ref \
  --mlp-backend torch_ref \
  --workload "$ENGINE_WORKLOAD" \
  --preset "$ENGINE_PRESET" \
  --output-tokens "$ENGINE_OUTPUT_TOKENS" \
  --arrival-pattern "$ENGINE_ARRIVAL_PATTERN" \
  --arrival-rate "$ENGINE_ARRIVAL_RATE" \
  --base-prompt "$BASE_PROMPT" \
  --max-concurrent "$ENGINE_MAX_CONCURRENT" \
  --warmup-requests "$ENGINE_WARMUP_REQUESTS"

run_jsonl engine__prefill_bundle \
  -m benchmarks.bench_engine \
  --model-name "$MODEL_NAME" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attention-backend triton \
  --decode-attention-backend torch_ref \
  --mlp-backend triton \
  --workload "$ENGINE_WORKLOAD" \
  --preset "$ENGINE_PRESET" \
  --output-tokens "$ENGINE_OUTPUT_TOKENS" \
  --arrival-pattern "$ENGINE_ARRIVAL_PATTERN" \
  --arrival-rate "$ENGINE_ARRIVAL_RATE" \
  --base-prompt "$BASE_PROMPT" \
  --max-concurrent "$ENGINE_MAX_CONCURRENT" \
  --warmup-requests "$ENGINE_WARMUP_REQUESTS"

run_jsonl engine__decode_bundle \
  -m benchmarks.bench_engine \
  --model-name "$MODEL_NAME" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attention-backend torch_ref \
  --decode-attention-backend triton \
  --mlp-backend torch_ref \
  --workload "$ENGINE_WORKLOAD" \
  --preset "$ENGINE_PRESET" \
  --output-tokens "$ENGINE_OUTPUT_TOKENS" \
  --arrival-pattern "$ENGINE_ARRIVAL_PATTERN" \
  --arrival-rate "$ENGINE_ARRIVAL_RATE" \
  --base-prompt "$BASE_PROMPT" \
  --max-concurrent "$ENGINE_MAX_CONCURRENT" \
  --warmup-requests "$ENGINE_WARMUP_REQUESTS"

run_jsonl engine__full_triton \
  -m benchmarks.bench_engine \
  --model-name "$MODEL_NAME" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attention-backend triton \
  --decode-attention-backend triton \
  --mlp-backend triton \
  --workload "$ENGINE_WORKLOAD" \
  --preset "$ENGINE_PRESET" \
  --output-tokens "$ENGINE_OUTPUT_TOKENS" \
  --arrival-pattern "$ENGINE_ARRIVAL_PATTERN" \
  --arrival-rate "$ENGINE_ARRIVAL_RATE" \
  --base-prompt "$BASE_PROMPT" \
  --max-concurrent "$ENGINE_MAX_CONCURRENT" \
  --warmup-requests "$ENGINE_WARMUP_REQUESTS"

run_python scripts/render_resume_benchmark_report.py \
  --results-dir "$OUT_DIR" \
  --charts-dir "$CHARTS_DIR" \
  --readme "$README_PATH" \
  --local-notes "$LOCAL_NOTES_PATH"

printf 'resume benchmark results saved to %s\n' "$OUT_DIR"
printf 'readme refreshed: %s\n' "$README_PATH"
