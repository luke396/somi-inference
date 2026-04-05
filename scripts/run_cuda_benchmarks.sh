#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-local}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-auto}"
PROMPT="${PROMPT:-人工智能的发展历程可以追溯到上世纪五十年代。}"
RESULT_JSONL="${RESULT_JSONL:-benchmarks/results/$(date +%F)-${MODE}-cuda-benchmarks.jsonl}"

case "$MODE" in
local)
  WARMUP_ITERS="${WARMUP_ITERS:-1}"
  MEASURE_ITERS="${MEASURE_ITERS:-5}"
  PAGED_ATTENTION_BATCH_SIZES="${PAGED_ATTENTION_BATCH_SIZES:-1 4}"
  PAGED_ATTENTION_SEQ_LENS="${PAGED_ATTENTION_SEQ_LENS:-128 512 2048}"
  PREFILL_PROMPT_LENS="${PREFILL_PROMPT_LENS:-32 128 512}"
  DECODE_BATCH_SIZES="${DECODE_BATCH_SIZES:-1 2}"
  DECODE_CONTEXT_LENS="${DECODE_CONTEXT_LENS:-128 512}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
  E2E_NUM_BLOCKS="${E2E_NUM_BLOCKS:-64}"
  E2E_MAX_CONCURRENT="${E2E_MAX_CONCURRENT:-4}"
  ENGINE_NUM_PROMPTS="${ENGINE_NUM_PROMPTS:-16}"
  ENGINE_PROMPT_LEN="${ENGINE_PROMPT_LEN:-128}"
  ENGINE_OUTPUT_LEN="${ENGINE_OUTPUT_LEN:-16}"
  ENGINE_MAX_CONCURRENT="${ENGINE_MAX_CONCURRENT:-4}"
  ENGINE_WARMUP_REQUESTS="${ENGINE_WARMUP_REQUESTS:-1}"
  ;;
server)
  WARMUP_ITERS="${WARMUP_ITERS:-2}"
  MEASURE_ITERS="${MEASURE_ITERS:-10}"
  PAGED_ATTENTION_BATCH_SIZES="${PAGED_ATTENTION_BATCH_SIZES:-1 2 4 8}"
  PAGED_ATTENTION_SEQ_LENS="${PAGED_ATTENTION_SEQ_LENS:-128 512 2048 4096}"
  PREFILL_PROMPT_LENS="${PREFILL_PROMPT_LENS:-32 128 512 2048}"
  DECODE_BATCH_SIZES="${DECODE_BATCH_SIZES:-1 2 4 8}"
  DECODE_CONTEXT_LENS="${DECODE_CONTEXT_LENS:-128 512 2048}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
  E2E_NUM_BLOCKS="${E2E_NUM_BLOCKS:-256}"
  E2E_MAX_CONCURRENT="${E2E_MAX_CONCURRENT:-16}"
  ENGINE_NUM_PROMPTS="${ENGINE_NUM_PROMPTS:-128}"
  ENGINE_PROMPT_LEN="${ENGINE_PROMPT_LEN:-256}"
  ENGINE_OUTPUT_LEN="${ENGINE_OUTPUT_LEN:-64}"
  ENGINE_MAX_CONCURRENT="${ENGINE_MAX_CONCURRENT:-16}"
  ENGINE_WARMUP_REQUESTS="${ENGINE_WARMUP_REQUESTS:-1}"
  ;;
*)
  printf 'unsupported MODE=%s (expected local or server)\n' "$MODE" >&2
  exit 1
  ;;
esac

read -r -a paged_attention_batch_sizes <<<"$PAGED_ATTENTION_BATCH_SIZES"
read -r -a paged_attention_seq_lens <<<"$PAGED_ATTENTION_SEQ_LENS"
read -r -a prefill_prompt_lens <<<"$PREFILL_PROMPT_LENS"
read -r -a decode_batch_sizes <<<"$DECODE_BATCH_SIZES"
read -r -a decode_context_lens <<<"$DECODE_CONTEXT_LENS"

mkdir -p "$(dirname "$RESULT_JSONL")"
: > "$RESULT_JSONL"

run() {
  uv run --no-sync python "$@"
}

printf 'running CUDA benchmarks in %s mode\n' "$MODE"
printf 'results -> %s\n' "$RESULT_JSONL"

run -m benchmarks.bench_paged_attention \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --warmup-iters "$WARMUP_ITERS" \
  --measure-iters "$MEASURE_ITERS" \
  --batch-sizes "${paged_attention_batch_sizes[@]}" \
  --seq-lens "${paged_attention_seq_lens[@]}" \
  --output-file "$RESULT_JSONL"

run -m benchmarks.bench_prefill \
  --model-name "$MODEL_NAME" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --warmup-iters "$WARMUP_ITERS" \
  --measure-iters "$MEASURE_ITERS" \
  --prompt-lens "${prefill_prompt_lens[@]}" \
  --output-file "$RESULT_JSONL"

run -m benchmarks.bench_decode \
  --model-name "$MODEL_NAME" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --warmup-iters "$WARMUP_ITERS" \
  --measure-iters "$MEASURE_ITERS" \
  --batch-sizes "${decode_batch_sizes[@]}" \
  --context-lens "${decode_context_lens[@]}" \
  --output-file "$RESULT_JSONL"

run -m benchmarks.bench_e2e \
  --model-name "$MODEL_NAME" \
  --num-blocks "$E2E_NUM_BLOCKS" \
  --max-concurrent "$E2E_MAX_CONCURRENT" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --warmup-iters "$WARMUP_ITERS" \
  --measure-iters "$MEASURE_ITERS" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --output-file "$RESULT_JSONL"

run -m benchmarks.bench_engine \
  --model-name "$MODEL_NAME" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --num-prompts "$ENGINE_NUM_PROMPTS" \
  --prompt-len "$ENGINE_PROMPT_LEN" \
  --output-len "$ENGINE_OUTPUT_LEN" \
  --max-concurrent "$ENGINE_MAX_CONCURRENT" \
  --warmup-requests "$ENGINE_WARMUP_REQUESTS" \
  --output-file "$RESULT_JSONL"

printf 'saved results to %s\n' "$RESULT_JSONL"
