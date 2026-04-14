"""Benchmark single-request end-to-end generation via LLM.generate()."""

from __future__ import annotations

import argparse
import time

from benchmarks.common import (
    append_jsonl,
    collect_environment_metadata,
    measure_runtime,
    print_result,
    resolve_device,
    resolve_dtype,
    seed_everything,
    summarize_latencies,
    synchronize,
)
from somi_inference.entrypoints.llm import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_CONCURRENT,
    LLM,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark single-request end-to-end generation latency."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HF model name or local path.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=256,
        help="Total KV cache blocks for the LLM instance.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="KV cache block size.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help="Scheduler max concurrent sequences.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model / cache dtype. 'auto' picks a device-appropriate default.",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="auto",
        choices=["auto", "torch_ref", "triton"],
        help=(
            "Prefill attention backend. "
            "'auto' prefers Triton on supported CUDA inputs."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=2,
        help="Number of warmup iterations before timing.",
    )
    parser.add_argument(
        "--measure-iters",
        type=int,
        default=10,
        help="Number of timed iterations.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Append benchmark results to a JSONL file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain how paged attention works in one paragraph.",
        help="Single prompt to benchmark.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of generated tokens per iteration.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0.0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling cutoff. -1 disables top-k filtering.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling cutoff.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty applied by the sampler.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the end-to-end benchmark."""
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    seed_everything(args.seed)
    environment = collect_environment_metadata(device)

    llm = LLM(
        args.model_name,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_concurrent=args.max_concurrent,
        device=device,
        dtype=dtype,
        prefill_attention_backend=args.attention_backend,
    )
    prompt_token_count = len(llm.tokenizer.encode(args.prompt))

    def generate_once(max_new_tokens: int) -> str:
        return llm.generate(
            args.prompt,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

    ttft_latencies = measure_runtime(
        lambda: generate_once(1),
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        device=device,
    )

    for _ in range(args.warmup_iters):
        synchronize(device)
        generate_once(args.max_new_tokens)
        synchronize(device)

    generation_latencies: list[float] = []
    output_token_counts: list[int] = []
    for _ in range(args.measure_iters):
        synchronize(device)
        start = time.perf_counter()
        output_text = generate_once(args.max_new_tokens)
        synchronize(device)
        generation_latencies.append(time.perf_counter() - start)
        output_token_counts.append(len(llm.tokenizer.encode(output_text)))

    metrics = summarize_latencies(generation_latencies)
    metrics.update(
        {
            f"ttft_{key}": value
            for key, value in summarize_latencies(ttft_latencies).items()
        }
    )
    total_time = sum(generation_latencies)
    total_input_tokens = prompt_token_count * args.measure_iters
    total_output_tokens = sum(output_token_counts)
    metrics["mean_output_tokens"] = total_output_tokens / args.measure_iters
    metrics["input_tokens_per_s"] = total_input_tokens / total_time
    metrics["output_tokens_per_s"] = total_output_tokens / total_time
    metrics["total_tokens_per_s"] = (
        total_input_tokens + total_output_tokens
    ) / total_time

    payload = {
        "benchmark": "e2e",
        "config": {
            "model_name": args.model_name,
            "num_blocks": args.num_blocks,
            "block_size": args.block_size,
            "max_concurrent": args.max_concurrent,
            "prompt_chars": len(args.prompt),
            "prompt_tokens": prompt_token_count,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "device": str(device),
            "dtype": str(dtype),
            "attention_backend": args.attention_backend,
            "warmup_iters": args.warmup_iters,
            "measure_iters": args.measure_iters,
        },
        "metrics": metrics,
        "environment": environment,
    }
    print_result(
        title=" E2E Benchmark ",
        config=payload["config"],
        metrics=payload["metrics"],
    )
    append_jsonl(args.output_file, payload)


if __name__ == "__main__":
    main()
