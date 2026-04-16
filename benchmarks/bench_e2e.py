"""Benchmark single-request end-to-end generation via LLM.generate()."""

from __future__ import annotations

import argparse
import time
from typing import Literal, cast

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

MLPBackend = Literal["auto", "torch_ref", "triton"]


def _make_target_prompt(llm: LLM, base_prompt: str, target_tokens: int) -> str:
    """Build a prompt string whose encoded length closely tracks `target_tokens`."""
    if target_tokens <= 0:
        message = "--prompt-lens values must be positive."
        raise ValueError(message)
    base_token_ids = llm.tokenizer.encode(base_prompt)
    if not base_token_ids:
        message = "Prompt text produced zero tokens; provide a non-empty prompt."
        raise ValueError(message)
    target_decode_len = target_tokens
    best_prompt = base_prompt
    best_distance = float("inf")

    for _ in range(8):
        repeats = (target_decode_len + len(base_token_ids) - 1) // len(base_token_ids)
        repeated_ids = (base_token_ids * repeats)[:target_decode_len]
        prompt_text = llm.tokenizer.decode(repeated_ids)
        actual_tokens = len(llm.tokenizer.encode(prompt_text))
        distance = abs(actual_tokens - target_tokens)
        if distance < best_distance:
            best_prompt = prompt_text
            best_distance = distance
        if actual_tokens == target_tokens:
            return prompt_text
        target_decode_len = max(target_decode_len + (target_tokens - actual_tokens), 1)

    return best_prompt


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
        "--mlp-backend",
        type=str,
        default="auto",
        choices=["auto", "torch_ref", "triton"],
        help=(
            "MLP projection backend. "
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
        "--prompt-lens",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional target prompt token lengths. "
            "When set, bench_e2e runs once per length using "
            "a decoded repeated-token prompt."
        ),
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
        mlp_backend=cast("MLPBackend", args.mlp_backend),
    )
    prompt_texts = (
        [
            _make_target_prompt(llm, args.prompt, prompt_len)
            for prompt_len in args.prompt_lens
        ]
        if args.prompt_lens is not None
        else [args.prompt]
    )

    requested_prompt_tokens = args.prompt_lens or [None] * len(prompt_texts)

    for prompt_text, requested_prompt_len in zip(
        prompt_texts,
        requested_prompt_tokens,
        strict=True,
    ):
        prompt_token_count = len(llm.tokenizer.encode(prompt_text))

        def generate_once(max_new_tokens: int, prompt: str = prompt_text) -> str:
            return llm.generate(
                prompt,
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
                "prompt_chars": len(prompt_text),
                "requested_prompt_tokens": requested_prompt_len,
                "prompt_tokens": prompt_token_count,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "device": str(device),
                "dtype": str(dtype),
                "attention_backend": args.attention_backend,
                "mlp_backend": args.mlp_backend,
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
