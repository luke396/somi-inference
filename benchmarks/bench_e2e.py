"""Benchmark deterministic workload turns via LLM.generate()."""

from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING, cast, get_args

import torch

from benchmarks.common import (
    append_jsonl,
    collect_environment_metadata,
    measure_runtime,
    print_result,
    seed_everything,
    summarize_latencies,
    synchronize,
)
from benchmarks.workloads import (
    DEFAULT_BASE_PROMPT_SEED,
    PresetName,
    WorkloadName,
    WorkloadTurnCase,
    build_workload_turn_cases,
    filter_turn_cases_by_output_tokens,
)
from somi_inference.entrypoints.llm import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_CONCURRENT,
    LLM,
)

if TYPE_CHECKING:
    from typing import Any

    from somi_inference.core.paged_attention import PagedAttentionBackend
    from somi_inference.entrypoints.llm import MLPBackend, PrefillAttentionBackend

DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
EXPLICIT_BACKEND_CHOICES = ("torch_ref", "triton")


def _measure_prompt_generation(
    *,
    llm: LLM,
    prompt_text: str,
    prompt_token_count: int,
    max_new_tokens: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Measure TTFT and full-generation latency for one prompt."""

    def generate_once(token_limit: int, prompt: str = prompt_text) -> str:
        return llm.generate(
            prompt,
            max_new_tokens=token_limit,
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
        generate_once(max_new_tokens)
        synchronize(device)

    generation_latencies: list[float] = []
    output_token_counts: list[int] = []
    for _ in range(args.measure_iters):
        synchronize(device)
        start = time.perf_counter()
        output_text = generate_once(max_new_tokens)
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
    return metrics


def _base_benchmark_config(
    *,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, str | int]:
    """Return the config fields shared by every E2E benchmark case."""
    return {
        "model_name": args.model_name,
        "num_blocks": args.num_blocks,
        "block_size": args.block_size,
        "max_concurrent": args.max_concurrent,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "device": str(device),
        "dtype": str(dtype),
        "attention_backend": args.attention_backend,
        "decode_attention_backend": args.decode_attention_backend,
        "mlp_backend": args.mlp_backend,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
    }


def _build_benchmark_payload(
    *,
    case: WorkloadTurnCase,
    metrics: dict[str, float],
    base_config: dict[str, str | int],
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSONL payload for one workload turn."""
    config = dict(base_config)
    config.update(
        {
            "mode": "workload_turn",
            "workload": case.workload,
            "preset": case.preset,
            "scenario": case.scenario,
            "session_id": case.session_id,
            "turn_idx": case.turn_idx,
            "num_turns": case.num_turns,
            "base_prompt_tokens": case.base_prompt_tokens,
            "user_tokens": case.user_tokens,
            "tool_tokens": case.tool_tokens,
            "prompt_chars": len(case.prompt_text),
            "requested_prompt_tokens": case.requested_prompt_tokens,
            "actual_prompt_tokens": case.actual_prompt_tokens,
            "prompt_tokens": case.actual_prompt_tokens,
            "requested_output_tokens": case.requested_output_tokens,
            "max_new_tokens": case.requested_output_tokens,
        }
    )
    return {
        "benchmark": "e2e",
        "config": config,
        "metrics": metrics,
        "environment": environment,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark deterministic multi-turn chat or agent workloads via "
            "LLM.generate()."
        )
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
        required=True,
        choices=["cpu", "cuda"],
        help="Execution device.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        required=True,
        choices=tuple(DTYPE_BY_NAME),
        help="Model and KV-cache dtype.",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        required=True,
        choices=EXPLICIT_BACKEND_CHOICES,
        help=(
            "Prefill attention backend. "
            "`cuda` is selected by --device, not here."
        ),
    )
    parser.add_argument(
        "--decode-attention-backend",
        type=str,
        required=True,
        choices=EXPLICIT_BACKEND_CHOICES,
        help=(
            "Decode paged-attention backend. "
            "`cuda` is selected by --device, not here."
        ),
    )
    parser.add_argument(
        "--mlp-backend",
        type=str,
        required=True,
        choices=EXPLICIT_BACKEND_CHOICES,
        help=(
            "MLP projection backend. "
            "`cuda` is selected by --device, not here."
        ),
    )
    parser.add_argument(
        "--workload",
        type=str,
        required=True,
        choices=get_args(WorkloadName),
        help="Deterministic workload family to benchmark.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="mid",
        choices=get_args(PresetName),
        help="Preset size used by --workload.",
    )
    parser.add_argument(
        "--base-prompt",
        type=str,
        default=DEFAULT_BASE_PROMPT_SEED,
        help="Seed text used to synthesize the base system prompt.",
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
        "--output-tokens",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of requested output-token variants to benchmark.",
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
    return parser.parse_args(argv)


def main() -> None:
    """Run the end-to-end benchmark."""
    args = parse_args()
    device = torch.device(args.device)
    dtype = DTYPE_BY_NAME[args.dtype]
    seed_everything(args.seed)
    environment = collect_environment_metadata(device)

    llm = LLM(
        args.model_name,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_concurrent=args.max_concurrent,
        device=device,
        dtype=dtype,
        prefill_attention_backend=cast(
            "PrefillAttentionBackend", args.attention_backend
        ),
        decode_attention_backend=cast(
            "PagedAttentionBackend", args.decode_attention_backend
        ),
        mlp_backend=cast("MLPBackend", args.mlp_backend),
    )
    turn_cases = build_workload_turn_cases(
        tokenizer=llm.tokenizer,
        workload=cast("WorkloadName", args.workload),
        preset=cast("PresetName", args.preset),
        base_prompt_seed=args.base_prompt,
    )
    turn_cases = filter_turn_cases_by_output_tokens(
        turn_cases,
        None if args.output_tokens is None else tuple(args.output_tokens),
    )
    base_config = _base_benchmark_config(args=args, device=device, dtype=dtype)

    for case in turn_cases:
        metrics = _measure_prompt_generation(
            llm=llm,
            prompt_text=case.prompt_text,
            prompt_token_count=case.actual_prompt_tokens,
            max_new_tokens=case.requested_output_tokens,
            args=args,
            device=device,
        )
        payload = _build_benchmark_payload(
            case=case,
            metrics=metrics,
            base_config=base_config,
            environment=environment,
        )
        print_result(
            title=" E2E Workload Turn ",
            config=payload["config"],
            metrics=payload["metrics"],
        )
        append_jsonl(args.output_file, payload)


if __name__ == "__main__":
    main()
