"""Benchmark the continuous batching engine with synthetic arrivals."""

from __future__ import annotations

import argparse
import math
import time
from collections import deque

import numpy as np
import torch

from benchmarks.common import (
    DEFAULT_BLOCK_SIZE,
    append_jsonl,
    collect_environment_metadata,
    create_kv_manager,
    load_benchmark_adapter,
    print_result,
    required_blocks,
    resolve_device,
    resolve_dtype,
    seed_everything,
)
from somi_inference.core.continuous_batching import (
    ContinuousBatchingEngine,
    Scheduler,
    Sequence,
    SequenceStatus,
)
from somi_inference.core.model_runner import ModelRunner
from somi_inference.core.sampler import Sampler, SamplingParams


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark the continuous batching engine throughput."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HF model name or local path.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="KV cache block size.",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic workload generation.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Append benchmark results to a JSONL file.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=128,
        help="Number of requests in the benchmark workload.",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=256,
        help="Prompt length for each request.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=64,
        help="Target generated tokens per request.",
    )
    parser.add_argument(
        "--arrival-pattern",
        type=str,
        default="burst",
        choices=["burst", "uniform", "poisson"],
        help="How requests arrive into the scheduler.",
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=1.0,
        help="Average requests per engine step for uniform/poisson arrivals.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=16,
        help="Scheduler max concurrent sequences.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Total KV cache blocks. Defaults to enough blocks for max_concurrent.",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="Warmup request count before the main run.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the engine benchmark."""
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    seed_everything(args.seed)
    adapter, config = load_benchmark_adapter(args.model_name, device, dtype)
    environment = collect_environment_metadata(device)

    if args.prompt_len + args.output_len > config["max_position_embeddings"]:
        message = (
            f"prompt_len + output_len exceeds max_position_embeddings: "
            f"{args.prompt_len + args.output_len} > {config['max_position_embeddings']}"
        )
        raise ValueError(message)

    blocks_per_sequence = required_blocks(
        args.prompt_len + args.output_len, args.block_size
    )
    num_blocks = args.num_blocks or (args.max_concurrent * blocks_per_sequence)
    kv_manager = create_kv_manager(
        config,
        num_blocks=num_blocks,
        block_size=args.block_size,
        device=device,
        dtype=dtype,
    )
    runner = ModelRunner(adapter, Sampler(), kv_manager)
    scheduler = Scheduler(
        max_concurrent=args.max_concurrent,
        block_size=args.block_size,
        total_blocks=num_blocks,
    )
    engine = ContinuousBatchingEngine(runner, scheduler, eos_token_id=-1)

    warmup_requests = build_requests(
        num_prompts=args.warmup_requests,
        prompt_len=min(args.prompt_len, 32),
        output_len=min(args.output_len, 8),
        vocab_size=config["vocab_size"],
        arrival_pattern="burst",
        arrival_rate=1.0,
        seed=args.seed,
    )
    if warmup_requests:
        engine.run(deque(warmup_requests))

    requests = deque(
        build_requests(
            num_prompts=args.num_prompts,
            prompt_len=args.prompt_len,
            output_len=args.output_len,
            vocab_size=config["vocab_size"],
            arrival_pattern=args.arrival_pattern,
            arrival_rate=args.arrival_rate,
            seed=args.seed + 1,
        )
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    finished = engine.run(requests)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_time = time.perf_counter() - start

    total_input_tokens = sum(len(seq.prompt_tokens) for seq in finished)
    total_output_tokens = sum(len(seq.output_tokens) for seq in finished)
    metrics = {
        "duration_s": total_time,
        "completed_requests": len(finished),
        "request_throughput": len(finished) / total_time,
        "input_tokens_per_s": total_input_tokens / total_time,
        "output_tokens_per_s": total_output_tokens / total_time,
        "total_tokens_per_s": (total_input_tokens + total_output_tokens) / total_time,
        "mean_output_len": total_output_tokens / len(finished),
    }
    payload = {
        "benchmark": "engine",
        "config": {
            "model_name": args.model_name,
            "prompt_len": args.prompt_len,
            "output_len": args.output_len,
            "num_prompts": args.num_prompts,
            "arrival_pattern": args.arrival_pattern,
            "arrival_rate": args.arrival_rate,
            "max_concurrent": args.max_concurrent,
            "num_blocks": num_blocks,
            "block_size": args.block_size,
            "device": str(device),
            "dtype": str(dtype),
        },
        "metrics": metrics,
        "environment": environment,
    }
    print_result(
        title=" Engine Benchmark ",
        config=payload["config"],
        metrics=payload["metrics"],
    )
    append_jsonl(args.output_file, payload)


def build_requests(
    *,
    num_prompts: int,
    prompt_len: int,
    output_len: int,
    vocab_size: int,
    arrival_pattern: str,
    arrival_rate: float,
    seed: int,
) -> list[tuple[int, Sequence]]:
    """Build a deterministic synthetic workload for the engine benchmark."""
    if arrival_rate <= 0.0:
        message = "arrival_rate must be > 0.0"
        raise ValueError(message)

    rng = np.random.default_rng(seed)
    arrival_steps = make_arrival_steps(
        num_prompts=num_prompts,
        arrival_pattern=arrival_pattern,
        arrival_rate=arrival_rate,
        rng=rng,
    )
    requests: list[tuple[int, Sequence]] = []
    for seq_id, arrival_step in enumerate(arrival_steps):
        prompt_tokens = rng.integers(0, vocab_size, size=prompt_len).tolist()
        requests.append(
            (
                arrival_step,
                Sequence(
                    seq_id=seq_id,
                    status=SequenceStatus.WAITING,
                    prompt_tokens=prompt_tokens,
                    output_tokens=[],
                    max_new_tokens=output_len,
                    sampling_params=SamplingParams(temperature=0.0),
                ),
            )
        )
    return requests


def make_arrival_steps(
    *,
    num_prompts: int,
    arrival_pattern: str,
    arrival_rate: float,
    rng: np.random.Generator,
) -> list[int]:
    """Generate integer arrival steps for the engine scheduler."""
    if arrival_pattern == "burst":
        return [0] * num_prompts
    if arrival_pattern == "uniform":
        return [math.floor(index / arrival_rate) for index in range(num_prompts)]

    step = 0.0
    arrival_steps: list[int] = []
    for _ in range(num_prompts):
        step += float(rng.exponential(scale=1.0 / arrival_rate))
        arrival_steps.append(math.floor(step))
    return arrival_steps


if __name__ == "__main__":
    main()
