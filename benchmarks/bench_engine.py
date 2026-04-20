"""Benchmark the continuous batching engine with workload arrivals."""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from typing import cast, get_args

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
from benchmarks.workloads import (
    DEFAULT_BASE_PROMPT_SEED,
    BenchmarkTokenizer,
    PresetName,
    WorkloadName,
    WorkloadTurnCase,
    build_workload_turn_cases,
)
from somi_inference.core.continuous_batching import (
    ContinuousBatchingEngine,
    Scheduler,
    Sequence,
    SequenceStatus,
)
from somi_inference.core.model_runner import ModelRunner
from somi_inference.core.sampler import Sampler, SamplingParams
from somi_inference.tokenizer import Tokenizer

WorkloadRequestEntry = tuple[int, WorkloadTurnCase, Sequence]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    return parser.parse_args(argv)


def main() -> None:
    """Run the engine benchmark."""
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    seed_everything(args.seed)
    adapter, config = load_benchmark_adapter(args.model_name, device, dtype)
    environment = collect_environment_metadata(device)

    tokenizer = Tokenizer(args.model_name)
    workload_request_entries = build_workload_request_entries(
        tokenizer=tokenizer,
        workload=cast("WorkloadName", args.workload),
        preset=cast("PresetName", args.preset),
        base_prompt_seed=args.base_prompt,
        arrival_pattern=args.arrival_pattern,
        arrival_rate=args.arrival_rate,
        seed=args.seed + 1,
    )
    max_sequence_tokens = max(
        entry[1].actual_prompt_tokens + entry[1].requested_output_tokens
        for entry in workload_request_entries
    )
    if max_sequence_tokens > config["max_position_embeddings"]:
        message = (
            f"workload sequence exceeds max_position_embeddings: "
            f"{max_sequence_tokens} > {config['max_position_embeddings']}"
        )
        raise ValueError(message)
    requests = strip_request_metadata(workload_request_entries)
    blocks_per_sequence = max(
        required_blocks(
            case.actual_prompt_tokens + case.requested_output_tokens,
            args.block_size,
        )
        for _, case, _ in workload_request_entries
    )
    warmup_requests = build_warmup_requests_from_entries(
        workload_request_entries,
        args.warmup_requests,
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

    if warmup_requests:
        engine.run(deque(warmup_requests))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    finished = engine.run(deque(requests))
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
        "config": build_engine_config(
            args=args,
            device=device,
            dtype=dtype,
            num_blocks=num_blocks,
            workload_request_entries=workload_request_entries,
        ),
        "metrics": metrics,
        "environment": environment,
    }
    print_result(
        title=" Engine Benchmark ",
        config=payload["config"],
        metrics=payload["metrics"],
    )
    append_jsonl(args.output_file, payload)

def build_workload_request_entries(
    *,
    tokenizer: BenchmarkTokenizer,
    workload: WorkloadName,
    preset: PresetName,
    base_prompt_seed: str,
    arrival_pattern: str,
    arrival_rate: float,
    seed: int,
) -> list[WorkloadRequestEntry]:
    """Build arrival-tagged engine requests from deterministic workload cases."""
    turn_cases = build_workload_turn_cases(
        tokenizer=tokenizer,
        workload=workload,
        preset=preset,
        base_prompt_seed=base_prompt_seed,
    )
    cases_by_session: dict[str, list[WorkloadTurnCase]] = {}
    for case in turn_cases:
        cases_by_session.setdefault(case.session_id, []).append(case)

    rng = np.random.default_rng(seed)
    session_start_steps = make_arrival_steps(
        num_prompts=len(cases_by_session),
        arrival_pattern=arrival_pattern,
        arrival_rate=arrival_rate,
        rng=rng,
    )

    request_entries: list[WorkloadRequestEntry] = []
    next_seq_id = 0
    for (session_id, session_cases), start_step in zip(
        cases_by_session.items(),
        session_start_steps,
        strict=True,
    ):
        del session_id
        for case in sorted(session_cases, key=lambda item: item.turn_idx):
            request_entries.append(
                (
                    start_step + case.turn_idx - 1,
                    case,
                    Sequence(
                        seq_id=next_seq_id,
                        status=SequenceStatus.WAITING,
                        prompt_tokens=tokenizer.encode(case.prompt_text),
                        output_tokens=[],
                        max_new_tokens=case.requested_output_tokens,
                        sampling_params=SamplingParams(temperature=0.0),
                    ),
                )
            )
            next_seq_id += 1

    request_entries.sort(key=lambda item: (item[0], item[2].seq_id))
    return request_entries


def strip_request_metadata(
    request_entries: list[WorkloadRequestEntry],
) -> list[tuple[int, Sequence]]:
    """Drop workload metadata and return engine-ready requests."""
    return [(arrival_step, seq) for arrival_step, _, seq in request_entries]


def build_warmup_requests_from_entries(
    request_entries: list[WorkloadRequestEntry],
    warmup_requests: int,
) -> list[tuple[int, Sequence]]:
    """Reuse the first workload requests as a cheap warmup input."""
    requests: list[tuple[int, Sequence]] = []
    for seq_id, (_, _, seq) in enumerate(request_entries[:warmup_requests]):
        requests.append(
            (
                0,
                Sequence(
                    seq_id=seq_id,
                    status=SequenceStatus.WAITING,
                    prompt_tokens=list(seq.prompt_tokens),
                    output_tokens=[],
                    max_new_tokens=min(seq.max_new_tokens, 8),
                    sampling_params=SamplingParams(temperature=0.0),
                ),
            )
        )
    return requests


def build_engine_config(
    *,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    num_blocks: int,
    workload_request_entries: list[WorkloadRequestEntry],
) -> dict[str, object]:
    """Build the JSONL config payload for one engine benchmark run."""
    prompt_tokens = [len(seq.prompt_tokens) for _, _, seq in workload_request_entries]
    output_tokens = [seq.max_new_tokens for _, _, seq in workload_request_entries]
    config: dict[str, object] = {
        "model_name": args.model_name,
        "mode": "workload_turn_trace",
        "workload": args.workload,
        "preset": args.preset,
        "arrival_pattern": args.arrival_pattern,
        "arrival_rate": args.arrival_rate,
        "max_concurrent": args.max_concurrent,
        "num_blocks": num_blocks,
        "block_size": args.block_size,
        "device": str(device),
        "dtype": str(dtype),
        "num_requests": len(workload_request_entries),
        "num_sessions": len(
            {case.session_id for _, case, _ in workload_request_entries}
        ),
        "min_prompt_tokens": min(prompt_tokens),
        "max_prompt_tokens": max(prompt_tokens),
        "min_output_tokens": min(output_tokens),
        "max_output_tokens": max(output_tokens),
    }
    return config


def make_arrival_steps(
    *,
    num_prompts: int,
    arrival_pattern: str,
    arrival_rate: float,
    rng: np.random.Generator,
) -> list[int]:
    """Generate integer arrival steps for the engine scheduler."""
    if arrival_rate <= 0.0:
        message = "arrival_rate must be > 0.0"
        raise ValueError(message)
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
