"""Benchmark prefill throughput and latency."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import torch

from benchmarks.common import (
    DEFAULT_BLOCK_SIZE,
    append_jsonl,
    collect_environment_metadata,
    create_kv_manager,
    load_benchmark_adapter,
    measure_runtime,
    print_result,
    required_blocks,
    resolve_device,
    resolve_dtype,
    seed_everything,
    summarize_latencies,
)

if TYPE_CHECKING:
    from somi_inference.core.paged_attention import KVCacheManager


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Benchmark prefill latency.")
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
        help="Random seed for prompt generation and synthetic weights.",
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
        "--prompt-lens",
        type=int,
        nargs="+",
        default=[32, 128, 512, 2048],
        help="Prompt lengths to benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the prefill benchmark."""
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    seed_everything(args.seed)
    adapter, config = load_benchmark_adapter(args.model_name, device, dtype)
    environment = collect_environment_metadata(device)

    for prompt_len in args.prompt_lens:
        if prompt_len > config["max_position_embeddings"]:
            message = (
                f"prompt_len={prompt_len} exceeds "
                f"max_position_embeddings={config['max_position_embeddings']}"
            )
            raise ValueError(message)

        kv_manager = create_kv_manager(
            config,
            num_blocks=required_blocks(prompt_len, args.block_size) + 1,
            block_size=args.block_size,
            device=device,
            dtype=dtype,
        )
        input_ids = torch.randint(
            config["vocab_size"],
            (1, prompt_len),
            device=device,
            dtype=torch.long,
        )
        next_seq_id = 0

        def run_once(
            cache_manager: KVCacheManager = kv_manager,
            prompt_ids: torch.Tensor = input_ids,
        ) -> torch.Tensor:
            nonlocal next_seq_id
            seq_id = next_seq_id
            next_seq_id += 1
            cache_manager.register_sequence(seq_id)
            with torch.inference_mode():
                logits = adapter.prefill(prompt_ids, cache_manager, seq_id)
            cache_manager.free_sequence(seq_id)
            return logits[:, -1, :]

        latencies = measure_runtime(
            run_once,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            device=device,
        )
        metrics = summarize_latencies(latencies)
        total_tokens = prompt_len * args.measure_iters
        total_time = sum(latencies)
        metrics["input_tokens_per_s"] = total_tokens / total_time
        payload = {
            "benchmark": "prefill",
            "config": {
                "model_name": args.model_name,
                "prompt_len": prompt_len,
                "block_size": args.block_size,
                "device": str(device),
                "dtype": str(dtype),
                "warmup_iters": args.warmup_iters,
                "measure_iters": args.measure_iters,
            },
            "metrics": metrics,
            "environment": environment,
        }
        print_result(
            title=" Prefill Benchmark ",
            config=payload["config"],
            metrics=payload["metrics"],
        )
        append_jsonl(args.output_file, payload)


if __name__ == "__main__":
    main()
