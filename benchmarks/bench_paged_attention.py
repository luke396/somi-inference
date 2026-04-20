"""Benchmark paged_attention_decode directly."""

from __future__ import annotations

import argparse

import torch

from benchmarks.common import (
    DEFAULT_BLOCK_SIZE,
    append_jsonl,
    collect_environment_metadata,
    measure_runtime,
    print_result,
    required_blocks,
    resolve_device,
    resolve_dtype,
    seed_everything,
    summarize_latencies,
)
from somi_inference.core.paged_attention import pack_kv_cache, paged_attention_decode


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark the raw paged attention decode kernel."
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
        help="Kernel dtype. 'auto' picks a device-appropriate default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic benchmark tensors.",
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
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Sequence batch sizes to benchmark.",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[128, 512, 2048, 4096],
        help="Context lengths to benchmark.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Paged attention block size.",
    )
    parser.add_argument(
        "--num-q-heads",
        type=int,
        default=8,
        help="Number of query heads.",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=4,
        help="Number of KV heads.",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=32,
        help="Per-head hidden dimension.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="torch_ref",
        choices=["torch_ref", "triton"],
        help="Attention backend to benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the paged attention benchmark."""
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    seed_everything(args.seed)
    environment = collect_environment_metadata(device)

    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lens:
            max_blocks_per_seq = required_blocks(seq_len, args.block_size)
            num_blocks = batch_size * max_blocks_per_seq
            q = torch.randn(
                batch_size,
                args.num_q_heads,
                args.head_dim,
                device=device,
                dtype=dtype,
            )
            key_cache = torch.randn(
                num_blocks,
                args.block_size,
                args.num_kv_heads,
                args.head_dim,
                device=device,
                dtype=dtype,
            )
            value_cache = torch.randn_like(key_cache)
            kv_cache = pack_kv_cache(key_cache, value_cache)
            block_tables = torch.arange(
                num_blocks, device=device, dtype=torch.long
            ).reshape(batch_size, max_blocks_per_seq)
            seq_lens = torch.full(
                (batch_size,), seq_len, device=device, dtype=torch.long
            )

            def run_once(
                q_tensor: torch.Tensor = q,
                kv_cache_tensor: torch.Tensor = kv_cache,
                block_tables_tensor: torch.Tensor = block_tables,
                seq_lens_tensor: torch.Tensor = seq_lens,
            ) -> torch.Tensor:
                return paged_attention_decode(
                    q_tensor,
                    kv_cache_tensor,
                    block_tables_tensor,
                    seq_lens_tensor,
                    backend=args.backend,
                )

            latencies = measure_runtime(
                run_once,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                device=device,
            )
            metrics = summarize_latencies(latencies)
            total_decode_tokens = batch_size * args.measure_iters
            total_context_tokens = batch_size * seq_len * args.measure_iters
            total_time = sum(latencies)
            metrics["decode_tokens_per_s"] = total_decode_tokens / total_time
            metrics["context_tokens_per_s"] = total_context_tokens / total_time
            payload = {
                "benchmark": "paged_attention",
                "config": {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "block_size": args.block_size,
                    "num_q_heads": args.num_q_heads,
                    "num_kv_heads": args.num_kv_heads,
                    "head_dim": args.head_dim,
                    "backend": args.backend,
                    "device": str(device),
                    "dtype": str(dtype),
                    "warmup_iters": args.warmup_iters,
                    "measure_iters": args.measure_iters,
                },
                "metrics": metrics,
                "environment": environment,
            }
            print_result(
                title=" Paged Attention Benchmark ",
                config=payload["config"],
                metrics=payload["metrics"],
            )
            append_jsonl(args.output_file, payload)


if __name__ == "__main__":
    main()
