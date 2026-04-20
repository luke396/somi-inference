"""Benchmark fixed-context single-step decode throughput and latency."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Protocol, cast

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
    from somi_inference.models.base import ModelAdapter


class DecodeBenchmarkAdapter(Protocol):
    """Benchmark adapter contract for selecting decode-stage backends."""

    decode_attention_backend: str
    mlp_backend: str

    def prefill(
        self,
        input_ids: torch.Tensor,
        kv_manager: KVCacheManager,
        seq_id: int,
    ) -> torch.Tensor:
        """Prefill a single sequence."""

    def decode(
        self,
        input_ids: torch.Tensor,
        kv_manager: KVCacheManager,
        seq_ids: list[int],
    ) -> torch.Tensor:
        """Decode one token for each active sequence."""


class FixedContextDecodeBenchmark:
    """Own the per-iteration state for fixed-context decode benchmarking."""

    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        config: dict[str, int],
        batch_size: int,
        context_len: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Capture immutable benchmark inputs and mutable per-iteration state."""
        self.adapter = adapter
        self.config = config
        self.batch_size = batch_size
        self.context_len = context_len
        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.num_blocks = batch_size * required_blocks(context_len + 1, block_size)
        self.seq_ids = list(range(batch_size))
        self.prompts = [
            torch.randint(
                config["vocab_size"],
                (1, context_len),
                device=device,
                dtype=torch.long,
            )
            for _ in self.seq_ids
        ]
        self.kv_manager: KVCacheManager | None = None
        self.input_ids: torch.Tensor | None = None

    def prepare_iteration(self) -> None:
        """Rebuild KV state so each timed run measures one fixed decode step."""
        self.kv_manager = create_kv_manager(
            self.config,
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            device=self.device,
            dtype=self.dtype,
        )
        token_inputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for seq_id, prompt in zip(self.seq_ids, self.prompts, strict=True):
                self.kv_manager.register_sequence(seq_id)
                logits = self.adapter.prefill(prompt, self.kv_manager, seq_id)
                token_inputs.append(
                    torch.argmax(logits[:, 0, :], dim=-1, keepdim=True)
                )
        self.input_ids = torch.cat(token_inputs, dim=0)

    def run_once(self) -> torch.Tensor:
        """Decode exactly one token for the prepared batch."""
        assert self.kv_manager is not None
        assert self.input_ids is not None
        with torch.inference_mode():
            return self.adapter.decode(self.input_ids, self.kv_manager, self.seq_ids)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark fixed-context single-step decode latency."
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
        "--decode-attention-backend",
        type=str,
        default="torch_ref",
        choices=["torch_ref", "triton"],
        help="Decode paged-attention backend.",
    )
    parser.add_argument(
        "--mlp-backend",
        type=str,
        default="torch_ref",
        choices=["torch_ref", "triton"],
        help="MLP projection backend used during decode.",
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
        help="Decode batch sizes to benchmark.",
    )
    parser.add_argument(
        "--context-lens",
        type=int,
        nargs="+",
        default=[128, 512, 2048],
        help="Fixed context lengths to benchmark.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Run the decode benchmark."""
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    seed_everything(args.seed)
    adapter, config = load_benchmark_adapter(args.model_name, device, dtype)
    benchmark_adapter = cast("DecodeBenchmarkAdapter", adapter)
    benchmark_adapter.decode_attention_backend = args.decode_attention_backend
    benchmark_adapter.mlp_backend = args.mlp_backend
    environment = collect_environment_metadata(device)

    for batch_size in args.batch_sizes:
        for context_len in args.context_lens:
            if context_len + 1 > config["max_position_embeddings"]:
                message = (
                    f"context_len={context_len} exceeds "
                    f"max_position_embeddings={config['max_position_embeddings']}"
                )
                raise ValueError(message)

            benchmark_case = FixedContextDecodeBenchmark(
                adapter=adapter,
                config=config,
                batch_size=batch_size,
                context_len=context_len,
                block_size=args.block_size,
                device=device,
                dtype=dtype,
            )

            latencies = measure_runtime(
                benchmark_case.run_once,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                device=device,
                before_each=benchmark_case.prepare_iteration,
            )
            metrics = summarize_latencies(latencies)
            total_decode_tokens = batch_size * args.measure_iters
            total_context_tokens = batch_size * context_len * args.measure_iters
            total_time = sum(latencies)
            metrics["output_tokens_per_s"] = total_decode_tokens / total_time
            metrics["attended_context_tokens_per_s"] = (
                total_context_tokens / total_time
            )
            payload = {
                "benchmark": "decode",
                "config": {
                    "model_name": args.model_name,
                    "mode": "fixed_context_single_step",
                    "batch_size": batch_size,
                    "context_len": context_len,
                    "block_size": args.block_size,
                    "device": str(device),
                    "dtype": str(dtype),
                    "decode_attention_backend": args.decode_attention_backend,
                    "mlp_backend": args.mlp_backend,
                    "warmup_iters": args.warmup_iters,
                    "measure_iters": args.measure_iters,
                },
                "metrics": metrics,
                "environment": environment,
            }
            print_result(
                title=" Decode Benchmark ",
                config=payload["config"],
                metrics=payload["metrics"],
            )
            append_jsonl(args.output_file, payload)


if __name__ == "__main__":
    main()
