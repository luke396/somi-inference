"""Tests for decode benchmark CLI configuration."""

from __future__ import annotations

from benchmarks import bench_decode


def test_parse_args_accepts_explicit_decode_backends() -> None:
    """bench_decode should accept explicit decode and MLP backend choices."""
    args = bench_decode.parse_args(
        [
            "--model-name",
            "Qwen/Qwen2.5-0.5B",
            "--decode-attention-backend",
            "triton",
            "--mlp-backend",
            "triton",
        ]
    )

    assert args.decode_attention_backend == "triton"
    assert args.mlp_backend == "triton"
