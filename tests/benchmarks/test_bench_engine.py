"""Tests for engine benchmark workload integration."""

from __future__ import annotations

import pytest

from benchmarks import bench_engine


class FakeTokenizer:
    """Simple tokenizer whose token count equals string length."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))

    def decode(self, token_ids: list[int]) -> str:
        return "x" * len(token_ids)


def test_build_workload_request_entries_burst_interleaves_sessions_by_turn() -> None:
    """Burst arrival should keep turn order within sessions and interleave turns."""
    entries = bench_engine.build_workload_request_entries(
        tokenizer=FakeTokenizer(),
        workload="agent-session",
        preset="mid",
        base_prompt_seed="seed",
        arrival_pattern="burst",
        arrival_rate=1.0,
        seed=42,
    )

    assert len(entries) == 12
    assert [arrival_step for arrival_step, _, _ in entries] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
    ]
    assert [
        (case.session_id, case.turn_idx)
        for _, case, _ in entries[:4]
    ] == [
        ("agent-session-mid-out1", 1),
        ("agent-session-mid-out32", 1),
        ("agent-session-mid-out1", 2),
        ("agent-session-mid-out32", 2),
    ]


def test_build_workload_request_entries_uses_case_output_lengths() -> None:
    """Each workload request should keep the per-scenario output token target."""
    entries = bench_engine.build_workload_request_entries(
        tokenizer=FakeTokenizer(),
        workload="chat-serving",
        preset="short",
        base_prompt_seed="seed",
        arrival_pattern="uniform",
        arrival_rate=2.0,
        seed=42,
    )

    assert len(entries) == 18
    assert {seq.max_new_tokens for _, _, seq in entries} == {64, 128, 256}


def test_parse_args_accepts_workload_mode() -> None:
    """bench_engine should accept workload and preset options."""
    args = bench_engine.parse_args(
        [
            "--model-name",
            "Qwen/Qwen2.5-0.5B",
            "--workload",
            "chat-serving",
            "--preset",
            "long",
        ]
    )

    assert args.workload == "chat-serving"
    assert args.preset == "long"


def test_parse_args_requires_workload_mode() -> None:
    """bench_engine should require an explicit workload family."""
    with pytest.raises(SystemExit):
        bench_engine.parse_args(
            [
                "--model-name",
                "Qwen/Qwen2.5-0.5B",
            ]
        )


def test_make_arrival_steps_rejects_non_positive_rate() -> None:
    """Arrival rate validation should reject non-positive values."""
    with pytest.raises(ValueError, match="arrival_rate must be > 0.0"):
        bench_engine.make_arrival_steps(
            num_prompts=2,
            arrival_pattern="uniform",
            arrival_rate=0.0,
            rng=bench_engine.np.random.default_rng(42),
        )
