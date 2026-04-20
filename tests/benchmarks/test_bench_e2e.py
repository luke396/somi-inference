"""Tests for workload benchmark helpers."""

from __future__ import annotations

from collections import defaultdict

import benchmarks.workloads as workloads
import pytest

from benchmarks import bench_e2e


class FakeTokenizer:
    """Simple tokenizer whose token count equals string length."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))

    def decode(self, token_ids: list[int]) -> str:
        return "x" * len(token_ids)


def test_make_target_prompt_tracks_requested_token_count() -> None:
    """The target prompt builder should hit the requested length for simple tokenizers."""
    tokenizer = FakeTokenizer()

    prompt = workloads.make_target_prompt(tokenizer, "seed", 24)

    assert len(tokenizer.encode(prompt)) == 24


def test_build_workload_turn_cases_expands_agent_sessions() -> None:
    """Agent workloads should expand into one six-turn session per output variant."""
    tokenizer = FakeTokenizer()

    cases = workloads.build_workload_turn_cases(
        tokenizer=tokenizer,
        workload="agent-session",
        preset="mid",
        base_prompt_seed="seed",
    )

    assert len(cases) == 12
    assert {case.session_id for case in cases} == {
        "agent-session-mid-out1",
        "agent-session-mid-out32",
    }

    cases_by_session: dict[str, list[workloads.WorkloadTurnCase]] = defaultdict(list)
    for case in cases:
        cases_by_session[case.session_id].append(case)

    for session_id, session_cases in cases_by_session.items():
        assert [case.turn_idx for case in session_cases] == [1, 2, 3, 4, 5, 6]
        assert [case.tool_tokens for case in session_cases] == [0, 256, 0, 256, 0, 0]
        assert all(case.user_tokens == 64 for case in session_cases)
        assert [case.requested_prompt_tokens for case in session_cases] == sorted(
            case.requested_prompt_tokens for case in session_cases
        )
        assert [case.actual_prompt_tokens for case in session_cases] == sorted(
            case.actual_prompt_tokens for case in session_cases
        )
        expected_output_tokens = 1 if session_id.endswith("out1") else 32
        assert all(
            case.requested_output_tokens == expected_output_tokens
            for case in session_cases
        )


def test_build_workload_turn_cases_expands_chat_sessions() -> None:
    """Chat workloads should produce one six-turn session per decode-length variant."""
    tokenizer = FakeTokenizer()

    cases = workloads.build_workload_turn_cases(
        tokenizer=tokenizer,
        workload="chat-serving",
        preset="long",
        base_prompt_seed="seed",
    )

    assert len(cases) == 18
    assert {case.requested_output_tokens for case in cases} == {64, 128, 256}
    assert all(case.tool_tokens == 0 for case in cases)
    assert all(case.base_prompt_tokens == 512 for case in cases)


def test_parse_args_accepts_explicit_execution_config() -> None:
    """bench_e2e should require explicit device, dtype, and backend choices."""
    args = bench_e2e.parse_args(
        [
            "--model-name",
            "Qwen/Qwen2.5-0.5B",
            "--device",
            "cuda",
            "--dtype",
            "float16",
            "--attention-backend",
            "triton",
            "--decode-attention-backend",
            "triton",
            "--mlp-backend",
            "triton",
            "--workload",
            "agent-session",
        ]
    )

    assert args.device == "cuda"
    assert args.dtype == "float16"
    assert args.attention_backend == "triton"
    assert args.decode_attention_backend == "triton"
    assert args.mlp_backend == "triton"


def test_parse_args_accepts_output_token_filter() -> None:
    """bench_e2e should accept filtering output-token variants."""
    args = bench_e2e.parse_args(
        [
            "--model-name",
            "Qwen/Qwen2.5-0.5B",
            "--device",
            "cuda",
            "--dtype",
            "float16",
            "--attention-backend",
            "torch_ref",
            "--decode-attention-backend",
            "torch_ref",
            "--mlp-backend",
            "torch_ref",
            "--workload",
            "agent-session",
            "--output-tokens",
            "1",
        ]
    )

    assert args.output_tokens == [1]


def test_parse_args_rejects_auto_values() -> None:
    """bench_e2e should no longer accept fuzzy auto choices."""
    with pytest.raises(SystemExit):
        bench_e2e.parse_args(
            [
                "--model-name",
                "Qwen/Qwen2.5-0.5B",
                "--device",
                "auto",
                "--dtype",
                "float16",
                "--attention-backend",
                "torch_ref",
                "--decode-attention-backend",
                "torch_ref",
                "--mlp-backend",
                "torch_ref",
                "--workload",
                "agent-session",
            ]
        )
