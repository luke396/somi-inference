"""Tests for continuous batching components."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, cast

import pytest
import torch
from torch import Tensor

from somi_inference.core.continuous_batching import (
    ContinuousBatchingEngine,
    Scheduler,
    Sequence,
    SequenceStatus,
)
from somi_inference.core.paged_attention import KVCacheManager
from somi_inference.core.sampler import SamplingParams

if TYPE_CHECKING:
    from somi_inference.core.model_runner import ModelRunner

BLOCK_SIZE = 4
TOTAL_BLOCKS = 10
MAX_CONCURRENT = 2
EOS_TOKEN_ID = 2
NON_EOS_TOKEN = 5
PREFILL_TOKEN_0 = 10
PREFILL_TOKEN_1 = 20
DECODE_TOKEN_0 = 11
DECODE_TOKEN_1 = 21


class MockModelRunner:
    """ModelRunner stub that returns pre-baked token outputs."""

    def __init__(
        self,
        kv_manager: KVCacheManager,
        prefill_tokens: dict[int, int],
        decode_batches: list[Tensor] | None = None,
    ) -> None:
        """Initialize the stub with deterministic outputs."""
        self.kv_manager = kv_manager
        self.prefill_tokens = prefill_tokens
        self.decode_batches = list(decode_batches or [])
        self.prefill_calls: list[tuple[Tensor, int, SamplingParams]] = []
        self.decode_calls: list[
            tuple[Tensor, list[int], list[SamplingParams], list[list[int]]]
        ] = []

    def prefill(
        self,
        input_ids: Tensor,
        seq_id: int,
        sampling_params: SamplingParams,
    ) -> int:
        """Record the prefill request and return a fixed token."""
        self.prefill_calls.append((input_ids.clone(), seq_id, sampling_params))
        return self.prefill_tokens[seq_id]

    def decode(
        self,
        input_ids: Tensor,
        seq_ids: list[int],
        sampling_params: list[SamplingParams],
        token_histories: list[list[int]],
    ) -> Tensor:
        """Record the decode batch and return the next token batch."""
        history_copy = [list(history) for history in token_histories]
        self.decode_calls.append(
            (input_ids.clone(), list(seq_ids), list(sampling_params), history_copy)
        )
        return self.decode_batches.pop(0).clone()


def make_sequence(
    seq_id: int,
    prompt_tokens: list[int],
    max_new_tokens: int,
    *,
    status: SequenceStatus = SequenceStatus.WAITING,
    output_tokens: list[int] | None = None,
    sampling_params: SamplingParams | None = None,
) -> Sequence:
    """Build a sequence with sensible defaults for tests."""
    return Sequence(
        seq_id=seq_id,
        status=status,
        prompt_tokens=prompt_tokens,
        output_tokens=list(output_tokens or []),
        max_new_tokens=max_new_tokens,
        sampling_params=sampling_params or SamplingParams(),
    )


@pytest.fixture
def kv_manager() -> KVCacheManager:
    """Create a KVCacheManager for testing."""
    return KVCacheManager(
        num_blocks=TOTAL_BLOCKS,
        block_size=BLOCK_SIZE,
        num_kv_heads=2,
        head_dim=8,
        n_layers=1,
    )


@pytest.fixture
def scheduler(kv_manager: KVCacheManager) -> Scheduler:
    """Create a Scheduler for testing."""
    return Scheduler(
        max_concurrent=MAX_CONCURRENT,
        block_size=BLOCK_SIZE,
        total_blocks=kv_manager.allocator.num_free_blocks(),
    )


class TestScheduler:
    """Test Scheduler functionality."""

    def test_add_request(self) -> None:
        """Adding requests should append them to the waiting queue."""
        scheduler = Scheduler(
            max_concurrent=MAX_CONCURRENT,
            block_size=BLOCK_SIZE,
            total_blocks=TOTAL_BLOCKS,
        )
        seq = make_sequence(0, [1, 2, 3], max_new_tokens=3)

        scheduler.add_request(seq)

        assert len(scheduler.waiting) == 1
        assert scheduler.waiting[0] == seq

    def test_schedule_admit_new(self) -> None:
        """Scheduler should admit requests when reservation budget allows it."""
        scheduler = Scheduler(
            max_concurrent=MAX_CONCURRENT,
            block_size=BLOCK_SIZE,
            total_blocks=TOTAL_BLOCKS,
        )
        sequences = [
            make_sequence(0, [1, 2, 3], max_new_tokens=3),
            make_sequence(1, [4, 5], max_new_tokens=2),
        ]
        for seq in sequences:
            scheduler.add_request(seq)

        output = scheduler.schedule()

        assert output.prefill_seq == sequences
        assert output.decode_seq == []
        assert scheduler.running == sequences
        assert len(scheduler.waiting) == 0

    def test_schedule_reserves_blocks_for_running_sequence_growth(self) -> None:
        """Running sequences should reserve future decode growth first."""
        scheduler = Scheduler(
            max_concurrent=MAX_CONCURRENT,
            block_size=BLOCK_SIZE,
            total_blocks=3,
        )
        running_seq = make_sequence(
            0,
            [1, 2, 3, 4],
            max_new_tokens=5,
            status=SequenceStatus.RUNNING,
            output_tokens=[PREFILL_TOKEN_0],
        )
        waiting_seq = make_sequence(1, [7] * 8, max_new_tokens=1)
        scheduler.running.append(running_seq)
        scheduler.add_request(waiting_seq)

        output = scheduler.schedule()

        assert output.decode_seq == [running_seq]
        assert output.prefill_seq == []
        assert list(scheduler.waiting) == [waiting_seq]
        assert scheduler.running == [running_seq]

    def test_schedule_releases_finished_reservation(self) -> None:
        """Finished sequences should release reserved capacity before new admission."""
        scheduler = Scheduler(
            max_concurrent=MAX_CONCURRENT,
            block_size=BLOCK_SIZE,
            total_blocks=2,
        )
        finished_seq = make_sequence(
            0,
            [1, 2, 3, 4],
            max_new_tokens=5,
            status=SequenceStatus.FINISHED,
            output_tokens=[PREFILL_TOKEN_0],
        )
        waiting_seq = make_sequence(1, [9], max_new_tokens=1)
        scheduler.running.append(finished_seq)
        scheduler.add_request(waiting_seq)

        output = scheduler.schedule()

        assert output.freed_seq_ids == [finished_seq.seq_id]
        assert output.prefill_seq == [waiting_seq]
        assert output.decode_seq == []
        assert scheduler.running == [waiting_seq]
        assert scheduler.finished == [finished_seq]

    def test_schedule_max_concurrent(self) -> None:
        """Scheduler should respect max_concurrent even with spare blocks."""
        scheduler = Scheduler(
            max_concurrent=1,
            block_size=BLOCK_SIZE,
            total_blocks=TOTAL_BLOCKS,
        )
        scheduler.add_request(make_sequence(0, [1, 2, 3], max_new_tokens=3))
        scheduler.add_request(make_sequence(1, [4, 5], max_new_tokens=2))

        first = scheduler.schedule()
        second = scheduler.schedule()

        assert len(first.prefill_seq) == 1
        assert len(scheduler.waiting) == 1
        assert len(second.decode_seq) == 1
        assert second.prefill_seq == []


class TestContinuousBatchingEngine:
    """Test ContinuousBatchingEngine functionality."""

    def test_single_sequence_prefill_decode(
        self,
        kv_manager: KVCacheManager,
        scheduler: Scheduler,
    ) -> None:
        """Engine should prefill once and decode until max_new_tokens."""
        expected_decode_steps = 2
        runner = cast(
            "ModelRunner",
            MockModelRunner(
                kv_manager,
                prefill_tokens={0: NON_EOS_TOKEN},
                decode_batches=[
                    torch.tensor([NON_EOS_TOKEN]),
                    torch.tensor([NON_EOS_TOKEN]),
                ],
            ),
        )
        engine = ContinuousBatchingEngine(runner, scheduler, eos_token_id=EOS_TOKEN_ID)
        seq = make_sequence(0, [1, 2, 3], max_new_tokens=3)
        requests = deque([(0, seq)])

        finished = engine.run(requests)
        runner_impl = cast("MockModelRunner", runner)

        assert len(finished) == 1
        assert finished[0].seq_id == 0
        assert finished[0].output_tokens == [NON_EOS_TOKEN] * 3
        assert len(runner_impl.prefill_calls) == 1
        assert len(runner_impl.decode_calls) == expected_decode_steps
        assert not kv_manager.seq_to_block

    def test_eos_token_handling(
        self,
        kv_manager: KVCacheManager,
        scheduler: Scheduler,
    ) -> None:
        """EOS from prefill should stop generation without decode."""
        runner = cast(
            "ModelRunner",
            MockModelRunner(kv_manager, prefill_tokens={0: EOS_TOKEN_ID}),
        )
        engine = ContinuousBatchingEngine(runner, scheduler, eos_token_id=EOS_TOKEN_ID)
        seq = make_sequence(0, [1, 2, 3], max_new_tokens=5)
        requests = deque([(0, seq)])

        finished = engine.run(requests)
        runner_impl = cast("MockModelRunner", runner)

        assert finished[0].output_tokens == [EOS_TOKEN_ID]
        assert runner_impl.decode_calls == []

    def test_decode_batch_forwards_full_token_history(
        self,
        kv_manager: KVCacheManager,
        scheduler: Scheduler,
    ) -> None:
        """Decode should receive per-sequence params and full token history."""
        params_0 = SamplingParams(temperature=0.0)
        params_1 = SamplingParams(temperature=0.8, top_p=0.9)
        runner = cast(
            "ModelRunner",
            MockModelRunner(
                kv_manager,
                prefill_tokens={0: PREFILL_TOKEN_0, 1: PREFILL_TOKEN_1},
                decode_batches=[torch.tensor([DECODE_TOKEN_0, DECODE_TOKEN_1])],
            ),
        )
        engine = ContinuousBatchingEngine(runner, scheduler, eos_token_id=EOS_TOKEN_ID)
        requests = deque(
            [
                (
                    0,
                    make_sequence(
                        0,
                        [1, 2, 3],
                        max_new_tokens=2,
                        sampling_params=params_0,
                    ),
                ),
                (
                    0,
                    make_sequence(
                        1,
                        [4, 5],
                        max_new_tokens=2,
                        sampling_params=params_1,
                    ),
                ),
            ]
        )

        finished = engine.run(requests)
        runner_impl = cast("MockModelRunner", runner)
        decode_input_ids, decode_seq_ids, decode_params, token_histories = (
            runner_impl.decode_calls[0]
        )

        assert [seq.seq_id for seq in finished] == [0, 1]
        assert finished[0].output_tokens == [PREFILL_TOKEN_0, DECODE_TOKEN_0]
        assert finished[1].output_tokens == [PREFILL_TOKEN_1, DECODE_TOKEN_1]
        torch.testing.assert_close(
            decode_input_ids,
            torch.tensor([[PREFILL_TOKEN_0], [PREFILL_TOKEN_1]]),
        )
        assert decode_seq_ids == [0, 1]
        assert decode_params == [params_0, params_1]
        assert token_histories == [
            [1, 2, 3, PREFILL_TOKEN_0],
            [4, 5, PREFILL_TOKEN_1],
        ]
