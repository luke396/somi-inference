"""Phase 2 tests for continuous batching with sampling params and ModelRunner."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, cast

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
EOS_TOKEN_ID = 2


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
        params: SamplingParams,
    ) -> int:
        """Record the prefill request and return a fixed token."""
        self.prefill_calls.append((input_ids.clone(), seq_id, params))
        return self.prefill_tokens[seq_id]

    def decode(
        self,
        input_ids: Tensor,
        seq_ids: list[int],
        params: list[SamplingParams],
        token_histories: list[list[int]],
    ) -> Tensor:
        """Record the decode batch and return the next token batch."""
        history_copy = [list(history) for history in token_histories]
        self.decode_calls.append(
            (input_ids.clone(), list(seq_ids), list(params), history_copy)
        )
        return self.decode_batches.pop(0).clone()


def make_kv_manager() -> KVCacheManager:
    """Create a KVCacheManager for Phase 2 engine tests."""
    return KVCacheManager(
        num_blocks=TOTAL_BLOCKS,
        block_size=BLOCK_SIZE,
        num_kv_heads=2,
        head_dim=8,
        n_layers=1,
    )


def make_scheduler(kv_manager: KVCacheManager) -> Scheduler:
    """Create a Scheduler for Phase 2 engine tests."""
    return Scheduler(
        max_concurrent=2,
        block_size=BLOCK_SIZE,
        total_blocks=kv_manager.allocator.num_free_blocks(),
    )


def make_sequence(
    seq_id: int,
    prompt_tokens: list[int],
    max_new_tokens: int,
    *,
    sampling_params: SamplingParams | None = None,
) -> Sequence:
    """Build a sequence object with the desired sampling parameters."""
    return Sequence(
        seq_id=seq_id,
        status=SequenceStatus.WAITING,
        prompt_tokens=prompt_tokens,
        output_tokens=[],
        max_new_tokens=max_new_tokens,
        sampling_params=sampling_params or SamplingParams(),
    )


def test_single_sequence_prefill_then_decode() -> None:
    """Engine should prefill once, decode until max_new_tokens, then finish."""
    kv_manager = make_kv_manager()
    scheduler = make_scheduler(kv_manager)
    params = SamplingParams(temperature=0.7, top_k=50)
    runner = cast(
        "ModelRunner",
        MockModelRunner(
            kv_manager,
            prefill_tokens={0: 5},
            decode_batches=[torch.tensor([6]), torch.tensor([7])],
        ),
    )
    engine = ContinuousBatchingEngine(runner, scheduler, eos_token_id=EOS_TOKEN_ID)
    requests = deque(
        [(0, make_sequence(0, [1, 2, 3], max_new_tokens=3, sampling_params=params))]
    )

    finished = engine.run(requests)
    runner_impl = cast("MockModelRunner", runner)
    _, decode_seq_ids, decode_params, token_histories = runner_impl.decode_calls[0]

    assert len(finished) == 1
    assert finished[0].output_tokens == [5, 6, 7]
    assert runner_impl.prefill_calls[0][1] == 0
    assert runner_impl.prefill_calls[0][2] is params
    assert decode_seq_ids == [0]
    assert decode_params == [params]
    assert token_histories == [[1, 2, 3, 5]]
    assert not kv_manager.seq_to_block


def test_eos_from_prefill_stops_without_decode() -> None:
    """If prefill emits EOS, the engine should not schedule decode."""
    kv_manager = make_kv_manager()
    scheduler = make_scheduler(kv_manager)
    runner = cast(
        "ModelRunner",
        MockModelRunner(kv_manager, prefill_tokens={0: EOS_TOKEN_ID}),
    )
    engine = ContinuousBatchingEngine(runner, scheduler, eos_token_id=EOS_TOKEN_ID)
    requests = deque([(0, make_sequence(0, [1, 2, 3], max_new_tokens=5))])

    finished = engine.run(requests)
    runner_impl = cast("MockModelRunner", runner)

    assert finished[0].output_tokens == [EOS_TOKEN_ID]
    assert runner_impl.decode_calls == []


def test_decode_batch_uses_per_sequence_sampling_params() -> None:
    """Decode batches should preserve seq_ids, histories, and params order."""
    kv_manager = make_kv_manager()
    scheduler = make_scheduler(kv_manager)
    params_0 = SamplingParams(temperature=0.0)
    params_1 = SamplingParams(temperature=0.8, top_p=0.9)
    runner = cast(
        "ModelRunner",
        MockModelRunner(
            kv_manager,
            prefill_tokens={0: 10, 1: 20},
            decode_batches=[torch.tensor([11, 21])],
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
    assert finished[0].output_tokens == [10, 11]
    assert finished[1].output_tokens == [20, 21]
    torch.testing.assert_close(decode_input_ids, torch.tensor([[10], [20]]))
    assert decode_seq_ids == [0, 1]
    assert decode_params == [params_0, params_1]
    assert token_histories == [[1, 2, 3, 10], [4, 5, 20]]
