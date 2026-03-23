"""Phase 2 tests for continuous batching with sampling params and ModelRunner."""

from collections import deque

import torch

from somi_inference.core.continuous_batching import (
    ContinuousBatchingEngine,
    Scheduler,
    Sequence,
    SequenceStatus,
)
from somi_inference.core.paged_attention import KVCacheManager
from somi_inference.core.sampler import SamplingParams


class MockModelRunner:
    """ModelRunner stub that returns pre-baked token outputs."""

    def __init__(
        self,
        prefill_tokens: dict[int, int],
        decode_batches: list[torch.Tensor] | None = None,
    ) -> None:
        self.prefill_tokens = prefill_tokens
        self.decode_batches = list(decode_batches or [])
        self.prefill_calls = []
        self.decode_calls = []

    def prefill(self, input_ids, seq_id, params):
        """Record the prefill request and return a fixed token."""
        self.prefill_calls.append(
            {
                "input_ids": input_ids.clone(),
                "seq_id": seq_id,
                "params": params,
            }
        )
        return self.prefill_tokens[seq_id]

    def decode(self, input_ids, seq_ids, params):
        """Record the decode batch and return the next token batch."""
        self.decode_calls.append(
            {
                "input_ids": input_ids.clone(),
                "seq_ids": list(seq_ids),
                "params": list(params),
            }
        )
        return self.decode_batches.pop(0).clone()


def make_kv_manager() -> KVCacheManager:
    """Create a KVCacheManager for Phase 2 engine tests."""
    return KVCacheManager(
        num_blocks=10,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        n_layers=1,
    )


def make_scheduler(kv_manager: KVCacheManager) -> Scheduler:
    """Create a Scheduler for Phase 2 engine tests."""
    return Scheduler(
        max_concurrent=2,
        block_size=4,
        free_block_num_fn=kv_manager.allocator.num_free_blocks,
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


def test_single_sequence_prefill_then_decode():
    """Engine should prefill once, decode until max_new_tokens, then finish."""
    kv_manager = make_kv_manager()
    scheduler = make_scheduler(kv_manager)
    params = SamplingParams(temperature=0.7, top_k=50)
    runner = MockModelRunner(
        prefill_tokens={0: 5},
        decode_batches=[torch.tensor([6]), torch.tensor([7])],
    )
    engine = ContinuousBatchingEngine(runner, kv_manager, scheduler, eos_token_id=2)
    requests = deque(
        [(0, make_sequence(0, [1, 2, 3], max_new_tokens=3, sampling_params=params))]
    )

    finished = engine.run(requests)

    assert len(finished) == 1
    assert finished[0].output_tokens == [5, 6, 7]
    assert runner.prefill_calls[0]["seq_id"] == 0
    assert runner.prefill_calls[0]["params"] is params
    torch.testing.assert_close(
        runner.decode_calls[0]["input_ids"],
        torch.tensor([[5]]),
    )
    assert runner.decode_calls[0]["seq_ids"] == [0]
    assert runner.decode_calls[0]["params"] == [params]
    assert not kv_manager.seq_to_block


def test_eos_from_prefill_stops_without_decode():
    """If prefill emits EOS, the engine should not schedule decode."""
    kv_manager = make_kv_manager()
    scheduler = make_scheduler(kv_manager)
    runner = MockModelRunner(prefill_tokens={0: 2})
    engine = ContinuousBatchingEngine(runner, kv_manager, scheduler, eos_token_id=2)
    requests = deque([(0, make_sequence(0, [1, 2, 3], max_new_tokens=5))])

    finished = engine.run(requests)

    assert finished[0].output_tokens == [2]
    assert runner.decode_calls == []


def test_decode_batch_uses_per_sequence_sampling_params():
    """Decode batches should preserve seq_ids, previous tokens, and params order."""
    kv_manager = make_kv_manager()
    scheduler = make_scheduler(kv_manager)
    params_0 = SamplingParams(temperature=0.0)
    params_1 = SamplingParams(temperature=0.8, top_p=0.9)
    runner = MockModelRunner(
        prefill_tokens={0: 10, 1: 20},
        decode_batches=[torch.tensor([11, 21])],
    )
    engine = ContinuousBatchingEngine(runner, kv_manager, scheduler, eos_token_id=2)
    requests = deque(
        [
            (0, make_sequence(0, [1, 2, 3], max_new_tokens=2, sampling_params=params_0)),
            (0, make_sequence(1, [4, 5], max_new_tokens=2, sampling_params=params_1)),
        ]
    )

    finished = engine.run(requests)

    assert [seq.seq_id for seq in finished] == [0, 1]
    assert finished[0].output_tokens == [10, 11]
    assert finished[1].output_tokens == [20, 21]

    decode_call = runner.decode_calls[0]
    torch.testing.assert_close(
        decode_call["input_ids"],
        torch.tensor([[10], [20]]),
    )
    assert decode_call["seq_ids"] == [0, 1]
    assert decode_call["params"] == [params_0, params_1]
