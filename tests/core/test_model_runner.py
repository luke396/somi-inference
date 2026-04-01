"""Tests for ModelRunner coordination logic."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from somi_inference.core.model_runner import ModelRunner
from somi_inference.core.paged_attention import KVCacheManager
from somi_inference.core.sampler import Sampler, SamplingParams

PREFILL_SEQ_ID = 3
PREFILL_SAMPLED_TOKEN = 17


class MockModelAdapter:
    """Mock adapter that returns predefined logits and records calls."""

    def __init__(self, prefill_logits: Tensor, decode_logits: Tensor) -> None:
        """Initialize the mock with predefined prefill/decode outputs."""
        self.prefill_logits = prefill_logits
        self.decode_logits = decode_logits
        self.prefill_calls: list[tuple[Tensor, KVCacheManager, int]] = []
        self.decode_calls: list[tuple[Tensor, KVCacheManager, list[int]]] = []

    def prefill(
        self,
        input_ids: Tensor,
        kv_manager: KVCacheManager,
        seq_id: int,
    ) -> Tensor:
        """Record prefill inputs and return predefined logits."""
        self.prefill_calls.append((input_ids.clone(), kv_manager, seq_id))
        return self.prefill_logits.clone()

    def decode(
        self,
        input_ids: Tensor,
        kv_manager: KVCacheManager,
        seq_ids: list[int],
    ) -> Tensor:
        """Record decode inputs and return predefined logits."""
        self.decode_calls.append((input_ids.clone(), kv_manager, list(seq_ids)))
        return self.decode_logits.clone()


class MockSampler(Sampler):
    """Mock sampler that records inputs and returns predefined tokens."""

    def __init__(self, sampled_tokens: Tensor) -> None:
        """Initialize the mock with a fixed sampling result."""
        self.sampled_tokens = sampled_tokens
        self.calls: list[
            tuple[Tensor, SamplingParams | list[SamplingParams], list[list[int]] | None]
        ] = []

    def sample(
        self,
        logits: Tensor,
        params: SamplingParams | list[SamplingParams],
        token_history: list[list[int]] | None = None,
    ) -> Tensor:
        """Record sampler inputs and return predefined tokens."""
        history_copy = None
        if token_history is not None:
            history_copy = [list(history) for history in token_history]

        self.calls.append((logits.clone(), params, history_copy))
        return self.sampled_tokens.clone()


@pytest.fixture
def kv_manager() -> KVCacheManager:
    """Create a minimal KV cache manager for tests."""
    return KVCacheManager(
        num_blocks=8,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        n_layers=1,
    )


def test_model_runner_prefill_samples_last_prompt_position(
    kv_manager: KVCacheManager,
) -> None:
    """Prefill should sample from the last prompt position."""
    prefill_logits = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
            ]
        ]
    )
    adapter = MockModelAdapter(
        prefill_logits=prefill_logits,
        decode_logits=torch.zeros(1, 1, 3),
    )
    sampler = MockSampler(sampled_tokens=torch.tensor([PREFILL_SAMPLED_TOKEN]))
    runner = ModelRunner(adapter, sampler, kv_manager)

    input_ids = torch.tensor([[11, 12, 13]])
    params = SamplingParams(temperature=0.0)

    sampled_token = runner.prefill(
        input_ids,
        seq_id=PREFILL_SEQ_ID,
        sampling_params=params,
    )

    assert sampled_token == PREFILL_SAMPLED_TOKEN
    assert len(adapter.prefill_calls) == 1
    adapter_input_ids, adapter_kv_manager, adapter_seq_id = adapter.prefill_calls[0]
    torch.testing.assert_close(adapter_input_ids, input_ids)
    assert adapter_kv_manager is kv_manager
    assert adapter_seq_id == PREFILL_SEQ_ID

    assert len(sampler.calls) == 1
    sampler_logits, sampler_params, token_history = sampler.calls[0]
    torch.testing.assert_close(sampler_logits, prefill_logits[:, -1, :])
    assert sampler_params is params
    assert token_history == [[11, 12, 13]]


def test_model_runner_decode_samples_batch_and_forwards_histories(
    kv_manager: KVCacheManager,
) -> None:
    """Decode should sample from the single decode step for the full batch."""
    decode_logits = torch.tensor(
        [
            [[10.0, 11.0, 12.0]],
            [[20.0, 21.0, 22.0]],
        ]
    )
    adapter = MockModelAdapter(
        prefill_logits=torch.zeros(1, 1, 3),
        decode_logits=decode_logits,
    )
    sampler = MockSampler(sampled_tokens=torch.tensor([5, 6]))
    runner = ModelRunner(adapter, sampler, kv_manager)

    input_ids = torch.tensor([[101], [202]])
    seq_ids = [7, 8]
    sampling_params = [
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=1.0, top_k=2),
    ]
    token_histories = [[1, 2, 101], [3, 4, 202]]

    sampled_tokens = runner.decode(
        input_ids=input_ids,
        seq_ids=seq_ids,
        sampling_params=sampling_params,
        token_histories=token_histories,
    )

    torch.testing.assert_close(sampled_tokens, torch.tensor([5, 6]))
    assert len(adapter.decode_calls) == 1
    adapter_input_ids, adapter_kv_manager, adapter_seq_ids = adapter.decode_calls[0]
    torch.testing.assert_close(adapter_input_ids, input_ids)
    assert adapter_kv_manager is kv_manager
    assert adapter_seq_ids == seq_ids

    assert len(sampler.calls) == 1
    sampler_logits, sampler_params, sampler_history = sampler.calls[0]
    torch.testing.assert_close(sampler_logits, decode_logits[:, 0, :])
    assert sampler_params == sampling_params
    assert sampler_history == token_histories
