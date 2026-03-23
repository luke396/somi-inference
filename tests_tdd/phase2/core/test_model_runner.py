"""Tests for ModelRunner."""

import torch

from somi_inference.core.model_runner import ModelRunner
from somi_inference.core.paged_attention import KVCacheManager
from somi_inference.core.sampler import SamplingParams


class MockAdapter:
    """Adapter stub that returns deterministic logits."""

    def __init__(self) -> None:
        self.prefill_calls = []
        self.decode_calls = []
        self.prefill_logits = torch.tensor(
            [
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                ]
            ]
        )
        self.decode_logits = torch.tensor(
            [
                [[1.0, 5.0, 0.0, 0.0]],
                [[0.0, 0.0, 2.0, 9.0]],
            ]
        )

    def prefill(self, input_ids, kv_manager, seq_id):
        """Return full-sequence logits for prefill."""
        self.prefill_calls.append(
            {
                "input_ids": input_ids.clone(),
                "kv_manager": kv_manager,
                "seq_id": seq_id,
            }
        )
        return self.prefill_logits.clone()

    def decode(self, input_ids, kv_manager, seq_ids):
        """Return one-step logits for batched decode."""
        self.decode_calls.append(
            {
                "input_ids": input_ids.clone(),
                "kv_manager": kv_manager,
                "seq_ids": list(seq_ids),
            }
        )
        return self.decode_logits[: input_ids.size(0)].clone()


class MockSampler:
    """Sampler stub that records its inputs."""

    def __init__(self, return_tokens: torch.Tensor) -> None:
        self.return_tokens = return_tokens
        self.calls = []

    def sample(self, logits, params, input_ids=None):
        """Record logits/params and return fixed tokens."""
        self.calls.append(
            {
                "logits": logits.clone(),
                "params": params,
                "input_ids": input_ids,
            }
        )
        return self.return_tokens.clone()


def make_kv_manager() -> KVCacheManager:
    """Create a minimal KV cache manager for runner tests."""
    return KVCacheManager(
        num_blocks=8,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        n_layers=1,
    )


def test_model_runner_prefill_samples_last_position_logits():
    """prefill should sample from the last prompt position and return an int."""
    adapter = MockAdapter()
    sampler = MockSampler(return_tokens=torch.tensor([2]))
    kv_manager = make_kv_manager()
    kv_manager.register_sequence(0)
    runner = ModelRunner(adapter, sampler, kv_manager)

    params = SamplingParams(temperature=0.0)
    token = runner.prefill(torch.tensor([[1, 2, 3]]), seq_id=0, params=params)

    assert token == 2
    assert isinstance(token, int)
    assert adapter.prefill_calls[0]["seq_id"] == 0
    torch.testing.assert_close(
        sampler.calls[0]["logits"],
        adapter.prefill_logits[:, -1, :],
    )
    assert sampler.calls[0]["params"] is params


def test_model_runner_decode_samples_batched_last_step_logits():
    """decode should sample one token per sequence from the decode logits."""
    adapter = MockAdapter()
    sampler = MockSampler(return_tokens=torch.tensor([1, 3]))
    kv_manager = make_kv_manager()
    runner = ModelRunner(adapter, sampler, kv_manager)

    params = [
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=0.8, top_k=10),
    ]
    tokens = runner.decode(
        input_ids=torch.tensor([[11], [22]]),
        seq_ids=[10, 11],
        params=params,
    )

    torch.testing.assert_close(tokens, torch.tensor([1, 3]))
    assert adapter.decode_calls[0]["seq_ids"] == [10, 11]
    torch.testing.assert_close(
        sampler.calls[0]["logits"],
        adapter.decode_logits[:, 0, :],
    )
    assert sampler.calls[0]["params"] == params
