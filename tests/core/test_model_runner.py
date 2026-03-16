"""Tests for ModelRunner."""

import pytest
import torch
from somi_inference.core.model_runner import ModelRunner
from somi_inference.core.sampler import Sampler, SamplingParams
from somi_inference.core.paged_attention import KVCacheManager
from somi_inference.models.qwen2_adapter import load_from_hf


@pytest.fixture
def qwen_config():
    """Qwen2.5-0.5B config."""
    return {
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "hidden_size": 896,
    }


@pytest.fixture
def model_runner(qwen_config):
    """Create ModelRunner with Qwen adapter."""
    adapter = load_from_hf("Qwen/Qwen2.5-0.5B")
    sampler = Sampler()
    kv_manager = KVCacheManager(
        num_layers=qwen_config["num_hidden_layers"],
        num_kv_heads=qwen_config["num_key_value_heads"],
        head_dim=qwen_config["hidden_size"] // qwen_config["num_attention_heads"],
        block_size=16,
        num_blocks=128,
        dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return ModelRunner(adapter, sampler, kv_manager)


# ============================================================================
# Task 8: ModelRunner Prefill Tests
# ============================================================================


@pytest.mark.slow
def test_model_runner_prefill_greedy(model_runner):
    """ModelRunner.prefill with greedy sampling"""
    input_ids = torch.tensor([[1, 2, 3, 4]])
    seq_id = 0
    params = SamplingParams(temperature=0.0)

    model_runner.kv_manager.register_sequence(seq_id)
    token = model_runner.prefill(input_ids, seq_id, params)

    assert isinstance(token, int)
    assert token >= 0


# ============================================================================
# Task 9: ModelRunner Decode Tests
# ============================================================================


@pytest.mark.slow
def test_model_runner_decode_batch(model_runner):
    """ModelRunner.decode with batch"""
    # Prefill first
    input_ids = torch.tensor([[1, 2, 3]])
    seq_id_0 = 0
    model_runner.kv_manager.register_sequence(seq_id_0)
    token_0 = model_runner.prefill(input_ids, seq_id_0, SamplingParams(temperature=0.0))

    seq_id_1 = 1
    model_runner.kv_manager.register_sequence(seq_id_1)
    token_1 = model_runner.prefill(input_ids, seq_id_1, SamplingParams(temperature=0.0))

    # Decode batch
    decode_input = torch.tensor([[token_0], [token_1]])
    seq_ids = [seq_id_0, seq_id_1]
    params = [
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=1.0),
    ]

    tokens = model_runner.decode(decode_input, seq_ids, params)

    assert tokens.shape == (2,)
    assert tokens[0].item() >= 0
    assert tokens[1].item() >= 0
