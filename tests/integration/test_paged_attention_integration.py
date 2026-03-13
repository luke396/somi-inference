"""Integration tests for paged attention with QwenAdapter."""

import pytest
import torch

from somi_inference.core.paged_attention import KVCacheManager
from somi_inference.models.qwen2 import QwenModel
from somi_inference.models.qwen2_adapter import QwenAdapter


@pytest.fixture
def small_adapter():
    """Create a small QwenAdapter for testing."""
    model = QwenModel(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_seq_size=512,
    )
    return QwenAdapter(model)


@pytest.fixture
def kv_manager():
    """Create KVCacheManager for testing."""
    return KVCacheManager(
        num_blocks=20,
        block_size=4,
        num_kv_heads=2,
        head_dim=16,
        n_layers=2,
    )


class TestPagedAttentionIntegration:
    def test_prefill_then_decode_uses_paged_attention(self, small_adapter, kv_manager):
        """Verify decode mode uses paged attention correctly."""
        torch.manual_seed(42)
        kv_manager.register_sequence(0)

        # Prefill
        prompt = torch.randint(0, 100, (1, 6))
        prefill_logits = small_adapter.prefill(prompt, kv_manager, seq_id=0)

        # Decode should use paged attention
        next_token = prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decode_logits = small_adapter.decode(next_token, kv_manager, seq_ids=[0])

        # Verify KV cache was used
        assert kv_manager.get_num_tokens(0) == 7  # 6 prefill + 1 decode
        assert decode_logits.shape == (1, 1, 100)
        assert torch.isfinite(decode_logits).all()

    def test_multi_sequence_decode_batch(self, small_adapter, kv_manager):
        """Verify batched decode with multiple sequences."""
        torch.manual_seed(42)

        # Prefill two sequences
        for seq_id in [0, 1]:
            kv_manager.register_sequence(seq_id)
            prompt = torch.randint(0, 100, (1, 4))
            small_adapter.prefill(prompt, kv_manager, seq_id=seq_id)

        # Batch decode
        tokens = torch.randint(0, 100, (2, 1))
        logits = small_adapter.decode(tokens, kv_manager, seq_ids=[0, 1])

        assert logits.shape == (2, 1, 100)
        assert kv_manager.get_num_tokens(0) == 5
        assert kv_manager.get_num_tokens(1) == 5

    def test_decode_with_different_sequence_lengths(self, small_adapter, kv_manager):
        """Verify decode works with sequences of different lengths."""
        torch.manual_seed(42)

        # Prefill sequences with different lengths
        kv_manager.register_sequence(0)
        small_adapter.prefill(torch.randint(0, 100, (1, 3)), kv_manager, seq_id=0)

        kv_manager.register_sequence(1)
        small_adapter.prefill(torch.randint(0, 100, (1, 7)), kv_manager, seq_id=1)

        # Batch decode
        tokens = torch.randint(0, 100, (2, 1))
        logits = small_adapter.decode(tokens, kv_manager, seq_ids=[0, 1])

        assert logits.shape == (2, 1, 100)
        assert kv_manager.get_num_tokens(0) == 4
        assert kv_manager.get_num_tokens(1) == 8
