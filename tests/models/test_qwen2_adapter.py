"""Tests for QwenAdapter."""

import torch

from somi_inference.core.paged_attention import KVCacheManager
from somi_inference.models.qwen2 import QwenModel
from somi_inference.models.qwen2_adapter import QwenAdapter

ADAPTER_CONFIG = dict(
    vocab_size=100,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_seq_size=512,
)


def _make_kv_manager():
    return KVCacheManager(
        num_blocks=20,
        block_size=4,
        num_kv_heads=2,
        head_dim=16,
        n_layers=2,
    )


class TestQwenAdapterPrefill:
    def test_prefill_logits_shape(self):
        """prefill returns (batch_size, seq_len, vocab_size) logits."""
        model = QwenModel(**ADAPTER_CONFIG)
        adapter = QwenAdapter(model)
        kv = _make_kv_manager()
        kv.register_sequence(0)

        tokens = torch.randint(0, 100, (1, 6))
        logits = adapter.prefill(tokens, kv, seq_id=0)
        assert logits.shape == (1, 6, 100)

    def test_prefill_writes_kv_cache(self):
        """prefill writes KV for all layers and advances token count."""
        model = QwenModel(**ADAPTER_CONFIG)
        adapter = QwenAdapter(model)
        kv = _make_kv_manager()
        kv.register_sequence(0)

        tokens = torch.randint(0, 100, (1, 5))
        adapter.prefill(tokens, kv, seq_id=0)

        assert kv.get_num_tokens(0) == 5
        assert len(kv.get_block_ids(0)) == 2  # 5 tokens / block_size=4 → 2 blocks


class TestQwenAdapterDecode:
    def test_decode_logits_shape(self):
        """decode returns (batch, 1, vocab_size) logits."""
        model = QwenModel(**ADAPTER_CONFIG)
        adapter = QwenAdapter(model)
        kv = _make_kv_manager()

        # Prefill first to populate cache
        kv.register_sequence(0)
        adapter.prefill(torch.randint(0, 100, (1, 4)), kv, seq_id=0)

        # Decode one token
        input_ids = torch.randint(0, 100, (1, 1))
        logits = adapter.decode(input_ids, kv, seq_ids=[0])
        assert logits.shape == (1, 1, 100)

    def test_decode_advances_cache(self):
        """decode writes 1 new KV token per sequence."""
        model = QwenModel(**ADAPTER_CONFIG)
        adapter = QwenAdapter(model)
        kv = _make_kv_manager()

        kv.register_sequence(0)
        adapter.prefill(torch.randint(0, 100, (1, 3)), kv, seq_id=0)
        assert kv.get_num_tokens(0) == 3

        adapter.decode(torch.randint(0, 100, (1, 1)), kv, seq_ids=[0])
        assert kv.get_num_tokens(0) == 4

    def test_decode_batch(self):
        """decode handles multiple sequences in a batch."""
        model = QwenModel(**ADAPTER_CONFIG)
        adapter = QwenAdapter(model)
        kv = _make_kv_manager()

        # Prefill two sequences
        for sid in [0, 1]:
            kv.register_sequence(sid)
            adapter.prefill(torch.randint(0, 100, (1, 3)), kv, seq_id=sid)

        # Batch decode
        input_ids = torch.randint(0, 100, (2, 1))
        logits = adapter.decode(input_ids, kv, seq_ids=[0, 1])
        assert logits.shape == (2, 1, 100)


class TestQwenAdapterConsistency:
    def test_prefill_then_decode_produces_valid_logits(self):
        """Full prefill→decode flow produces finite, non-zero logits."""
        torch.manual_seed(42)
        model = QwenModel(**ADAPTER_CONFIG)
        adapter = QwenAdapter(model)
        kv = _make_kv_manager()
        kv.register_sequence(0)

        # Prefill
        prompt = torch.randint(0, 100, (1, 6))
        prefill_logits = adapter.prefill(prompt, kv, seq_id=0)
        assert torch.isfinite(prefill_logits).all()
        assert not torch.allclose(prefill_logits, torch.zeros_like(prefill_logits))

        # Decode 3 tokens
        next_token = prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
        for step in range(3):
            logits = adapter.decode(next_token, kv, seq_ids=[0])
            assert torch.isfinite(logits).all()
            assert logits.shape == (1, 1, 100)
            next_token = logits[:, 0, :].argmax(dim=-1, keepdim=True)

        assert kv.get_num_tokens(0) == 9  # 6 prefill + 3 decode
