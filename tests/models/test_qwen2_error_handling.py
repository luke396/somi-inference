"""Tests for error handling in Qwen2 components."""

import pytest
import torch
from somi_inference.models.qwen2 import (
    QwenAttention,
    QwenDecoderLayer,
    QwenModel,
    causal_attention,
    ForwardContext,
    ForwardMode,
)


class TestCausalAttentionErrors:
    def test_mismatched_seq_len(self):
        """Mismatched Q/K/V seq_len should raise error."""
        q = torch.randn(1, 4, 5, 32)  # seq_len=5
        k = torch.randn(1, 4, 3, 32)  # seq_len=3
        v = torch.randn(1, 4, 3, 32)
        with pytest.raises((RuntimeError, ValueError)):
            causal_attention(q, k, v)

    def test_mismatched_head_dim(self):
        """Mismatched head_dim should raise error."""
        q = torch.randn(1, 4, 5, 32)
        k = torch.randn(1, 4, 5, 64)  # Different head_dim
        v = torch.randn(1, 4, 5, 32)
        with pytest.raises((RuntimeError, ValueError)):
            causal_attention(q, k, v)

    def test_invalid_gqa_ratio(self):
        """Non-divisible GQA ratio should raise ValueError."""
        q = torch.randn(1, 7, 4, 32)  # 7 not divisible by 3
        k = torch.randn(1, 3, 4, 32)
        v = torch.randn(1, 3, 4, 32)
        with pytest.raises(ValueError, match="divisible"):
            causal_attention(q, k, v)


class TestQwenModelErrors:
    def test_invalid_vocab_size(self):
        """Token IDs >= vocab_size should raise error."""
        model = QwenModel(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            max_seq_size=512,
        )

        ctx = ForwardContext(
            mode=ForwardMode.PREFILL,
            attn_fn=lambda q, k, v, layer_idx: causal_attention(q, k, v),
            posi_idx=torch.arange(5).unsqueeze(0),
        )

        input_ids = torch.tensor([[0, 1, 150]])  # 150 >= vocab_size
        with pytest.raises((IndexError, RuntimeError)):
            model(input_ids, ctx)

    def test_negative_token_ids(self):
        """Negative token IDs should raise error."""
        model = QwenModel(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            max_seq_size=512,
        )

        ctx = ForwardContext(
            mode=ForwardMode.PREFILL,
            attn_fn=lambda q, k, v, layer_idx: causal_attention(q, k, v),
            posi_idx=torch.arange(3).unsqueeze(0),
        )

        input_ids = torch.tensor([[0, -1, 2]])
        with pytest.raises((IndexError, RuntimeError)):
            model(input_ids, ctx)
