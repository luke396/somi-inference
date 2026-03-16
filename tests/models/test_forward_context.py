"""Tests for ForwardContext."""

import pytest
import torch

from somi_inference.models.qwen2 import (
    ForwardContext,
    ForwardMode,
    causal_attention,
)


class TestForwardContext:
    def test_prefill_mode_creation(self):
        """Create ForwardContext for prefill mode."""
        ctx = ForwardContext(
            mode=ForwardMode.PREFILL,
            attn_fn=lambda q, k, v, layer_idx: causal_attention(q, k, v),
            posi_idx=torch.arange(10).unsqueeze(0),
        )
        assert ctx.mode == ForwardMode.PREFILL
        assert ctx.posi_idx.shape == (1, 10)

    def test_decode_mode_creation(self):
        """Create ForwardContext for decode mode."""
        def mock_paged_attn(q, k, v, layer_idx):
            return q

        ctx = ForwardContext(
            mode=ForwardMode.DECODE,
            attn_fn=mock_paged_attn,
            posi_idx=torch.tensor([[5]]),
        )
        assert ctx.mode == ForwardMode.DECODE
        assert ctx.posi_idx.shape == (1, 1)

    def test_attn_fn_callable(self):
        """attn_fn should be callable with correct signature."""
        def custom_attn(q, k, v, layer_idx):
            return q * 2

        ctx = ForwardContext(
            mode=ForwardMode.PREFILL,
            attn_fn=custom_attn,
            posi_idx=torch.arange(5).unsqueeze(0),
        )

        q = torch.randn(1, 4, 5, 32)
        k = torch.randn(1, 4, 5, 32)
        v = torch.randn(1, 4, 5, 32)

        result = ctx.attn_fn(q, k, v, layer_idx=0)
        torch.testing.assert_close(result, q * 2)

    def test_batch_posi_idx(self):
        """posi_idx supports batch dimension."""
        ctx = ForwardContext(
            mode=ForwardMode.PREFILL,
            attn_fn=lambda q, k, v, layer_idx: causal_attention(q, k, v),
            posi_idx=torch.arange(8).unsqueeze(0).expand(4, -1),  # (4, 8)
        )
        assert ctx.posi_idx.shape == (4, 8)
