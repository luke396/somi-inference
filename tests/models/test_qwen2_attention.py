"""Tests for QwenAttention."""

import torch
from somi_inference.models.qwen2 import QwenAttention


class TestQwenAttention:
    def test_forward_shape_mha(self, make_rope_inputs, make_forward_context):
        """Output shape should match input for MHA."""
        attn = QwenAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=32,
            layer_idx=0,
        )
        cos, sin = make_rope_inputs(head_dim=32, seq_len=10, batch=2)
        ctx = make_forward_context(seq_len=10, batch=2)

        x = torch.randn(2, 10, 128)
        out = attn(x, cos, sin, ctx)
        assert out.shape == (2, 10, 128)

    def test_causal_masking(self, make_rope_inputs, make_forward_context):
        """Verify attention is causal (no future leakage)."""
        torch.manual_seed(42)
        attn = QwenAttention(
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            layer_idx=0,
        )
        cos, sin = make_rope_inputs(head_dim=32, seq_len=4)
        ctx = make_forward_context(seq_len=4)

        x = torch.randn(1, 4, 64)
        out = attn(x, cos, sin, ctx)

        # Modify future tokens and verify first token output unchanged
        x_modified = x.clone()
        x_modified[0, 1:] = torch.randn(3, 64)
        out_modified = attn(x_modified, cos, sin, ctx)

        torch.testing.assert_close(out[0, 0], out_modified[0, 0], atol=1e-5, rtol=1e-5)

    def test_project_qkv_shapes(self, make_rope_inputs):
        """project_qkv returns (q, k, v) with correct shapes."""
        attn = QwenAttention(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            layer_idx=0,
        )
        cos, sin = make_rope_inputs(head_dim=16, seq_len=5, batch=2)

        x = torch.randn(2, 5, 64)
        q, k, v = attn._project_qkv(x, cos, sin)
        assert q.shape == (2, 4, 5, 16)   # (batch, num_q_heads, seq_len, head_dim)
        assert k.shape == (2, 2, 5, 16)   # (batch, num_kv_heads, seq_len, head_dim)
        assert v.shape == (2, 2, 5, 16)
