"""Tests for QwenDecoderLayer."""

import torch
from somi_inference.models.qwen2 import (
    ForwardContext,
    ForwardMode,
    QwenDecoderLayer,
    causal_attention,
)


class TestQwenDecoderLayer:
    def test_forward_shape(self, make_rope_inputs, make_forward_context):
        """Output shape should match input shape."""
        layer = QwenDecoderLayer(
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=32,
            layer_idx=0,
            rms_norm_eps=1e-6,
        )
        cos, sin = make_rope_inputs(head_dim=32, seq_len=10, batch=2)
        ctx = make_forward_context(seq_len=10, batch=2)

        x = torch.randn(2, 10, 128)
        out = layer(x, cos, sin, ctx)
        assert out.shape == (2, 10, 128)

    def test_residual_connections(self, make_rope_inputs, make_forward_context):
        """Verify residual connections preserve information."""
        torch.manual_seed(42)
        layer = QwenDecoderLayer(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            layer_idx=0,
        )
        cos, sin = make_rope_inputs(head_dim=32, seq_len=5)
        ctx = make_forward_context(seq_len=5)

        # Zero out all weights so attn/mlp contribute nothing
        with torch.no_grad():
            for proj in [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ]:
                proj.weight.zero_()
                proj.bias.zero_()
            layer.self_attn.o_proj.weight.zero_()
            layer.mlp.gate_proj.weight.zero_()
            layer.mlp.up_proj.weight.zero_()
            layer.mlp.down_proj.weight.zero_()

        x = torch.randn(1, 5, 64)
        out = layer(x, cos, sin, ctx)

        # With zero weights, attn and mlp output zero -- pure residual
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_gradient_flow(self, make_rope_inputs, make_forward_context):
        """Verify gradients flow through all components."""
        layer = QwenDecoderLayer(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            layer_idx=0,
        )
        cos, sin = make_rope_inputs(head_dim=32, seq_len=5)
        ctx = make_forward_context(seq_len=5)

        x = torch.randn(1, 5, 64, requires_grad=True)
        out = layer(x, cos, sin, ctx)
        out.sum().backward()

        assert layer.self_attn.q_proj.weight.grad is not None
        assert layer.mlp.gate_proj.weight.grad is not None
        assert layer.input_layernorm.weight.grad is not None
        assert x.grad is not None

    def test_forward_with_custom_attn_fn(self, make_rope_inputs):
        """forward() works with a custom attn_fn (simulating decode mode)."""
        layer = QwenDecoderLayer(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=32,
            intermediate_size=256,
            layer_idx=0,
        )
        cos, sin = make_rope_inputs(head_dim=32, seq_len=1, batch=2)

        def identity_attn(q, k, v, layer_idx):
            return q

        ctx = ForwardContext(
            mode=ForwardMode.DECODE,
            attn_fn=identity_attn,
            posi_idx=torch.tensor([[0], [0]]),
        )

        x = torch.randn(2, 1, 128)
        out = layer(x, cos, sin, ctx)
        assert out.shape == (2, 1, 128)
