"""Tests for QwenDecoderLayer."""

import torch
from somi_inference.models.qwen2 import QwenDecoderLayer, RotaryEmbedding

# Shared small config for tests
SMALL_CONFIG = dict(
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=32,
)


def _make_rope_inputs(seq_len: int, batch: int = 1):
    """Create RotaryEmbedding and compute cos/sin for given sequence length."""
    rope = RotaryEmbedding(hid_dim=32, max_seq_len=512)
    posi_idx = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    cos, sin = rope(posi_idx)
    return cos, sin


class TestQwenDecoderLayer:
    def test_forward_shape(self):
        """Output shape should match input shape."""
        layer = QwenDecoderLayer(
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=32,
            rms_norm_eps=1e-6,
        )
        cos, sin = _make_rope_inputs(seq_len=10, batch=2)

        x = torch.randn(2, 10, 128)
        out = layer(x, cos, sin)
        assert out.shape == (2, 10, 128)

    def test_residual_connections(self):
        """Verify residual connections preserve information."""
        torch.manual_seed(42)
        layer = QwenDecoderLayer(**SMALL_CONFIG)
        cos, sin = _make_rope_inputs(seq_len=5)

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
        out = layer(x, cos, sin)

        # With zero weights, attn and mlp output zero -- pure residual
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_gradient_flow(self):
        """Verify gradients flow through all components."""
        layer = QwenDecoderLayer(**SMALL_CONFIG)
        cos, sin = _make_rope_inputs(seq_len=5)

        x = torch.randn(1, 5, 64, requires_grad=True)
        out = layer(x, cos, sin)
        out.sum().backward()

        assert layer.self_attn.q_proj.weight.grad is not None
        assert layer.mlp.gate_proj.weight.grad is not None
        assert layer.input_layernorm.weight.grad is not None
        assert x.grad is not None
