"""Tests for QwenAttention."""

import torch
from somi_inference.models.qwen2 import QwenAttention, RotaryEmbedding


def _make_rope_inputs(head_dim: int, seq_len: int, batch: int = 1):
    """Create RotaryEmbedding and compute cos/sin for given sequence length."""
    rope = RotaryEmbedding(hid_dim=head_dim, max_seq_len=512)
    posi_idx = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    cos, sin = rope(posi_idx)
    return cos, sin


class TestQwenAttention:
    def test_forward_shape_mha(self):
        """Output shape should match input for MHA."""
        attn = QwenAttention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=32,
        )
        cos, sin = _make_rope_inputs(head_dim=32, seq_len=10, batch=2)

        x = torch.randn(2, 10, 128)
        out = attn(x, cos, sin)
        assert out.shape == (2, 10, 128)

    def test_forward_shape_gqa(self):
        """GQA: 12 Q heads, 2 KV heads should work."""
        attn = QwenAttention(
            hidden_size=384,
            num_attention_heads=12,
            num_key_value_heads=2,
            head_dim=32,
        )
        cos, sin = _make_rope_inputs(head_dim=32, seq_len=8, batch=2)

        x = torch.randn(2, 8, 384)
        out = attn(x, cos, sin)
        assert out.shape == (2, 8, 384)

    def test_causal_masking(self):
        """Verify attention is causal (no future leakage)."""
        torch.manual_seed(42)
        attn = QwenAttention(
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
        )
        cos, sin = _make_rope_inputs(head_dim=32, seq_len=4)

        x = torch.randn(1, 4, 64)
        out = attn(x, cos, sin)

        # Modify future tokens and verify first token output unchanged
        x_modified = x.clone()
        x_modified[0, 1:] = torch.randn(3, 64)
        out_modified = attn(x_modified, cos, sin)

        torch.testing.assert_close(out[0, 0], out_modified[0, 0], atol=1e-5, rtol=1e-5)
