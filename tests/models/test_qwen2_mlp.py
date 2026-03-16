"""Tests for QwenMLP."""

import torch
from somi_inference.models.qwen2 import QwenMLP


class TestQwenMLP:
    def test_forward_shape(self):
        """Output shape should be (batch, seq_len, hidden_size)."""
        mlp = QwenMLP(hidden_size=128, intermediate_size=256)
        x = torch.randn(2, 10, 128)
        out = mlp(x)
        assert out.shape == (2, 10, 128)

    def test_swiglu_activation(self):
        """Verify SwiGLU formula: down(silu(gate(x)) * up(x))."""
        mlp = QwenMLP(hidden_size=4, intermediate_size=8)
        with torch.no_grad():
            mlp.gate_proj.weight.fill_(1.0)
            mlp.up_proj.weight.fill_(1.0)
            mlp.down_proj.weight.fill_(1.0)

        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # (1, 1, 4)
        out = mlp(x)

        # Manual computation
        gate_out = torch.nn.functional.silu(
            x.sum(dim=-1, keepdim=True).expand(-1, -1, 8)
        )
        up_out = x.sum(dim=-1, keepdim=True).expand(-1, -1, 8)
        expected = (gate_out * up_out).sum(dim=-1, keepdim=True).expand(-1, -1, 4)

        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_dtype_preserved(self):
        """Output dtype should match input dtype."""
        mlp = QwenMLP(hidden_size=64, intermediate_size=128).half()
        x = torch.randn(1, 5, 64, dtype=torch.float16)
        out = mlp(x)
        assert out.dtype == torch.float16
