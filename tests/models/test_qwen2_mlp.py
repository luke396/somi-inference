"""Tests for QwenMLP."""

import pytest
import torch
from somi_inference.core.mlp_triton import TRITON_AVAILABLE
from somi_inference.models.qwen2 import QwenMLP


class TestQwenMLP:
    def test_forward_shape(self):
        """Output shape should be (batch, seq_len, hidden_size)."""
        mlp = QwenMLP(hidden_size=128, intermediate_size=256)
        x = torch.randn(2, 10, 128)
        out = mlp(x)
        assert out.shape == (2, 10, 128)

    def test_gate_up_proj_shape(self):
        """Merged gate/up projection should double the intermediate width."""
        mlp = QwenMLP(hidden_size=128, intermediate_size=256)
        assert mlp.gate_up_proj.weight.shape == (512, 128)

    def test_swiglu_activation(self):
        """Verify SwiGLU formula: down(silu(gate(x)) * up(x))."""
        mlp = QwenMLP(hidden_size=4, intermediate_size=8)
        with torch.no_grad():
            mlp.gate_up_proj.weight.fill_(1.0)
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

    def test_explicit_triton_backend_rejects_cpu_inputs(self):
        """Forcing Triton on CPU should raise a runtime error."""
        mlp = QwenMLP(hidden_size=64, intermediate_size=128, backend="triton")
        x = torch.randn(1, 5, 64)

        with pytest.raises(RuntimeError, match="not available"):
            mlp(x)

    @pytest.mark.skipif(
        not torch.cuda.is_available() or not TRITON_AVAILABLE,
        reason="CUDA + Triton required for Triton MLP parity",
    )
    def test_triton_backend_matches_torch_ref(self):
        """Supported CUDA inputs should match the torch reference path."""
        torch.manual_seed(42)
        ref_mlp = QwenMLP(
            hidden_size=64,
            intermediate_size=128,
            backend="torch_ref",
        ).cuda().half()
        triton_mlp = QwenMLP(
            hidden_size=64,
            intermediate_size=128,
            backend="triton",
        ).cuda().half()
        triton_mlp.load_state_dict(ref_mlp.state_dict())
        x = torch.randn(1, 65, 64, device="cuda", dtype=torch.float16)

        with torch.inference_mode():
            expected = ref_mlp(x)
            actual = triton_mlp(x)

        torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)
