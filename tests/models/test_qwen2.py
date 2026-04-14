"""Tests for Qwen2 base components: RMSNorm, RotaryEmbedding, causal_attention."""

import math

import pytest
import torch
from somi_inference.models.qwen2 import (
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    causal_attention,
    causal_attention_torch_ref,
    rotate_half,
)


class TestRMSNorm:
    def test_basic(self):
        """Verify against hand-computed result."""
        norm = RMSNorm(hidden_size=4, eps=0.0)
        # x = [1, 2, 3, 4], rms = sqrt(mean([1, 4, 9, 16])) = sqrt(7.5)
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # (1, 1, 4)
        out = norm(x)
        rms = math.sqrt(7.5)
        expected = torch.tensor([[[1 / rms, 2 / rms, 3 / rms, 4 / rms]]])
        torch.testing.assert_close(out, expected)

    def test_weight_scaling(self):
        """Weight parameter should scale the output."""
        norm = RMSNorm(hidden_size=4, eps=0.0)
        norm.weight = torch.nn.Parameter(torch.tensor([2.0, 2.0, 2.0, 2.0]))
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        out = norm(x)
        rms = math.sqrt(7.5)
        expected = torch.tensor([[[2 / rms, 4 / rms, 6 / rms, 8 / rms]]])
        torch.testing.assert_close(out, expected)

    @pytest.mark.parametrize("batch,seq_len,hidden_size", [
        (1, 1, 4),
        (2, 10, 64),
        (4, 20, 128),
    ])
    def test_output_shape(self, batch, seq_len, hidden_size):
        """Output shape should match input shape."""
        norm = RMSNorm(hidden_size=hidden_size)
        x = torch.randn(batch, seq_len, hidden_size)
        out = norm(x)
        assert out.shape == (batch, seq_len, hidden_size)

    def test_dtype_preserved(self):
        """Output dtype should match input dtype."""
        norm = RMSNorm(hidden_size=8).half() # half to cast weight to float16
        x = torch.randn(1, 3, 8, dtype=torch.float16)
        out = norm(x)
        assert out.dtype == torch.float16


class TestRotaryEmbedding:
    def test_cos_sin_shape(self):
        """Verify output shapes for batch position indexing."""
        rope = RotaryEmbedding(hid_dim=128, max_seq_len=512)
        posi_idx = torch.arange(10).unsqueeze(0)  # (1, 10)
        cos, sin = rope(posi_idx)
        assert cos.shape == (1, 10, 128)
        assert sin.shape == (1, 10, 128)

    def test_position_zero(self):
        """At position 0, all freqs are 0 => cos=1, sin=0."""
        rope = RotaryEmbedding(hid_dim=8, max_seq_len=16)
        posi_idx = torch.tensor([[0]])  # (1, 1)
        cos, sin = rope(posi_idx)
        torch.testing.assert_close(cos, torch.ones(1, 1, 8))
        torch.testing.assert_close(sin, torch.zeros(1, 1, 8))

    def test_inv_freq_values(self):
        """Verify the first few inv_freq values for theta=1_000_000."""
        rope = RotaryEmbedding(hid_dim=8, theta_base=1_000_000.0, max_seq_len=16)
        # inv_freq[i] = 1 / (theta ^ (2i / dim))
        # i=0: 1 / (1e6 ^ 0) = 1.0
        # i=1: 1 / (1e6 ^ (2/8)) = 1 / (1e6 ^ 0.25)
        expected_0 = 1.0
        expected_1 = 1.0 / (1_000_000.0 ** 0.25)
        # cos_cached at position 1 should reflect these frequencies
        cos_at_1 = rope.cos_cached[1]  # (head_dim,)
        # First half-dim pair: cos(1 * inv_freq[0]) = cos(1.0)
        assert abs(cos_at_1[0].item() - math.cos(expected_0)) < 1e-5
        # Second half-dim pair: cos(1 * inv_freq[1])
        assert abs(cos_at_1[1].item() - math.cos(expected_1)) < 1e-5

    def test_batch_indexing(self):
        """Different sequences in a batch can have different positions."""
        rope = RotaryEmbedding(hid_dim=8, max_seq_len=64)
        posi_idx = torch.tensor([[0, 1, 2], [10, 11, 12]])  # (2, 3)
        cos, sin = rope(posi_idx)
        assert cos.shape == (2, 3, 8)
        # Verify batch 1 position 0 matches single lookup at position 10
        torch.testing.assert_close(cos[1, 0], rope.cos_cached[10])


class TestRotateHalf:
    def test_basic(self):
        """rotate_half([1,2,3,4]) => [-3,-4,1,2]."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = rotate_half(x)
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        torch.testing.assert_close(out, expected)


class TestApplyRotaryPosEmb:
    def test_magnitude_preserved(self):
        """RoPE is a rotation — vector magnitude should be preserved."""
        torch.manual_seed(42)
        rope = RotaryEmbedding(hid_dim=64, max_seq_len=128)
        q = torch.randn(2, 4, 8, 64)  # (batch, heads, seq_len, head_dim)
        k = torch.randn(2, 4, 8, 64)
        posi_idx = torch.arange(8).unsqueeze(0).expand(2, -1)  # (2, 8)
        cos, sin = rope(posi_idx)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        # Magnitude of each vector should be preserved
        q_norm = q.norm(dim=-1)
        q_rot_norm = q_rot.norm(dim=-1)
        torch.testing.assert_close(q_norm, q_rot_norm, atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        """Output shapes should match input shapes."""
        rope = RotaryEmbedding(hid_dim=32, max_seq_len=64)
        q = torch.randn(1, 8, 5, 32)
        k = torch.randn(1, 2, 5, 32)
        posi_idx = torch.arange(5).unsqueeze(0) # (1, 5)
        cos, sin = rope(posi_idx)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_position_zero_identity(self):
        """At position 0, cos=1 sin=0 => rotation is identity."""
        rope = RotaryEmbedding(hid_dim=8, max_seq_len=16)
        q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]]])  # (1,1,1,8)
        k = q.clone()
        posi_idx = torch.tensor([[0]]) # (1, 1)
        cos, sin = rope(posi_idx)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        torch.testing.assert_close(q_rot, q)
        torch.testing.assert_close(k_rot, k)


class TestCausalAttention:
    def test_no_future_leakage(self):
        """Token at position i should not attend to positions > i."""
        # Use identity-like Q/K so scores reflect position relationships
        seq_len = 4
        head_dim = 2
        q = torch.zeros(1, 1, seq_len, head_dim)
        k = torch.zeros(1, 1, seq_len, head_dim)

        # Make each position have a distinct key
        for i in range(seq_len):
            q[0, 0, i, 0] = 1.0
            k[0, 0, i, 0] = 1.0

        # Adjust v to match head_dim=2
        v = torch.randn(1, 1, seq_len, head_dim)
        out = causal_attention(q, k, v)

        # Position 0 should only attend to position 0
        # With uniform scores on valid positions, out[0] = v[0]
        torch.testing.assert_close(out[0, 0, 0], v[0, 0, 0])

    def test_hand_computed_2x2(self):
        """Verify against hand-computed 2x2 attention."""
        # q = k = [[1, 0], [0, 1]], so scores = I / sqrt(2)
        q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # (1, 1, 2, 2)
        k = q.clone()
        v = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

        out = causal_attention(q, k, v)

        # Position 0: only sees itself, score=1/sqrt(2), softmax=1.0 => out = v[0]
        torch.testing.assert_close(out[0, 0, 0], v[0, 0, 0])

        # Position 1: sees positions 0 and 1
        # score[1,0] = q[1]@k[0] / sqrt(2) = 0
        # score[1,1] = q[1]@k[1] / sqrt(2) = 1/sqrt(2)
        # softmax([0, 1/sqrt(2)]) => known values
        s0 = math.exp(0)
        s1 = math.exp(1 / math.sqrt(2))
        total = s0 + s1
        expected_1 = (s0 / total) * v[0, 0, 0] + (s1 / total) * v[0, 0, 1]
        torch.testing.assert_close(out[0, 0, 1], expected_1, atol=1e-5, rtol=1e-5)

    def test_gqa_support(self):
        """GQA: 12 Q heads, 2 KV heads should work."""
        torch.manual_seed(42)
        batch, seq_len, head_dim = 2, 8, 64
        q = torch.randn(batch, 12, seq_len, head_dim)
        k = torch.randn(batch, 2, seq_len, head_dim)
        v = torch.randn(batch, 2, seq_len, head_dim)
        out = causal_attention(q, k, v)
        assert out.shape == (batch, 12, seq_len, head_dim)

    def test_mha_compatibility(self):
        """MHA: equal Q and KV heads should work without expansion."""
        torch.manual_seed(42)
        batch, num_heads, seq_len, head_dim = 1, 4, 6, 32
        q = torch.randn(batch, num_heads, seq_len, head_dim)
        k = torch.randn(batch, num_heads, seq_len, head_dim)
        v = torch.randn(batch, num_heads, seq_len, head_dim)
        out = causal_attention(q, k, v)
        assert out.shape == (batch, num_heads, seq_len, head_dim)

    def test_invalid_gqa_ratio(self):
        """Non-divisible head counts should raise ValueError."""
        q = torch.randn(1, 5, 4, 8)
        k = torch.randn(1, 3, 4, 8)
        v = torch.randn(1, 3, 4, 8)
        with pytest.raises(ValueError, match="divisible"):
            causal_attention(q, k, v)

    def test_single_token(self):
        """seq_len=1 should work (no masking needed)."""
        q = torch.randn(1, 4, 1, 32)
        k = torch.randn(1, 4, 1, 32)
        v = torch.randn(1, 4, 1, 32)
        out = causal_attention(q, k, v)
        # With seq_len=1, output = v (softmax of single element = 1.0)
        torch.testing.assert_close(out, v)

    @pytest.mark.parametrize(("dtype",), [(torch.float16,), (torch.bfloat16,)])
    def test_reduced_precision_inputs(self, dtype: torch.dtype) -> None:
        """Reduced-precision inputs should not fail in the value projection step."""
        torch.manual_seed(42)
        q = torch.randn(1, 4, 8, 16, dtype=dtype)
        k = torch.randn(1, 2, 8, 16, dtype=dtype)
        v = torch.randn(1, 2, 8, 16, dtype=dtype)

        out = causal_attention(q, k, v)

        assert out.shape == (1, 4, 8, 16)
        assert out.dtype == dtype
        assert torch.isfinite(out.float()).all()

    def test_auto_backend_matches_torch_ref_on_cpu(self):
        """CPU auto-dispatch should fall back to the reference implementation."""
        torch.manual_seed(42)
        q = torch.randn(2, 4, 7, 16)
        k = torch.randn(2, 2, 7, 16)
        v = torch.randn(2, 2, 7, 16)

        expected = causal_attention_torch_ref(q, k, v)
        actual = causal_attention(q, k, v, backend="auto")

        torch.testing.assert_close(actual, expected)

    def test_invalid_backend(self):
        """Unsupported backend names should raise ValueError."""
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)

        with pytest.raises(ValueError, match="Unsupported causal attention backend"):
            causal_attention(q, k, v, backend="bad-backend")  # type: ignore[arg-type]

    def test_triton_backend_rejects_unsupported_cpu_inputs(self):
        """Forcing Triton on unsupported inputs should raise a runtime error."""
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)

        with pytest.raises(RuntimeError, match="not available"):
            causal_attention(q, k, v, backend="triton")

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Triton prefill parity requires CUDA",
    )
    def test_triton_matches_torch_ref(self):
        """Supported CUDA inputs should match the reference path."""
        torch.manual_seed(42)
        device = torch.device("cuda")
        dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        q = torch.randn(1, 4, 65, 64, device=device, dtype=dtype)
        k = torch.randn(1, 2, 65, 64, device=device, dtype=dtype)
        v = torch.randn(1, 2, 65, 64, device=device, dtype=dtype)

        expected = causal_attention_torch_ref(q, k, v)
        actual = causal_attention(q, k, v, backend="triton")

        torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_q_heads,num_kv_heads", [
    (4, 4),   # MHA
    (12, 2),  # GQA (Qwen2.5-1.5B ratio)
    (8, 1),   # Extreme GQA
])
class TestCausalAttentionHeadConfigs:
    def test_head_configuration(self, num_q_heads, num_kv_heads):
        """Test various Q/KV head configurations."""
        torch.manual_seed(42)
        batch, seq_len, head_dim = 2, 8, 64
        q = torch.randn(batch, num_q_heads, seq_len, head_dim)
        k = torch.randn(batch, num_kv_heads, seq_len, head_dim)
        v = torch.randn(batch, num_kv_heads, seq_len, head_dim)
        out = causal_attention(q, k, v)
        assert out.shape == (batch, num_q_heads, seq_len, head_dim)
