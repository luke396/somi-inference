"""Pytest configuration and fixtures."""
import pytest
import torch

from somi_inference.models.qwen2 import (
    ForwardContext,
    ForwardMode,
    RotaryEmbedding,
    causal_attention,
)


@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    return 42


@pytest.fixture
def small_model_config():
    """Small model config for unit tests."""
    return dict(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_seq_size=512,
        rms_norm_eps=1e-6,
    )


@pytest.fixture
def adapter_config():
    """Config for adapter tests."""
    return dict(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_seq_size=512,
    )


@pytest.fixture
def make_rope_inputs():
    """Factory for creating RoPE cos/sin tensors."""

    def _make(head_dim: int, seq_len: int, batch: int = 1):
        rope = RotaryEmbedding(hid_dim=head_dim, max_seq_len=512)
        posi_idx = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        cos, sin = rope(posi_idx)
        return cos, sin

    return _make


@pytest.fixture
def make_forward_context():
    """Factory for creating ForwardContext."""

    def _make(seq_len: int, batch: int = 1, mode=ForwardMode.PREFILL):
        return ForwardContext(
            mode=mode,
            attn_fn=lambda q, k, v, layer_idx: causal_attention(q, k, v),
            posi_idx=torch.arange(seq_len).unsqueeze(0).expand(batch, -1),
        )

    return _make
