"""Tests for QwenModel."""

import torch
from somi_inference.models.qwen2 import QwenModel

# Shared small-model config for tests
SMALL_CONFIG = dict(
    vocab_size=100,
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=32,
    max_seq_size=512,
)


class TestQwenModel:
    def test_forward_shape(self):
        """Output shape should be (batch, seq_len, hidden_size)."""
        model = QwenModel(**SMALL_CONFIG, num_hidden_layers=2)

        input_ids = torch.randint(0, 100, (2, 10))
        out = model(input_ids)
        assert out.shape == (2, 10, 64)

    def test_embedding_lookup(self):
        """Verify token embedding lookup works correctly."""
        model = QwenModel(**SMALL_CONFIG, num_hidden_layers=1)

        input_ids = torch.tensor([[0, 1, 2]])
        out = model(input_ids)

        assert not torch.allclose(out[0, 0], out[0, 1])
        assert not torch.allclose(out[0, 1], out[0, 2])

    def test_multi_layer_processing(self):
        """Verify multiple layers transform the input."""
        torch.manual_seed(42)
        model_1layer = QwenModel(**SMALL_CONFIG, num_hidden_layers=1)

        torch.manual_seed(42)
        model_2layer = QwenModel(**SMALL_CONFIG, num_hidden_layers=2)

        input_ids = torch.randint(0, 100, (1, 10))
        out_1layer = model_1layer(input_ids)
        out_2layer = model_2layer(input_ids)

        assert not torch.allclose(out_1layer, out_2layer, atol=0.1)

    def test_position_encoding(self):
        """Verify position encoding affects output."""
        torch.manual_seed(42)
        model = QwenModel(**SMALL_CONFIG, num_hidden_layers=1)

        # Same token at different absolute positions should produce different outputs
        out1 = model(torch.tensor([[5]]))       # token 5 at position 0
        out2 = model(torch.tensor([[7, 5]]))    # token 5 at position 1

        assert not torch.allclose(out1[0, 0], out2[0, 1], atol=0.01)

    def test_gradient_flow(self):
        """Verify gradients flow end-to-end through the full model."""
        model = QwenModel(**SMALL_CONFIG, num_hidden_layers=2)

        input_ids = torch.randint(0, 100, (1, 8))
        out = model(input_ids)
        out.sum().backward()

        assert model.token_embedding.weight.grad is not None
        assert model.layers[0].self_attn.q_proj.weight.grad is not None
        assert model.layers[1].mlp.gate_proj.weight.grad is not None
        assert model.final_layernorm.weight.grad is not None

    def test_forward_shape_gqa(self):
        """Full model with GQA config (fewer KV heads than Q heads)."""
        model = QwenModel(
            vocab_size=500,
            hidden_size=384,
            intermediate_size=768,
            num_hidden_layers=2,
            num_attention_heads=12,
            num_key_value_heads=2,
            head_dim=32,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000.0,
            max_seq_size=512,
        )

        input_ids = torch.randint(0, 500, (2, 16))
        out = model(input_ids)
        assert out.shape == (2, 16, 384)
