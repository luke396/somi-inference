"""Tests for QwenModel."""

import torch
from somi_inference.models.qwen2 import QwenModel


class TestQwenModel:
    def test_forward_shape(self, small_model_config, make_forward_context):
        """Output shape should be (batch, seq_len, hidden_size)."""
        model = QwenModel(**small_model_config, num_hidden_layers=2)

        input_ids = torch.randint(0, 100, (2, 10))
        out = model(input_ids, make_forward_context(seq_len=10, batch=2))
        assert out.shape == (2, 10, small_model_config["hidden_size"])

    def test_embedding_lookup(self, small_model_config, make_forward_context):
        """Verify token embedding lookup works correctly."""
        model = QwenModel(**small_model_config, num_hidden_layers=1)

        input_ids = torch.tensor([[0, 1, 2]])
        out = model(input_ids, make_forward_context(seq_len=3))

        assert not torch.allclose(out[0, 0], out[0, 1])
        assert not torch.allclose(out[0, 1], out[0, 2])

    def test_multi_layer_processing(self, small_model_config, make_forward_context):
        """Verify multiple layers transform the input."""
        torch.manual_seed(42)
        model_1layer = QwenModel(**small_model_config, num_hidden_layers=1)

        torch.manual_seed(42)
        model_2layer = QwenModel(**small_model_config, num_hidden_layers=2)

        input_ids = torch.randint(0, 100, (1, 10))
        ctx = make_forward_context(seq_len=10)
        out_1layer = model_1layer(input_ids, ctx)
        out_2layer = model_2layer(input_ids, ctx)

        assert not torch.allclose(out_1layer, out_2layer, atol=0.1)

    def test_position_encoding(self, small_model_config, make_forward_context):
        """Verify position encoding affects output."""
        torch.manual_seed(42)
        model = QwenModel(**small_model_config, num_hidden_layers=1)

        # Same token at different absolute positions should produce different outputs
        out1 = model(torch.tensor([[5]]), make_forward_context(seq_len=1))
        out2 = model(torch.tensor([[7, 5]]), make_forward_context(seq_len=2))

        assert not torch.allclose(out1[0, 0], out2[0, 1], atol=0.01)

    def test_gradient_flow(self, small_model_config, make_forward_context):
        """Verify gradients flow end-to-end through the full model."""
        model = QwenModel(**small_model_config, num_hidden_layers=2)

        input_ids = torch.randint(0, 100, (1, 8))
        out = model(input_ids, make_forward_context(seq_len=8))
        out.sum().backward()

        assert model.token_embedding.weight.grad is not None
        assert model.layers[0].self_attn.q_proj.weight.grad is not None
        assert model.layers[1].mlp.gate_proj.weight.grad is not None
        assert model.final_layernorm.weight.grad is not None
