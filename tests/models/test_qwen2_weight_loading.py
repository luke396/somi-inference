"""Tests for HF weight loading."""

from somi_inference.models.qwen2 import QwenModel


class TestWeightKeyMapping:
    """Test weight key mapping without downloading real model."""

    def test_somi_model_keys_match_expected(self):
        """Verify somi model state_dict keys match what load_from_hf expects."""
        model = QwenModel(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_seq_size=512,
        )
        sd = model.state_dict()

        # Must have these keys
        assert "token_embedding.weight" in sd
        assert "final_layernorm.weight" in sd
        assert "layers.0.self_attn.q_proj.weight" in sd
        assert "layers.0.self_attn.q_proj.bias" in sd
        assert "layers.0.self_attn.o_proj.weight" in sd
        assert "layers.0.mlp.gate_proj.weight" in sd
        assert "layers.1.input_layernorm.weight" in sd

        # Must NOT have lm_head (tied weights)
        assert "lm_head.weight" not in sd


class TestHfKeyMapping:
    """Test the HF-to-somi key mapping function."""

    def test_map_hf_key_strips_model_prefix(self):
        from somi_inference.models.qwen2_adapter import _map_hf_key

        assert _map_hf_key("model.layers.0.self_attn.q_proj.weight") == "layers.0.self_attn.q_proj.weight"

    def test_map_hf_key_renames_embed_tokens(self):
        from somi_inference.models.qwen2_adapter import _map_hf_key

        assert _map_hf_key("model.embed_tokens.weight") == "token_embedding.weight"

    def test_map_hf_key_renames_final_norm(self):
        from somi_inference.models.qwen2_adapter import _map_hf_key

        assert _map_hf_key("model.norm.weight") == "final_layernorm.weight"

    def test_map_hf_key_skips_lm_head(self):
        from somi_inference.models.qwen2_adapter import _map_hf_key

        assert _map_hf_key("lm_head.weight") is None

    def test_map_hf_key_skips_rotary_emb(self):
        from somi_inference.models.qwen2_adapter import _map_hf_key

        assert _map_hf_key("model.layers.0.self_attn.rotary_emb.inv_freq") is None
