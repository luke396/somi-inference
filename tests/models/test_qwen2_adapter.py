"""Tests for QwenAdapter."""

from typing import cast

import torch
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

from somi_inference.core.paged_attention import KVCacheManager
from somi_inference.models.qwen2 import QwenMLP, QwenModel
from somi_inference.models.qwen2_adapter import QwenAdapter, load_from_hf

ADAPTER_CONFIG = dict(
    vocab_size=100,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_seq_size=512,
)


def _make_kv_manager():
    return KVCacheManager(
        num_blocks=20,
        block_size=4,
        num_kv_heads=2,
        head_dim=16,
        n_layers=2,
    )


class TestQwenAdapterPrefill:
    def test_mlp_backend_setting_updates_all_decoder_layers(self):
        """Setting adapter.mlp_backend should update every decoder-layer MLP."""
        model = QwenModel(**ADAPTER_CONFIG)
        adapter = QwenAdapter(model)

        assert adapter.mlp_backend == "torch_ref"
        assert all(
            cast("QwenMLP", layer.mlp).backend == "torch_ref" for layer in model.layers
        )

        adapter.mlp_backend = "triton"

        assert adapter.mlp_backend == "triton"
        assert all(
            cast("QwenMLP", layer.mlp).backend == "triton" for layer in model.layers
        )

    def test_prefill_logits_shape(self):
        """prefill returns only the last prompt position logits."""
        model = QwenModel(**ADAPTER_CONFIG)
        adapter = QwenAdapter(model)
        kv = _make_kv_manager()
        kv.register_sequence(0)

        tokens = torch.randint(0, 100, (1, 6))
        logits = adapter.prefill(tokens, kv, seq_id=0)
        assert logits.shape == (1, 1, 100)

    def test_prefill_projects_only_last_hidden_state(self):
        """prefill should call lm_head with the final hidden state slice only."""
        model = QwenModel(**ADAPTER_CONFIG)
        captured_shapes: list[torch.Size] = []

        class RecordingAdapter(QwenAdapter):
            def _lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
                captured_shapes.append(hidden_states.shape)
                return super()._lm_head(hidden_states)

        adapter = RecordingAdapter(model)
        kv = _make_kv_manager()
        kv.register_sequence(0)

        tokens = torch.randint(0, 100, (1, 6))
        logits = adapter.prefill(tokens, kv, seq_id=0)

        assert captured_shapes == [torch.Size((1, 1, ADAPTER_CONFIG["hidden_size"]))]
        assert logits.shape == (1, 1, 100)

    def test_prefill_writes_kv_cache(self):
        """prefill writes KV for all layers and advances token count."""
        model = QwenModel(**ADAPTER_CONFIG)
        adapter = QwenAdapter(model)
        kv = _make_kv_manager()
        kv.register_sequence(0)

        tokens = torch.randint(0, 100, (1, 5))
        adapter.prefill(tokens, kv, seq_id=0)

        assert kv.get_num_tokens(0) == 5
        assert len(kv.get_block_ids(0)) == 2  # 5 tokens / block_size=4 → 2 blocks

    def test_prefill_default_matches_explicit_torch_ref(self):
        """The default prefill backend should match the explicit torch_ref path."""
        torch.manual_seed(42)
        model = QwenModel(**ADAPTER_CONFIG)
        default_adapter = QwenAdapter(model)
        ref_adapter = QwenAdapter(model, prefill_attention_backend="torch_ref")
        default_kv = _make_kv_manager()
        ref_kv = _make_kv_manager()
        default_kv.register_sequence(0)
        ref_kv.register_sequence(0)

        tokens = torch.randint(0, 100, (1, 6))
        default_logits = default_adapter.prefill(tokens, default_kv, seq_id=0)
        ref_logits = ref_adapter.prefill(tokens, ref_kv, seq_id=0)

        torch.testing.assert_close(default_logits, ref_logits)


class TestQwenAdapterDecode:
    def test_decode_logits_shape(self):
        """decode returns (batch, 1, vocab_size) logits."""
        model = QwenModel(**ADAPTER_CONFIG)
        adapter = QwenAdapter(model)
        kv = _make_kv_manager()

        # Prefill first to populate cache
        kv.register_sequence(0)
        adapter.prefill(torch.randint(0, 100, (1, 4)), kv, seq_id=0)

        # Decode one token
        input_ids = torch.randint(0, 100, (1, 1))
        logits = adapter.decode(input_ids, kv, seq_ids=[0])
        assert logits.shape == (1, 1, 100)

    def test_decode_advances_cache(self):
        """decode writes 1 new KV token per sequence."""
        model = QwenModel(**ADAPTER_CONFIG)
        adapter = QwenAdapter(model)
        kv = _make_kv_manager()

        kv.register_sequence(0)
        adapter.prefill(torch.randint(0, 100, (1, 3)), kv, seq_id=0)
        assert kv.get_num_tokens(0) == 3

        adapter.decode(torch.randint(0, 100, (1, 1)), kv, seq_ids=[0])
        assert kv.get_num_tokens(0) == 4

    def test_decode_batch(self):
        """decode handles multiple sequences in a batch."""
        model = QwenModel(**ADAPTER_CONFIG)
        adapter = QwenAdapter(model)
        kv = _make_kv_manager()

        # Prefill two sequences
        for sid in [0, 1]:
            kv.register_sequence(sid)
            adapter.prefill(torch.randint(0, 100, (1, 3)), kv, seq_id=sid)

        # Batch decode
        input_ids = torch.randint(0, 100, (2, 1))
        logits = adapter.decode(input_ids, kv, seq_ids=[0, 1])
        assert logits.shape == (2, 1, 100)


class TestQwenAdapterConsistency:
    def test_prefill_then_decode_produces_valid_logits(self):
        """Full prefill→decode flow produces finite, non-zero logits."""
        torch.manual_seed(42)
        model = QwenModel(**ADAPTER_CONFIG)
        adapter = QwenAdapter(model)
        kv = _make_kv_manager()
        kv.register_sequence(0)

        # Prefill
        prompt = torch.randint(0, 100, (1, 6))
        prefill_logits = adapter.prefill(prompt, kv, seq_id=0)
        assert torch.isfinite(prefill_logits).all()
        assert not torch.allclose(prefill_logits, torch.zeros_like(prefill_logits))

        # Decode 3 tokens
        next_token = prefill_logits[:, 0, :].argmax(dim=-1, keepdim=True)  # (1, 1)
        for step in range(3):
            logits = adapter.decode(next_token, kv, seq_ids=[0])
            assert torch.isfinite(logits).all()
            assert logits.shape == (1, 1, 100)
            next_token = logits[:, 0, :].argmax(dim=-1, keepdim=True)

        assert kv.get_num_tokens(0) == 9  # 6 prefill + 3 decode


def test_load_from_hf_loads_mapped_weights(monkeypatch):
    """load_from_hf should map HF keys into a frozen QwenAdapter model."""
    rope_theta = 12345.0
    reference_model = QwenModel(**ADAPTER_CONFIG, rms_norm_eps=1e-6, rope_theta=rope_theta)
    hf_state_dict = {}
    for key, tensor in reference_model.state_dict().items():
        if key.endswith(".mlp.gate_up_proj.weight"):
            prefix = key.removesuffix(".gate_up_proj.weight")
            intermediate_size = ADAPTER_CONFIG["intermediate_size"]
            hf_state_dict[f"model.{prefix}.gate_proj.weight"] = tensor[
                :intermediate_size
            ].clone()
            hf_state_dict[f"model.{prefix}.up_proj.weight"] = tensor[
                intermediate_size:
            ].clone()
            continue
        if key == "token_embedding.weight":
            hf_key = "model.embed_tokens.weight"
        elif key == "final_layernorm.weight":
            hf_key = "model.norm.weight"
        else:
            hf_key = f"model.{key}"
        hf_state_dict[hf_key] = tensor.clone()
    hf_state_dict["lm_head.weight"] = reference_model.token_embedding.weight.clone()
    hf_state_dict["model.layers.0.self_attn.rotary_emb.inv_freq"] = torch.ones(
        ADAPTER_CONFIG["head_dim"] // 2
    )

    class FakeConfig:
        vocab_size = ADAPTER_CONFIG["vocab_size"]
        hidden_size = ADAPTER_CONFIG["hidden_size"]
        intermediate_size = ADAPTER_CONFIG["intermediate_size"]
        num_hidden_layers = ADAPTER_CONFIG["num_hidden_layers"]
        num_attention_heads = ADAPTER_CONFIG["num_attention_heads"]
        num_key_value_heads = ADAPTER_CONFIG["num_key_value_heads"]
        max_position_embeddings = ADAPTER_CONFIG["max_seq_size"]
        rms_norm_eps = 1e-6
        rope_parameters = {"rope_theta": rope_theta}

    class FakeHFModel:
        def state_dict(self):
            return hf_state_dict

    def fake_config_from_pretrained(model_name):
        assert model_name == "fake/qwen"
        return FakeConfig()

    def fake_model_from_pretrained(model_name, dtype):
        assert model_name == "fake/qwen"
        assert dtype == torch.float32
        return FakeHFModel()

    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        staticmethod(fake_config_from_pretrained),
    )
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(fake_model_from_pretrained),
    )

    adapter = load_from_hf("fake/qwen")

    assert isinstance(adapter, QwenAdapter)
    for key, tensor in reference_model.state_dict().items():
        assert torch.equal(adapter.model.state_dict()[key], tensor)
    assert all(not parameter.requires_grad for parameter in adapter.model.parameters())
