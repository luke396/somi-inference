"""Tests for model loader dispatch."""

import pytest
import transformers

import somi_inference.models.loader as loader_module
from somi_inference.models.loader import load_model


class FakeConfig:
    """Minimal AutoConfig stub."""

    def __init__(self, model_type: str) -> None:
        self.model_type = model_type

    def to_dict(self) -> dict[str, object]:
        """Return a config dict with the fields needed by Phase 2."""
        return {
            "model_type": self.model_type,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "hidden_size": 896,
        }


def test_load_model_dispatches_qwen_to_qwen_loader(monkeypatch):
    """Load model should route qwen2 configs to the Qwen adapter loader."""
    fake_adapter = object()
    loaded_model_names = []

    def fake_from_pretrained(model_name, *args, **kwargs):
        loaded_model_names.append(model_name)
        return FakeConfig(model_type="qwen2")

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setitem(loader_module.MODEL_LOADERS, "qwen2", lambda _model_name: fake_adapter)

    adapter, config = load_model("Qwen/Qwen2.5-0.5B")

    assert adapter is fake_adapter
    assert config["model_type"] == "qwen2"
    assert config["num_hidden_layers"] == 24
    assert loaded_model_names == ["Qwen/Qwen2.5-0.5B"]


def test_load_model_rejects_unsupported_model_types(monkeypatch):
    """Load model should fail fast for unsupported model families."""
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda _model_name, *args, **kwargs: FakeConfig(model_type="gpt2"),
    )

    with pytest.raises(ValueError, match="Unsupported model type"):
        load_model("gpt2")
