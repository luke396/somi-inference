"""Tests for Model Loader."""

import pytest
from somi_inference.models.loader import load_model
from somi_inference.models.base import ModelAdapter


# ============================================================================
# Task 14: Model Loader Tests
# ============================================================================


@pytest.mark.slow
def test_load_model_qwen():
    """load_model should load Qwen2 model"""
    adapter, config = load_model("Qwen/Qwen2.5-0.5B")

    assert isinstance(adapter, ModelAdapter)
    assert isinstance(config, dict)
    assert config["model_type"] == "qwen2"
    assert "num_hidden_layers" in config
    assert "num_attention_heads" in config


def test_load_model_unsupported():
    """load_model should raise error for unsupported model"""
    with pytest.raises(ValueError, match="Unsupported model type"):
        load_model("gpt2")  # Not supported yet
