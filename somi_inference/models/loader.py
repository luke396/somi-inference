"""Model loader dispatching to model-family-specific adapters."""

from __future__ import annotations

from collections.abc import Callable

from transformers import AutoConfig

from somi_inference.models.base import ModelAdapter
from somi_inference.models.qwen2_adapter import load_from_hf as load_qwen2_from_hf

ModelLoader = Callable[[str], ModelAdapter]

MODEL_LOADERS: dict[str, ModelLoader] = {
    "qwen2": load_qwen2_from_hf,
}


def load_model(model_name: str) -> tuple[ModelAdapter, dict[str, object]]:
    """Load a model adapter and normalized config for a Hugging Face model."""
    config = AutoConfig.from_pretrained(model_name).to_dict()
    model_type = config.get("model_type")
    loader = MODEL_LOADERS.get(model_type)
    if loader is None:
        message = f"Unsupported model type: {model_type}"
        raise ValueError(message)

    adapter = loader(model_name)
    return adapter, config
