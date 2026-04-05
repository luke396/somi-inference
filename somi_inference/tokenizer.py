"""Tokenizer wrapper around Hugging Face tokenizers."""

from __future__ import annotations

from typing import Protocol, cast

from transformers import AutoTokenizer


class HFTokenizer(Protocol):
    """Minimal tokenizer protocol required by the inference wrapper."""

    eos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Encode a single string into token ids."""
        ...

    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
    ) -> dict[str, list[list[int]]]:
        """Encode a batch of strings."""
        ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        """Decode one token sequence into text."""
        ...

    def batch_decode(
        self,
        token_ids_list: list[list[int]],
        *,
        skip_special_tokens: bool,
    ) -> list[str]:
        """Decode a batch of token sequences into text."""
        ...


class Tokenizer:
    """Thin wrapper exposing the minimal tokenizer API for inference."""

    def __init__(self, model_name: str) -> None:
        """Load the Hugging Face tokenizer for the given model."""
        self.hf_tokenizer = cast(
            "HFTokenizer",
            AutoTokenizer.from_pretrained(model_name),
        )
        assert self.hf_tokenizer.eos_token_id is not None, (
            "Tokenizer must define eos_token_id"
        )
        self.eos_token_id = self.hf_tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        """Encode one string without implicitly adding special tokens."""
        return list(self.hf_tokenizer.encode(text, add_special_tokens=False))

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        """Encode a batch of strings while preserving order."""
        encoded = self.hf_tokenizer(texts, add_special_tokens=False)
        return [list(token_ids) for token_ids in encoded["input_ids"]]

    def decode(self, token_ids: list[int]) -> str:
        """Decode one token sequence while skipping special tokens."""
        return self.hf_tokenizer.decode(token_ids, skip_special_tokens=True)

    def batch_decode(self, token_ids_list: list[list[int]]) -> list[str]:
        """Decode a batch of token sequences while skipping special tokens."""
        return list(
            self.hf_tokenizer.batch_decode(
                token_ids_list,
                skip_special_tokens=True,
            )
        )
