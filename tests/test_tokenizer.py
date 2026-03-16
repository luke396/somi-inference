"""Tests for Tokenizer."""

import pytest
from somi_inference.tokenizer import Tokenizer


@pytest.fixture
def tokenizer():
    """Create Qwen tokenizer."""
    return Tokenizer("Qwen/Qwen2.5-0.5B")


# ============================================================================
# Task 13: Tokenizer Tests
# ============================================================================


def test_tokenizer_encode(tokenizer):
    """Tokenizer should encode text to token ids"""
    text = "Hello, world!"
    token_ids = tokenizer.encode(text)

    assert isinstance(token_ids, list)
    assert len(token_ids) > 0
    assert all(isinstance(t, int) for t in token_ids)


def test_tokenizer_decode(tokenizer):
    """Tokenizer should decode token ids to text"""
    token_ids = [9906, 11, 1879, 0]  # Example token ids
    text = tokenizer.decode(token_ids)

    assert isinstance(text, str)
    assert len(text) > 0


def test_tokenizer_roundtrip(tokenizer):
    """Encode then decode should preserve text (approximately)"""
    original = "Hello, world!"
    token_ids = tokenizer.encode(original)
    decoded = tokenizer.decode(token_ids)

    # May not be exact due to tokenization, but should be similar
    assert "Hello" in decoded or "hello" in decoded


def test_tokenizer_eos_token_id(tokenizer):
    """Tokenizer should expose eos_token_id"""
    assert isinstance(tokenizer.eos_token_id, int)
    assert tokenizer.eos_token_id > 0


def test_tokenizer_batch_encode(tokenizer):
    """Tokenizer should support batch encoding"""
    texts = ["Hello", "World"]
    token_ids_list = tokenizer.batch_encode(texts)

    assert isinstance(token_ids_list, list)
    assert len(token_ids_list) == 2
    assert all(isinstance(ids, list) for ids in token_ids_list)


def test_tokenizer_batch_decode(tokenizer):
    """Tokenizer should support batch decoding"""
    token_ids_list = [[9906], [1879]]
    texts = tokenizer.batch_decode(token_ids_list)

    assert isinstance(texts, list)
    assert len(texts) == 2
    assert all(isinstance(t, str) for t in texts)
