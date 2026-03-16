"""Tests for LLM high-level API."""

import pytest
import torch
from somi_inference.entrypoints.llm import LLM


# ============================================================================
# Task 15: LLM Initialization Tests
# ============================================================================


@pytest.mark.slow
def test_llm_initialization():
    """LLM should initialize successfully"""
    llm = LLM("Qwen/Qwen2.5-0.5B", num_blocks=128)

    assert llm.tokenizer is not None
    assert llm.kv_manager is not None
    assert llm.engine is not None
    assert llm._next_seq_id == 0


# ============================================================================
# Task 16: LLM Generate Tests
# ============================================================================


@pytest.mark.slow
def test_llm_generate_greedy():
    """LLM.generate with greedy decoding"""
    llm = LLM("Qwen/Qwen2.5-0.5B", num_blocks=128)

    output = llm.generate("Hello", max_new_tokens=5, temperature=0.0)

    assert isinstance(output, str)
    assert len(output) > 0


@pytest.mark.slow
def test_llm_generate_sampling():
    """LLM.generate with sampling"""
    llm = LLM("Qwen/Qwen2.5-0.5B", num_blocks=128)

    output = llm.generate("Hello", max_new_tokens=5, temperature=0.8)

    assert isinstance(output, str)
    assert len(output) > 0


@pytest.mark.slow
def test_llm_generate_with_top_k():
    """LLM.generate with top-k sampling"""
    llm = LLM("Qwen/Qwen2.5-0.5B", num_blocks=128)

    output = llm.generate("Hello", max_new_tokens=5, temperature=0.8, top_k=50)

    assert isinstance(output, str)
    assert len(output) > 0


@pytest.mark.slow
def test_llm_generate_with_top_p():
    """LLM.generate with top-p sampling"""
    llm = LLM("Qwen/Qwen2.5-0.5B", num_blocks=128)

    output = llm.generate("Hello", max_new_tokens=5, temperature=0.8, top_p=0.9)

    assert isinstance(output, str)
    assert len(output) > 0


def test_llm_generate_stream_not_implemented():
    """LLM.generate_stream should raise NotImplementedError"""
    llm = LLM("Qwen/Qwen2.5-0.5B", num_blocks=128)

    with pytest.raises(NotImplementedError, match="Streaming not yet implemented"):
        list(llm.generate_stream("Hello", max_new_tokens=5))
