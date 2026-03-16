"""Tests for Sampler and SamplingParams."""

import pytest
import torch
from somi_inference.core.sampler import Sampler, SamplingParams


# ============================================================================
# Task 1: SamplingParams Tests
# ============================================================================


def test_sampling_params_default_values():
    """默认参数应该是合法的"""
    params = SamplingParams()
    assert params.temperature == 1.0
    assert params.top_k == -1
    assert params.top_p == 1.0
    assert params.repetition_penalty == 1.0


def test_sampling_params_invalid_temperature():
    """temperature < 0 应该抛出异常"""
    with pytest.raises(AssertionError):
        SamplingParams(temperature=-0.1)


def test_sampling_params_invalid_top_k():
    """top_k = 0 应该抛出异常"""
    with pytest.raises(AssertionError):
        SamplingParams(top_k=0)


def test_sampling_params_invalid_top_p():
    """top_p <= 0 或 > 1.0 应该抛出异常"""
    with pytest.raises(AssertionError):
        SamplingParams(top_p=0.0)
    with pytest.raises(AssertionError):
        SamplingParams(top_p=1.1)


def test_sampling_params_invalid_repetition_penalty():
    """repetition_penalty <= 0 应该抛出异常"""
    with pytest.raises(AssertionError):
        SamplingParams(repetition_penalty=0.0)


# ============================================================================
# Task 2: Greedy Sampling Tests
# ============================================================================


def test_greedy_sampling_single():
    """temperature=0 应该返回 argmax token"""
    sampler = Sampler()
    logits = torch.tensor([[1.0, 3.0, 2.0]])  # argmax = 1
    params = SamplingParams(temperature=0.0)

    tokens = sampler.sample(logits, params)

    assert tokens.shape == (1,)
    assert tokens[0].item() == 1


def test_greedy_sampling_batch():
    """batch greedy sampling"""
    sampler = Sampler()
    logits = torch.tensor([
        [1.0, 3.0, 2.0],  # argmax = 1
        [5.0, 2.0, 3.0],  # argmax = 0
    ])
    params = SamplingParams(temperature=0.0)

    tokens = sampler.sample(logits, params)

    assert tokens.shape == (2,)
    assert tokens[0].item() == 1
    assert tokens[1].item() == 0


# ============================================================================
# Task 3: Temperature Scaling Tests
# ============================================================================


def test_temperature_scaling():
    """temperature > 0 应该进行 multinomial sampling"""
    sampler = Sampler()
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    params = SamplingParams(temperature=0.8)

    # Run multiple times to check randomness
    torch.manual_seed(42)
    tokens = [sampler.sample(logits, params).item() for _ in range(100)]

    # 应该有多个不同的 token（不是总是 argmax）
    unique_tokens = set(tokens)
    assert len(unique_tokens) > 1, f"Expected multiple tokens, got {unique_tokens}"


def test_temperature_high_makes_uniform():
    """temperature 很高应该使分布接近均匀"""
    sampler = Sampler()
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    params = SamplingParams(temperature=10.0)

    torch.manual_seed(42)
    tokens = [sampler.sample(logits, params).item() for _ in range(300)]

    # 统计每个 token 的频率
    from collections import Counter

    counts = Counter(tokens)

    # 每个 token 应该出现次数接近 100（均匀分布）
    for token_id in range(3):
        assert (
            50 < counts[token_id] < 150
        ), f"Token {token_id} count {counts[token_id]} not uniform"


# ============================================================================
# Task 4: Top-K Filtering Tests
# ============================================================================


def test_top_k_filtering():
    """top_k 应该只保留 top-k 个 logits"""
    sampler = Sampler()
    # logits: [0.1, 0.2, 0.3, 0.4, 0.5]
    # top_k=2 应该只保留 0.4 和 0.5 (indices 3, 4)
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    params = SamplingParams(temperature=1.0, top_k=2)

    torch.manual_seed(42)
    tokens = [sampler.sample(logits, params).item() for _ in range(100)]

    # 只应该采样到 token 3 或 4
    unique_tokens = set(tokens)
    assert unique_tokens.issubset(
        {3, 4}
    ), f"Expected only tokens 3,4, got {unique_tokens}"
    assert len(unique_tokens) == 2, "Should sample both top-k tokens"


def test_top_k_disabled():
    """top_k=-1 应该不过滤"""
    sampler = Sampler()
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    params = SamplingParams(temperature=1.0, top_k=-1)

    torch.manual_seed(42)
    tokens = [sampler.sample(logits, params).item() for _ in range(500)]

    # 应该能采样到所有 5 个 token
    unique_tokens = set(tokens)
    assert len(unique_tokens) == 5, f"Expected all 5 tokens, got {unique_tokens}"


# ============================================================================
# Task 5: Top-P Filtering Tests
# ============================================================================


def test_top_p_filtering():
    """top_p 应该保留累积概率 <= top_p 的 tokens"""
    sampler = Sampler()
    # Logits that give clear probability distribution
    logits = torch.tensor(
        [[10.0, 5.0, 1.0, 0.1, 0.01]]
    )  # Probs: ~0.88, ~0.11, ~0.002, ...
    params = SamplingParams(temperature=1.0, top_p=0.95)

    torch.manual_seed(42)
    tokens = [sampler.sample(logits, params).item() for _ in range(100)]

    # 应该只采样到 token 0 和 1（累积概率 ~0.99）
    unique_tokens = set(tokens)
    assert unique_tokens.issubset({0, 1}), f"Expected only tokens 0,1, got {unique_tokens}"


def test_top_p_disabled():
    """top_p=1.0 应该不过滤"""
    sampler = Sampler()
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    params = SamplingParams(temperature=1.0, top_p=1.0)

    torch.manual_seed(42)
    tokens = [sampler.sample(logits, params).item() for _ in range(500)]

    # 应该能采样到所有 token
    unique_tokens = set(tokens)
    assert len(unique_tokens) == 5, f"Expected all 5 tokens, got {unique_tokens}"


# ============================================================================
# Task 6: Batch Sampling with Different Params
# ============================================================================


def test_batch_sampling_different_params():
    """batch 内每个 seq 可以有不同的采样参数"""
    sampler = Sampler()
    logits = torch.tensor([
        [1.0, 2.0, 3.0],  # seq 0: greedy -> token 2
        [1.0, 2.0, 3.0],  # seq 1: temperature=1.0 -> random
    ])
    params = [
        SamplingParams(temperature=0.0),  # greedy
        SamplingParams(temperature=1.0),  # sampling
    ]

    torch.manual_seed(42)
    tokens = sampler.sample(logits, params)

    # Seq 0 应该总是返回 argmax
    assert tokens[0].item() == 2

    # Seq 1 应该有随机性（多次采样）
    torch.manual_seed(42)
    tokens_list = [sampler.sample(logits, params)[1].item() for _ in range(50)]
    unique_tokens = set(tokens_list)
    assert len(unique_tokens) > 1, "Seq 1 should have randomness"
