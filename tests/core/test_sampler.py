"""Tests for sampling strategies."""

import pytest
import torch

from somi_inference.core.sampler import Sampler, SamplingParams

EXPECTED_TOP_K_COUNT = 2


def test_sampling_params_default_values() -> None:
    """Default sampling params should be valid."""
    params = SamplingParams()

    assert params.temperature == 1.0
    assert params.top_k == -1
    assert params.top_p == 1.0
    assert params.repetition_penalty == 1.0


def test_sampling_params_invalid_temperature() -> None:
    """Negative temperature should be rejected."""
    with pytest.raises(AssertionError, match="temperature"):
        SamplingParams(temperature=-0.1)


def test_sampling_params_invalid_top_k() -> None:
    """Zero top-k should be rejected."""
    with pytest.raises(AssertionError, match="top_k"):
        SamplingParams(top_k=0)


def test_sampling_params_invalid_top_p_low() -> None:
    """Non-positive top-p should be rejected."""
    with pytest.raises(AssertionError, match="top_p"):
        SamplingParams(top_p=0.0)


def test_sampling_params_invalid_top_p_high() -> None:
    """Top-p above one should be rejected."""
    with pytest.raises(AssertionError, match="top_p"):
        SamplingParams(top_p=1.1)


def test_sampling_params_invalid_repetition_penalty() -> None:
    """Non-positive repetition penalty should be rejected."""
    with pytest.raises(AssertionError, match="repetition_penalty"):
        SamplingParams(repetition_penalty=0.0)


def test_greedy_sampling_single() -> None:
    """temperature=0 should return argmax token."""
    sampler = Sampler()
    logits = torch.tensor([[1.0, 3.0, 2.0]])

    tokens = sampler.sample(logits, SamplingParams(temperature=0.0))

    assert tokens.shape == (1,)
    assert tokens.tolist() == [1]


def test_greedy_sampling_batch() -> None:
    """Greedy sampling should work for a batch."""
    sampler = Sampler()
    logits = torch.tensor(
        [
            [1.0, 3.0, 2.0],
            [5.0, 2.0, 3.0],
        ]
    )

    tokens = sampler.sample(logits, SamplingParams(temperature=0.0))

    assert tokens.shape == (2,)
    assert tokens.tolist() == [1, 0]


def test_repetition_penalty_changes_greedy_choice() -> None:
    """Repetition penalty should affect greedy output."""
    sampler = Sampler()
    logits = torch.tensor([[2.0, 1.5]])
    params = SamplingParams(temperature=0.0, repetition_penalty=2.0)

    tokens = sampler.sample(logits, params, token_history=[[0]])

    assert tokens.tolist() == [1]


def test_top_k_keeps_exactly_k_tokens() -> None:
    """Top-k should keep exactly k candidates even with ties."""
    sampler = Sampler()
    logits = torch.tensor([[5.0, 4.0, 4.0, 1.0]])
    params = [SamplingParams(temperature=1.0, top_k=2)]

    filtered = sampler._apply_top_k(logits.clone(), params)  # noqa: SLF001
    finite_mask = torch.isfinite(filtered[0])

    assert finite_mask.sum().item() == EXPECTED_TOP_K_COUNT
    assert finite_mask[0].item() is True


def test_top_k_one_makes_sampling_deterministic() -> None:
    """Top-k=1 should reduce sampling to argmax."""
    sampler = Sampler()
    logits = torch.tensor([[1.0, 3.0, 2.0]])

    tokens = sampler.sample(logits, SamplingParams(temperature=1.0, top_k=1))

    assert tokens.tolist() == [1]


def test_top_p_filters_by_cumulative_probability() -> None:
    """Top-p should remove tokens outside the nucleus set."""
    sampler = Sampler()
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    params = [SamplingParams(temperature=1.0, top_p=0.5)]

    filtered = sampler._apply_top_p(logits.clone(), params)  # noqa: SLF001
    finite_mask = torch.isfinite(filtered[0])

    assert finite_mask.tolist() == [True, False, False, False]


def test_temperature_zero_ignores_top_k_and_top_p() -> None:
    """Greedy mode should ignore stochastic filtering params."""
    sampler = Sampler()
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    params = [SamplingParams(temperature=0.0, top_k=1, top_p=0.1)]

    top_k_filtered = sampler._apply_top_k(logits.clone(), params)  # noqa: SLF001
    top_p_filtered = sampler._apply_top_p(logits.clone(), params)  # noqa: SLF001

    torch.testing.assert_close(top_k_filtered, logits)
    torch.testing.assert_close(top_p_filtered, logits)


def test_sample_supports_heterogeneous_batch_params() -> None:
    """Each sequence should use its own sampling config."""
    sampler = Sampler()
    logits = torch.tensor(
        [
            [1.0, 3.0, 2.0],
            [0.0, 2.0, 1.0],
        ]
    )
    params = [
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=1.0, top_k=1),
    ]

    tokens = sampler.sample(logits, params)

    assert tokens.tolist() == [1, 1]
