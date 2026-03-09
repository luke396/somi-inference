"""Pytest configuration and fixtures."""
import pytest
import torch


@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    return 42
