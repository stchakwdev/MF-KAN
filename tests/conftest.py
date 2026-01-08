"""Pytest configuration and fixtures."""

import pytest
import torch


@pytest.fixture
def device():
    """Get test device."""
    return "cpu"


@pytest.fixture
def seed():
    """Fixed seed for reproducibility."""
    return 42


@pytest.fixture
def set_seed(seed):
    """Set random seeds."""
    torch.manual_seed(seed)
    return seed
