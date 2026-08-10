"""Shared pytest fixtures.

A synthetic, in-memory tabular dataset keeps the test suite fast and offline
(no MNIST download, no OpenML fetch).
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# Make the project importable when running pytest from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from susgrad.models import TabularMLP

N_FEATURES = 6
N_CLASSES = 2


@pytest.fixture
def normalized_loader() -> DataLoader:
    """A small, already-normalised binary tabular loader."""
    torch.manual_seed(0)
    X = torch.randn(64, N_FEATURES)            # ~zero mean / unit var
    y = (X.sum(dim=1) > 0).long()              # separable-ish binary target
    return DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)


@pytest.fixture
def raw_loader() -> DataLoader:
    """An un-normalised loader (values in the 0-255 range)."""
    torch.manual_seed(0)
    X = torch.randint(0, 256, (64, N_FEATURES)).float()
    y = (X.sum(dim=1) > X.sum(dim=1).median()).long()
    return DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)


@pytest.fixture
def model() -> TabularMLP:
    return TabularMLP(input_dim=N_FEATURES, output_dim=N_CLASSES, hidden_dims=(16, 8))
