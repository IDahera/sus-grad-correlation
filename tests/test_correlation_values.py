"""Exact-value unit tests for the correlation metrics.

Series are shaped ``(epochs, *layer)``; correlation is taken along the epoch axis
and returns one value per neuron.
"""

import pytest
import torch

from susgrad.correlation import pearson_correlation, spearman_correlation
from susgrad.correlation.metrics import CORRELATIONS


def test_pearson_perfect_positive_and_negative():
    # Two neurons: one perfectly increasing-with, one perfectly opposing.
    susp = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grad = torch.tensor([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    r = pearson_correlation(susp, grad)
    assert r.shape == (2,)
    assert float(r[0]) == pytest.approx(1.0)
    assert float(r[1]) == pytest.approx(-1.0)


def test_pearson_known_intermediate_value():
    susp = torch.tensor([1.0, 2.0, 3.0, 4.0]).unsqueeze(1)      # (4, 1)
    grad = torch.tensor([1.0, 2.0, 3.0, 100.0]).unsqueeze(1)
    r = pearson_correlation(susp, grad)
    assert float(r[0]) == pytest.approx(0.785026421, abs=1e-6)


def test_spearman_is_one_for_monotonic_even_when_pearson_is_not():
    susp = torch.tensor([1.0, 2.0, 3.0, 4.0]).unsqueeze(1)
    grad = torch.tensor([1.0, 2.0, 3.0, 100.0]).unsqueeze(1)    # monotonic, not linear
    assert float(spearman_correlation(susp, grad)[0]) == pytest.approx(1.0)
    assert float(pearson_correlation(susp, grad)[0]) < 0.99


def test_constant_series_correlation_is_zero():
    # Zero-variance suspiciousness series -> undefined correlation -> 0.
    susp = torch.tensor([2.0, 2.0, 2.0]).unsqueeze(1)
    grad = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(1)
    for fn in CORRELATIONS.values():
        assert float(fn(susp, grad)[0]) == pytest.approx(0.0)


def test_correlation_preserves_layer_shape():
    # (epochs=5, C=2, H=3, W=3) -> per-neuron result (2, 3, 3).
    susp = torch.rand(5, 2, 3, 3)
    grad = torch.rand(5, 2, 3, 3)
    for fn in CORRELATIONS.values():
        out = fn(susp, grad)
        assert tuple(out.shape) == (2, 3, 3)
        assert torch.isfinite(out).all()
        assert float(out.min()) >= -1.0 and float(out.max()) <= 1.0
