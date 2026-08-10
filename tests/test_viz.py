"""Tests for visualisation transforms: bounds + data-point preservation."""

import numpy as np
import pytest
import torch

from susgrad.viz import (
    VisualizationError,
    align_layers,
    build_heatmap,
    correlation_stats,
    sorted_comparison,
    sorted_descending,
    to_layer_view,
)
from susgrad.viz.transform import HEATMAP_MAX_CELLS


def test_correlation_stats_exact_values():
    # |r|: 1, 1, 0, 0.6, 0.6 -> mean 0.64; median of signed = 0;
    # |r|>0.5 in 4/5 = 0.8; exactly zero in 1/5 = 0.2.
    st = correlation_stats([1.0, -1.0, 0.0, 0.6, -0.6])
    assert st["n"] == 5
    assert st["mean_abs"] == pytest.approx(0.64)
    assert st["median"] == pytest.approx(0.0)
    assert st["frac_strong"] == pytest.approx(0.8)
    assert st["frac_zero"] == pytest.approx(0.2)


def test_correlation_stats_excludes_non_finite():
    st = correlation_stats([0.5, float("nan"), float("inf"), -0.5])
    assert st["n"] == 2
    assert st["mean_abs"] == pytest.approx(0.5)


def test_correlation_stats_handles_empty():
    st = correlation_stats([])
    assert st["n"] == 0 and st["mean_abs"] == 0.0


def test_to_layer_view_preserves_all_values():
    t = torch.tensor([0.1, 0.9, 0.3, 0.5, 0.2])
    view = to_layer_view("layer", t)
    assert view.n_neurons == t.numel()
    assert not view.downsampled
    # Same multiset of values, nothing dropped or altered.
    assert sorted(view.values.tolist()) == pytest.approx(sorted(t.tolist()))


def test_value_range_violation_raises():
    bad = torch.tensor([0.5, 1.4, 0.2])  # 1.4 outside [0, 1]
    with pytest.raises(VisualizationError):
        to_layer_view("ochiai_layer", bad, value_range=(0.0, 1.0))


def test_large_layer_downsamples_heatmap_but_keeps_all_neurons():
    n = HEATMAP_MAX_CELLS * 3 + 7
    view = to_layer_view("wide", torch.arange(n, dtype=torch.float32))
    # Full per-neuron data preserved for the comparison/alignment.
    assert view.n_neurons == n
    assert view.downsampled and view.reduction > 1
    # Heatmap is bounded.
    assert view.heatmap_grid().size <= HEATMAP_MAX_CELLS * 1.1  # grid padding allowance
    assert view.display_values.size <= HEATMAP_MAX_CELLS


def test_exceeding_hard_ceiling_raises():
    # Small explicit ceiling so the test stays fast.
    with pytest.raises(VisualizationError):
        to_layer_view("huge", torch.zeros(101), max_neurons=100)


def test_non_finite_raises():
    nan_tensor = torch.tensor([0.1, float("nan"), 0.3])
    with pytest.raises(VisualizationError):
        to_layer_view("layer", nan_tensor)


def test_align_layers_requires_matching_neuron_counts():
    susp = {"a": torch.zeros(8), "b": torch.zeros(4)}
    grad_mismatch = {"a": torch.zeros(8), "b": torch.zeros(5)}  # b differs
    with pytest.raises(VisualizationError):
        align_layers(susp, grad_mismatch)


def test_align_layers_requires_same_layer_set():
    susp = {"a": torch.zeros(8)}
    grad = {"a": torch.zeros(8), "b": torch.zeros(8)}
    with pytest.raises(VisualizationError):
        align_layers(susp, grad)


def test_align_layers_ok_for_matching():
    susp = {"a": torch.zeros(8), "b": torch.zeros(4)}
    grad = {"a": torch.ones(8), "b": torch.ones(4)}
    assert set(align_layers(susp, grad)) == {"a", "b"}


def test_build_heatmap_conv_selects_channel_and_preserves_HxW():
    # (C=6, H=28, W=28) conv layer.
    t = torch.arange(6 * 28 * 28, dtype=torch.float32).reshape(6, 28, 28)
    hm0 = build_heatmap("conv1", t, channel=0)
    hm3 = build_heatmap("conv1", t, channel=3)
    assert hm0.grid.shape == (28, 28)          # true spatial resolution kept
    assert "channel 0/6" in hm0.info and "28×28" in hm0.info
    assert "channel 3/6" in hm3.info
    # Different channels show different data.
    assert not (hm0.grid == hm3.grid).all()
    # Channel index is clamped into range.
    assert "channel 1/6" in build_heatmap("conv1", t, channel=7).info  # 7 % 6 == 1


def test_build_heatmap_dense_is_square_overview():
    hm = build_heatmap("fc", torch.arange(256, dtype=torch.float32))
    assert hm.grid.shape == (16, 16)
    assert "256 neurons" in hm.info


def test_build_heatmap_2d_layer_used_directly():
    hm = build_heatmap("plane", torch.zeros(10, 4))
    assert hm.grid.shape == (10, 4)
    assert "10×4" in hm.info


def test_sorted_descending_orders_high_to_low():
    out = sorted_descending(np.array([0.2, 0.9, 0.5, 0.1]))
    assert list(out) == [0.9, 0.5, 0.2, 0.1]
    assert out.size == 4  # no datapoints lost


def test_sorted_comparison_is_a_permutation_and_keeps_pairs():
    susp = np.array([0.2, 0.9, 0.5, 0.1])
    grad = np.array([10.0, 20.0, 30.0, 40.0])
    order, susp_sorted, grad_sorted = sorted_comparison(susp, grad)

    # Number of data points unchanged.
    assert susp_sorted.size == susp.size
    assert grad_sorted.size == grad.size
    # Descending suspiciousness.
    assert np.all(np.diff(susp_sorted) <= 0)
    # Same multiset of values (a permutation), pairs preserved via `order`.
    assert sorted(susp_sorted.tolist()) == pytest.approx(sorted(susp.tolist()))
    assert np.allclose(grad_sorted, grad[order])
