"""Tests that stored correlation tensors round-trip and keep the model's dimensions."""

import torch

from susgrad.correlation import compute_correlations
from susgrad.persistence import load_correlation, save_correlation
from susgrad.sbfl import get_activations_with_hooks


def _layer_neuron_shapes(model, loader):
    x, _ = next(iter(loader))
    acts, _ = get_activations_with_hooks(model, x)
    return {name: tuple(a.shape[1:]) for name, a in acts.items()}


def _fake_per_epoch(shapes, n_epochs, seed=0):
    """Build a list of {layer: tensor} snapshots with the given per-layer shapes."""
    g = torch.Generator().manual_seed(seed)
    return [
        {name: torch.rand(shape, generator=g) for name, shape in shapes.items()}
        for _ in range(n_epochs)
    ]


def test_correlation_roundtrip_preserves_dims(model, normalized_loader, tmp_path):
    shapes = _layer_neuron_shapes(model, normalized_loader)
    susp = _fake_per_epoch(shapes, n_epochs=5, seed=1)
    grad = _fake_per_epoch(shapes, n_epochs=5, seed=2)

    corr = compute_correlations(susp, grad, methods=["pearson", "spearman"])
    save_correlation(corr, "unit_combo", "e005", base_dir=tmp_path)
    loaded = load_correlation("unit_combo", "e005", base_dir=tmp_path)

    assert set(loaded) == {"pearson", "spearman"}
    for method, layers in loaded.items():
        assert set(layers) == set(shapes), f"{method} layers differ"
        for name, shape in shapes.items():
            assert tuple(layers[name].shape) == shape, f"{method}/{name} dim changed"
            assert torch.allclose(layers[name], corr[method][name])


def test_nested_per_metric_correlation_roundtrips(model, normalized_loader, tmp_path):
    # Real layout from correlate.py: {susp_metric: {corr_method: {layer: tensor}}}.
    shapes = _layer_neuron_shapes(model, normalized_loader)
    grad = _fake_per_epoch(shapes, n_epochs=4, seed=9)
    nested = {}
    for metric in ("ochiai", "tarantula", "dstar"):
        susp = _fake_per_epoch(shapes, n_epochs=4, seed=hash(metric) % 100)
        nested[metric] = compute_correlations(susp, grad, methods=["pearson", "spearman"])

    save_correlation(nested, "unit_combo", "e100", base_dir=tmp_path)
    loaded = load_correlation("unit_combo", "e100", base_dir=tmp_path)

    assert set(loaded) == {"ochiai", "tarantula", "dstar"}
    for metric, methods in loaded.items():
        assert set(methods) == {"pearson", "spearman"}
        for layers in methods.values():
            for name, shape in shapes.items():
                assert tuple(layers[name].shape) == shape


def test_correlation_values_within_unit_range(model, normalized_loader, tmp_path):
    shapes = _layer_neuron_shapes(model, normalized_loader)
    susp = _fake_per_epoch(shapes, n_epochs=6, seed=3)
    grad = _fake_per_epoch(shapes, n_epochs=6, seed=4)
    corr = compute_correlations(susp, grad)
    for layers in corr.values():
        for t in layers.values():
            assert torch.isfinite(t).all()
            assert float(t.min()) >= -1.0 and float(t.max()) <= 1.0
