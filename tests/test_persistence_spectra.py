"""Tests that stored grad/susp tensors round-trip and keep the model's dimensions."""

import torch

from susgrad.grads import compute_gradient_spectrum
from susgrad.persistence import (
    load_gradients,
    load_suspiciousness,
    save_gradients,
    save_suspiciousness,
)
from susgrad.sbfl import compute_suspiciousness, get_activations_with_hooks

CPU = torch.device("cpu")


def _layer_neuron_shapes(model, loader):
    x, _ = next(iter(loader))
    acts, _ = get_activations_with_hooks(model, x)
    return {name: tuple(a.shape[1:]) for name, a in acts.items()}


def test_gradients_roundtrip_preserves_dims(model, normalized_loader, tmp_path):
    grads = compute_gradient_spectrum(model, normalized_loader, device=CPU)
    save_gradients(grads, "unit_combo", "e010", epoch=1, base_dir=tmp_path)
    loaded = load_gradients("unit_combo", "e010", epoch=1, base_dir=tmp_path)

    expected = _layer_neuron_shapes(model, normalized_loader)
    assert set(loaded) == set(expected)
    for name, shape in expected.items():
        assert tuple(loaded[name].shape) == shape
        assert torch.allclose(loaded[name], grads[name])


def test_suspiciousness_roundtrip_preserves_dims(model, normalized_loader, tmp_path):
    susp = compute_suspiciousness(model, normalized_loader, device=CPU)
    save_suspiciousness(susp, "unit_combo", "e010", epoch=2, base_dir=tmp_path)
    loaded = load_suspiciousness("unit_combo", "e010", epoch=2, base_dir=tmp_path)

    from susgrad.sbfl import METRIC_NAMES

    expected = _layer_neuron_shapes(model, normalized_loader)
    assert set(loaded) == set(METRIC_NAMES)
    for metric, layers in loaded.items():
        assert set(layers) == set(expected), f"{metric} layers differ"
        for name, shape in expected.items():
            assert tuple(layers[name].shape) == shape, f"{metric}/{name} shape changed"
