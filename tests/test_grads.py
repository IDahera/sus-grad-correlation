"""Tests for per-neuron gradient capture and its dimensions."""

import torch

from susgrad.grads import compute_gradient_spectrum
from susgrad.sbfl import get_activations_with_hooks


def _layer_neuron_shapes(model, loader):
    """Expected per-layer activation shapes (minus batch) from a real forward pass."""
    x, _ = next(iter(loader))
    acts, _ = get_activations_with_hooks(model, x)
    return {name: tuple(a.shape[1:]) for name, a in acts.items()}


def test_gradient_spectrum_shapes_match_model_layers(model, normalized_loader):
    grads = compute_gradient_spectrum(model, normalized_loader, device=torch.device("cpu"))
    expected = _layer_neuron_shapes(model, normalized_loader)

    assert set(grads) == set(expected), "Gradient layers must match the model's layers."
    for name, shape in expected.items():
        assert tuple(grads[name].shape) == shape, f"{name}: {grads[name].shape} != {shape}"


def test_gradient_values_are_finite_and_nonnegative(model, normalized_loader):
    grads = compute_gradient_spectrum(model, normalized_loader, device=torch.device("cpu"))
    for name, t in grads.items():
        assert torch.isfinite(t).all(), f"{name} has non-finite gradients"
        assert float(t.min()) >= 0.0, f"{name} has negative mean-abs gradient"
