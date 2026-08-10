"""Tests for the SBFL hit-spectrum primitives."""

import torch

from susgrad.models import TabularMLP
from susgrad.sbfl import (
    BOUNDED_METRICS,
    METRICS,
    HitSpectrum,
    compute_hit_spectrum,
    get_activations_with_hooks,
    get_dstar,
    get_ochiai,
    get_tarantula,
    unsqueeze_tensors,
)


def _random_spectrum(n_neurons=64, n_samples=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    activations = torch.rand(n_samples, n_neurons, generator=g)
    targets = (torch.rand(n_samples, generator=g) > 0.5).unsqueeze(1).expand(-1, n_neurons)
    return compute_hit_spectrum(activations, targets, threshold=0.5)


def test_bounded_metrics_within_unit_interval_and_no_nan():
    hs = _random_spectrum()
    for name in BOUNDED_METRICS:  # ochiai, tarantula
        values = METRICS[name](hs)
        assert torch.isfinite(values).all(), f"{name} produced NaN/inf"
        assert float(values.min()) >= 0.0, f"{name} below 0"
        assert float(values.max()) <= 1.0, f"{name} above 1"


def test_metrics_handle_degenerate_zero_spectrum_without_nan():
    # No active neurons, no failures: every count zero -> metrics must stay finite.
    zeros = torch.zeros(10)
    hs = HitSpectrum(a_s=zeros, a_f=zeros, n_s=zeros, n_f=zeros, layer_shape=(10,))
    for name, fn in METRICS.items():
        values = fn(hs)
        assert torch.isfinite(values).all(), f"{name} produced NaN/inf on zero spectrum"


def test_ochiai_stays_bounded_when_successes_dominate():
    # Regression for the a_s-numerator bug: a_s >> n_f used to give ~sqrt(a_s/n_f)
    # (>> 1). With the correct a_f numerator it must stay within [0, 1].
    hs = HitSpectrum(
        a_s=torch.tensor([100.0]), a_f=torch.tensor([3.0]),
        n_s=torch.tensor([0.0]), n_f=torch.tensor([1.0]), layer_shape=(1,),
    )
    val = float(get_ochiai(hs)[0])
    assert 0.0 <= val <= 1.0, val


def test_dstar_is_nonnegative_finite_and_uses_exponent():
    hs = _random_spectrum()
    d3 = get_dstar(hs, star=3)
    assert torch.isfinite(d3).all()
    assert float(d3.min()) >= 0.0
    # Larger exponent must not introduce NaNs either.
    d5 = get_dstar(hs, star=5)
    assert torch.isfinite(d5).all()


def test_hit_spectrum_counts_partition_samples():
    # 4 samples, 3 neurons.
    activations = torch.tensor(
        [
            [1.0, -1.0, 2.0],
            [-1.0, 3.0, -2.0],
            [2.0, 2.0, -1.0],
            [-3.0, -1.0, 1.0],
        ]
    )
    # success mask broadcast per neuron
    targets = torch.tensor([True, False, True, False]).unsqueeze(1).expand(-1, 3)

    hs = compute_hit_spectrum(activations, targets, threshold=0.0)

    # Every sample is either active or not for each neuron, success or failure:
    # the four counts must sum to the number of samples for each neuron.
    total = hs.a_s + hs.a_f + hs.n_s + hs.n_f
    assert torch.all(total == 4)
    assert hs.layer_shape == (3,)


def test_suspiciousness_is_finite_and_shaped():
    activations = torch.rand(50, 8)
    targets = (torch.rand(50) > 0.5).unsqueeze(1).expand(-1, 8)
    hs = compute_hit_spectrum(activations, targets, threshold=0.5)

    ochiai = get_ochiai(hs)
    tarantula = get_tarantula(hs)

    assert ochiai.shape == (8,)
    assert tarantula.shape == (8,)
    assert torch.isfinite(ochiai).all()
    assert torch.isfinite(tarantula).all()


def test_hooks_capture_every_linear_layer():
    model = TabularMLP(input_dim=6, output_dim=2, hidden_dims=(16, 8))
    x = torch.randn(4, 6)
    activations, output = get_activations_with_hooks(model, x)
    # Three Linear layers -> three captured activation tensors.
    assert len(activations) == 3
    assert output.shape == (4, 2)


def test_unsqueeze_handles_conv_shapes():
    conv_act = torch.randn(5, 3, 4, 4)  # (batch, channels, h, w)
    targets = torch.tensor([True, False, True, False, True])
    flat_act, flat_targets = unsqueeze_tensors(conv_act, targets)
    assert flat_act.shape == (5, 3 * 4 * 4)
    assert flat_targets.shape == (5, 3 * 4 * 4)
