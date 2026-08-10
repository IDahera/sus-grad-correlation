"""Spectrum-Based Fault Localisation (SBFL) primitives for neural networks.

This is a self-contained port of the suspiciousness-computation logic so that
this project does not depend on the original ``static-sbfl-for-nn`` package.

Metrics provided (all derived from the same hit spectrum): Tarantula, Ochiai,
D*, Jaccard, Kulczynski2, Op2 (Naish), GP13 (Yoo). Tarantula, Ochiai, Jaccard and
Kulczynski2 are bounded to [0, 1]; D*, Op2 and GP13 are unbounded.

SBFL idea transferred to neural networks
----------------------------------------
In classical SBFL each program statement is "executed" or "not executed" on a
test, and the test "passes" or "fails". Here the analogy is:

    * a neuron is *active* when its activation exceeds a threshold
      (the equivalent of a statement being executed), and
    * a sample is a *success* when the model handles it correctly
      (the equivalent of a passing test).

For every neuron we count four quantities across all samples:

    a_s : active   & success   (neuron fired on a correctly-handled sample)
    a_f : active   & failure    (neuron fired on an incorrectly-handled sample)
    n_s : inactive & success
    n_f : inactive & failure

These four tensors -- shaped like the layer they describe -- are the "hit
spectrum". Suspiciousness metrics (Ochiai, Tarantula) are computed from them.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class HitSpectrum:
    """Per-neuron hit-spectrum counts for a single layer.

    Each tensor has the shape of the layer's activations (minus the batch dim).
    """

    a_s: torch.Tensor  # active   & success
    a_f: torch.Tensor  # active   & failure
    n_s: torch.Tensor  # inactive & success
    n_f: torch.Tensor  # inactive & failure
    layer_shape: Tuple[int, ...]


@dataclass
class SuspiciousnessResult:
    """Computed suspiciousness scores for one layer."""

    ochiai: torch.Tensor
    tarantula: torch.Tensor
    model_name: str = ""
    layer_name: str = ""


def get_activations_with_hooks(
    model: nn.Module, input_data: torch.Tensor
) -> Tuple[dict, torch.Tensor]:
    """Run a forward pass and capture the output of every Linear/Conv2d layer.

    Returns a ``(activations, output)`` tuple where ``activations`` maps the
    layer name to its detached output tensor.
    """
    activations: dict[str, torch.Tensor] = {}

    def hook_fn(name: str):
        def hook(_module, _input, output):
            activations[name] = output.detach()

        return hook

    hooks = []
    for name, module in model.named_modules():
        # Capture both dense and convolutional layers.
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    output = model(input_data)

    for hook in hooks:
        hook.remove()

    return activations, output


def unsqueeze_tensors(
    activations: torch.Tensor, targets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten activations to ``(batch, neurons)`` and broadcast targets to match.

    Conv activations ``(b, c, h, w)`` are flattened to ``(b, c*h*w)`` so each
    feature-map position is treated as an individual "neuron".
    """
    activations_copy = activations.clone()
    targets_copy = targets.clone()

    if activations.dim() == 4:  # convolutional layer
        b, c, h, w = activations_copy.shape
        activations_copy = activations_copy.view(b, c * h * w)
        targets_copy = targets_copy.unsqueeze(1).expand(-1, c * h * w)
    else:  # dense layer
        targets_copy = targets_copy.unsqueeze(1).expand(-1, activations_copy.shape[1])

    return activations_copy, targets_copy


def compute_hit_spectrum(
    activations: torch.Tensor, targets: torch.Tensor, threshold: float = 0.0
) -> HitSpectrum:
    """Count a_s / a_f / n_s / n_f per neuron.

    Args:
        activations: ``(batch, neurons)`` activation values.
        targets: ``(batch, neurons)`` boolean tensor, True where the sample is a
            *success* (correctly handled by the model).
        threshold: a neuron counts as *active* when its activation > threshold.
    """
    active = activations > threshold
    targets = targets.bool()

    return HitSpectrum(
        a_s=torch.sum(torch.logical_and(targets, active), dim=0),
        a_f=torch.sum(torch.logical_and(~targets, active), dim=0),
        n_s=torch.sum(torch.logical_and(targets, ~active), dim=0),
        n_f=torch.sum(torch.logical_and(~targets, ~active), dim=0),
        layer_shape=tuple(activations.shape[1:]),
    )


def get_ochiai(hs: HitSpectrum) -> torch.Tensor:
    """Ochiai suspiciousness (range [0, 1]).

        ochiai = a_f / sqrt((a_f + n_f) * (a_f + a_s))

    The numerator is the failure-aligned count ``a_f`` (neuron active on a wrongly
    handled sample). Because ``a_f <= a_f + n_f`` and ``a_f <= a_f + a_s``, the
    ratio never exceeds 1. (The original reference code used ``a_s`` in the
    numerator, which can exceed 1 and is not the standard Ochiai metric.)
    """
    a_s, a_f, n_f = hs.a_s.float(), hs.a_f.float(), hs.n_f.float()

    numerator = a_f
    denominator = torch.sqrt((a_f + n_f) * (a_f + a_s))

    result = torch.div(numerator, denominator)
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def get_tarantula(hs: HitSpectrum) -> torch.Tensor:
    """Tarantula suspiciousness derived from the hit spectrum (range [0, 1])."""
    a_s, a_f = hs.a_s.float(), hs.a_f.float()
    n_s, n_f = hs.n_s.float(), hs.n_f.float()

    numerator = torch.div(a_f, a_f + n_f)
    denominator_l = torch.div(a_f, a_f + n_f)
    denominator_r = torch.div(a_s, a_s + n_s)

    result = torch.div(numerator, denominator_l + denominator_r)
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def get_dstar(hs: HitSpectrum, star: int = 3) -> torch.Tensor:
    """D* (D-star) suspiciousness with exponent *star* (default 3, i.e. D^3).

    D*^star = a_f^star / (a_s + n_f).

    Unlike Ochiai/Tarantula this is **not** bounded to [0, 1]; larger means more
    suspicious. Division by zero is mapped to 0 via ``nan_to_num``.
    """
    a_f = hs.a_f.float()
    a_s, n_f = hs.a_s.float(), hs.n_f.float()

    numerator = torch.pow(a_f, star)
    denominator = a_s + n_f

    result = torch.div(numerator, denominator)
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def get_jaccard(hs: HitSpectrum) -> torch.Tensor:
    """Jaccard suspiciousness (range [0, 1]):  a_f / (a_f + n_f + a_s)."""
    a_s, a_f, n_f = hs.a_s.float(), hs.a_f.float(), hs.n_f.float()
    result = torch.div(a_f, a_f + n_f + a_s)
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def get_kulczynski2(hs: HitSpectrum) -> torch.Tensor:
    """Kulczynski2 suspiciousness (range [0, 1]):

        0.5 * ( a_f/(a_f+n_f) + a_f/(a_f+a_s) )

    The mean of the two conditional coverages (failure recall and precision).
    """
    a_s, a_f, n_f = hs.a_s.float(), hs.a_f.float(), hs.n_f.float()
    left = torch.nan_to_num(torch.div(a_f, a_f + n_f))
    right = torch.nan_to_num(torch.div(a_f, a_f + a_s))
    return 0.5 * (left + right)


def get_op2(hs: HitSpectrum) -> torch.Tensor:
    """Op2 (Naish) suspiciousness:  a_f - a_s / (a_s + n_s + 1).

    Proven 'maximal' for single-fault programs. **Not** bounded to [0, 1] (it can
    be mildly negative and grows with a_f).
    """
    a_s, a_f, n_s = hs.a_s.float(), hs.a_f.float(), hs.n_s.float()
    result = a_f - torch.div(a_s, a_s + n_s + 1.0)
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def get_gp13(hs: HitSpectrum) -> torch.Tensor:
    """GP13 (Yoo) suspiciousness:  a_f * (1 + 1 / (2*a_s + a_f)).

    A genetic-programming-evolved formula. Unbounded above.
    """
    a_s, a_f = hs.a_s.float(), hs.a_f.float()
    factor = 1.0 + torch.nan_to_num(torch.div(1.0, 2.0 * a_s + a_f))
    result = a_f * factor
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


# Registry so callers/scripts can iterate metrics by name. Order = sub-tab order.
METRICS = {
    "ochiai": get_ochiai,
    "tarantula": get_tarantula,
    "dstar": get_dstar,
    "jaccard": get_jaccard,
    "kulczynski2": get_kulczynski2,
    "op2": get_op2,
    "gp13": get_gp13,
}

METRIC_NAMES = tuple(METRICS)

# The three metrics the MAIN (ensemble) pipeline reports on. All seven remain
# available and tested; this is the deliberately small, well-established subset
# the experiments and the write-up focus on -- one bounded similarity coefficient
# (ochiai), one classic ratio (tarantula) and one unbounded family (D*).
CORE_METRIC_NAMES = ("ochiai", "tarantula", "dstar")

# Metrics whose values are mathematically bounded to the closed interval [0, 1].
BOUNDED_METRICS = ("ochiai", "tarantula", "jaccard", "kulczynski2")
