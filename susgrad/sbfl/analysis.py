"""Per-layer suspiciousness computation over a dataset.

Ties the spectrum primitives together: run the model over the evaluation data,
decide which samples were *successes* (predicted correctly), build a hit
spectrum per Linear/Conv2d layer and turn it into per-neuron suspiciousness.

A sample is a **success** when the model's argmax prediction equals the label
(the "proper test outcome" analogy). Each returned tensor has the layer's own
shape, so suspiciousness lines up 1:1 with the per-neuron gradient tensors.
"""

import logging
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from susgrad.sbfl.spectrum import (
    METRICS,
    compute_hit_spectrum,
    get_activations_with_hooks,
    unsqueeze_tensors,
)
from susgrad.utils.devices import DEVICE

logger = logging.getLogger(__name__)

# {metric_name: {layer_name: tensor}}
MetricLayerTensors = Dict[str, Dict[str, torch.Tensor]]


def collect_activations_and_success(model: nn.Module, loader: DataLoader, device):
    """Run *model* over *loader*, returning per-layer activations and a success mask.

    Returns ``(activations, success)`` where each maps layer_name -> tensor with
    the batch dimension concatenated across the whole loader. ``success`` is a
    boolean tensor (True where the model predicted correctly).
    """
    model.to(device)
    model.eval()

    all_acts: Dict[str, list] = {}
    all_success: Dict[str, list] = {}

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            activations, output = get_activations_with_hooks(model, data)
            success = output.argmax(dim=1) == target  # proper output -> success
            for name, act in activations.items():
                all_acts.setdefault(name, []).append(act)
                all_success.setdefault(name, []).append(success)

    activations = {n: torch.cat(parts, dim=0) for n, parts in all_acts.items()}
    success = {n: torch.cat(parts, dim=0) for n, parts in all_success.items()}
    return activations, success


def compute_suspiciousness(
    model: nn.Module,
    loader: DataLoader,
    *,
    metrics: Optional[Iterable[str]] = None,
    threshold: float = 0.0,
    dstar_exponent: int = 3,
    device: torch.device = DEVICE,
) -> MetricLayerTensors:
    """Compute per-neuron suspiciousness for every layer and requested metric.

    Args:
        metrics: subset of the names in ``METRICS`` (ochiai, tarantula, dstar,
            jaccard, kulczynski2, op2, gp13); default all.
        threshold: a neuron is active when activation > threshold.
        dstar_exponent: the * in D*.

    Returns ``{metric: {layer_name: tensor}}`` with each tensor shaped like its
    layer's activations.
    """
    metric_names = tuple(metrics) if metrics is not None else tuple(METRICS)
    unknown = set(metric_names) - set(METRICS)
    if unknown:
        raise ValueError(f"Unknown metric(s): {sorted(unknown)}. Choose from {list(METRICS)}.")

    activations, success = collect_activations_and_success(model, loader, device)

    results: MetricLayerTensors = {name: {} for name in metric_names}
    for layer_name, acts in activations.items():
        # Reshape suspiciousness back to the layer's own activation shape (e.g.
        # (C, H, W) for conv layers) so stored dims reflect the model dimensions.
        layer_shape = tuple(acts.shape[1:])
        flat_acts, flat_targets = unsqueeze_tensors(acts, success[layer_name])
        hs = compute_hit_spectrum(flat_acts, flat_targets, threshold)
        for name in metric_names:
            fn = METRICS[name]
            values = fn(hs, dstar_exponent) if name == "dstar" else fn(hs)
            results[name][layer_name] = values.reshape(layer_shape).cpu()

    return results
