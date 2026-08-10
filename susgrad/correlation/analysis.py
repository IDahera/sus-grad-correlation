"""Assemble per-epoch dumps into per-neuron correlation tensors."""

from typing import Dict, Iterable, List, Optional

import torch

from susgrad.correlation.metrics import CORRELATIONS

# {method: {layer: tensor}}
MethodLayerTensors = Dict[str, Dict[str, torch.Tensor]]


def _stack_epochs(per_epoch: List[Dict[str, torch.Tensor]], layer: str) -> torch.Tensor:
    """Stack a layer's per-epoch tensors into shape ``(epochs, *layer)``."""
    return torch.stack([snap[layer] for snap in per_epoch], dim=0)


def compute_correlations(
    susp_per_epoch: List[Dict[str, torch.Tensor]],
    grad_per_epoch: List[Dict[str, torch.Tensor]],
    *,
    methods: Optional[Iterable[str]] = None,
) -> MethodLayerTensors:
    """Per-neuron correlation between suspiciousness and gradient across epochs.

    Args:
        susp_per_epoch: list (ordered by epoch) of ``{layer: tensor}`` for one
            chosen suspiciousness metric.
        grad_per_epoch: matching list of ``{layer: tensor}`` gradients.
        methods: subset of ``("pearson", "spearman")``; default all.

    Returns ``{method: {layer: tensor}}`` where each tensor has the layer's shape.
    """
    if len(susp_per_epoch) != len(grad_per_epoch):
        raise ValueError(
            f"epoch count mismatch: {len(susp_per_epoch)} susp vs "
            f"{len(grad_per_epoch)} grad."
        )
    if len(susp_per_epoch) < 2:
        raise ValueError("Need at least 2 epochs to compute a correlation.")

    method_names = tuple(methods) if methods is not None else tuple(CORRELATIONS)
    unknown = set(method_names) - set(CORRELATIONS)
    if unknown:
        raise ValueError(f"Unknown method(s): {sorted(unknown)}. Choose from {list(CORRELATIONS)}.")

    layers = list(susp_per_epoch[0])
    if set(layers) != set(grad_per_epoch[0]):
        raise ValueError("Suspiciousness and gradient layers differ.")

    results: MethodLayerTensors = {m: {} for m in method_names}
    for layer in layers:
        susp_series = _stack_epochs(susp_per_epoch, layer)
        grad_series = _stack_epochs(grad_per_epoch, layer)
        if susp_series.shape != grad_series.shape:
            raise ValueError(
                f"Layer {layer!r}: susp series {tuple(susp_series.shape)} != "
                f"grad series {tuple(grad_series.shape)}."
            )
        for method in method_names:
            results[method][layer] = CORRELATIONS[method](susp_series, grad_series).cpu()

    return results
