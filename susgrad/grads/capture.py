"""Per-neuron gradient capture.

For every Linear/Conv2d layer we record the gradient of the loss with respect to
that layer's **output activations** and reduce it to one value per neuron by
taking the mean absolute gradient over the evaluation samples:

    grad[neuron] = mean_over_samples( | d loss / d activation[neuron] | )

The result has the same per-layer shape as the suspiciousness tensors, so the
two are directly comparable neuron-by-neuron.
"""

import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from susgrad.utils.devices import DEVICE

logger = logging.getLogger(__name__)

LayerTensors = Dict[str, torch.Tensor]


def compute_gradient_spectrum(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device = DEVICE,
) -> LayerTensors:
    """Return ``{layer_name: tensor}`` of mean |dloss/dactivation| per neuron."""
    model.to(device)
    model.eval()  # no dropout/batchnorm here; we only need gradients, not training

    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        captured: Dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(name):
            def hook(_module, _inp, output):
                # Keep grad on this non-leaf activation so we can read it post-backward.
                output.retain_grad()
                captured[name] = output

            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                handles.append(module.register_forward_hook(make_hook(name)))

        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()

        for name, act in captured.items():
            if act.grad is None:
                continue
            # |grad| summed over the batch dimension -> per-neuron accumulator.
            batch_abs = act.grad.abs().sum(dim=0).detach()
            if name not in sums:
                sums[name] = batch_abs.clone()
                counts[name] = act.shape[0]
            else:
                sums[name] += batch_abs
                counts[name] += act.shape[0]

        for h in handles:
            h.remove()

    return {name: (sums[name] / counts[name]).cpu() for name in sums}
