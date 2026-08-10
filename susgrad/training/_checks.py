"""Shared compatibility and normalisation checks used across the training code."""

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from susgrad.training.exceptions import DataNotNormalizedError, DimensionMismatchError


def _first_batch(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    for batch in loader:
        return batch
    raise DimensionMismatchError("Data loader is empty; nothing to check.")


def check_model_dataset_compatible(model: nn.Module, loader: DataLoader) -> None:
    """Raise :class:`DimensionMismatchError` if *model* cannot consume *loader*.

    Checks three things using one real batch:
      1. the per-sample input shape matches ``model.input_shape`` (if present);
      2. the labels stay within ``[0, model.output_dim)`` (if present);
      3. a real forward pass produces ``model.output_dim`` logits.
    """
    x, y = _first_batch(loader)

    expected_input = getattr(model, "input_shape", None)
    if expected_input is not None:
        actual_input = tuple(x.shape[1:])
        if actual_input != tuple(expected_input):
            raise DimensionMismatchError(
                f"Input shape mismatch: model expects {tuple(expected_input)} "
                f"per sample but dataset provides {actual_input}."
            )

    output_dim = getattr(model, "output_dim", None)
    if output_dim is not None and y.numel() > 0:
        max_label = int(y.max().item())
        if max_label >= output_dim:
            raise DimensionMismatchError(
                f"Label {max_label} is out of range for a model with "
                f"output_dim={output_dim}."
            )

    # Probe a real forward pass on a tiny slice to catch any remaining mismatch.
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            out = model(x[:1].to(device))
    except Exception as exc:  # noqa: BLE001 -- re-raise as a domain error
        raise DimensionMismatchError(
            f"Model failed a forward pass on the dataset: {exc}"
        ) from exc
    finally:
        model.train(was_training)

    if output_dim is not None and out.shape[-1] != output_dim:
        raise DimensionMismatchError(
            f"Model produced {out.shape[-1]} logits but output_dim={output_dim}."
        )


def assert_normalized(loader: DataLoader, max_abs: float = 50.0) -> None:
    """Raise :class:`DataNotNormalizedError` if inputs look un-normalised.

    Heuristic: normalised inputs (standardised tabular features, or images scaled
    to roughly zero mean / unit variance) have small magnitudes. Raw 0-255 pixels
    or un-scaled columns exceed *max_abs* and are rejected.
    """
    x, _ = _first_batch(loader)
    peak = float(x.abs().max().item())
    if peak > max_abs:
        raise DataNotNormalizedError(
            f"Input values reach {peak:.1f} (> {max_abs}); data does not look "
            "normalised for the model. Did you forget to scale/standardise it?"
        )
