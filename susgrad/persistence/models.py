"""Saving and loading whole models.

Uses ``torch.save`` / ``torch.load`` on the model object so a loaded model is
immediately usable without re-declaring its class. Files are ``<name>.pt`` under
:data:`susgrad.utils.paths.DEFAULT_MODEL_DIR` unless a directory is given.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from susgrad.utils.devices import DEVICE
from susgrad.utils.paths import DEFAULT_MODEL_DIR, ensure_dir

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def save_model(model: nn.Module, name: str, directory: Optional[PathLike] = None) -> Path:
    """Save *model* as ``<name>.pt`` and return the written path."""
    target_dir = ensure_dir(directory if directory is not None else DEFAULT_MODEL_DIR)
    filename = name if name.endswith(".pt") else f"{name}.pt"
    path = target_dir / filename
    torch.save(model, path)
    logger.info("Saved model to %s", path)
    return path


def load_model(path: PathLike, *, map_location: torch.device = DEVICE) -> nn.Module:
    """Load a model previously saved with :func:`save_model`.

    ``weights_only=False`` is required because the whole model object is stored.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model file at {path}")
    model = torch.load(path, map_location=map_location, weights_only=False)
    logger.info("Loaded model from %s", path)
    return model
