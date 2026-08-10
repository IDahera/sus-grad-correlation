"""Helpers for reporting tensor sizes in logs."""

from typing import Mapping

import torch


def human_bytes(n: int) -> str:
    """Format a byte count as a short human-readable string (e.g. '1.5 MB')."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def tensor_bytes(t: torch.Tensor) -> int:
    """Number of bytes backing a tensor's storage."""
    return t.element_size() * t.nelement()


def mapping_bytes(d: Mapping[str, torch.Tensor]) -> int:
    """Total bytes across a {name: tensor} mapping."""
    return sum(tensor_bytes(t) for t in d.values())


def describe_mapping(d: Mapping[str, torch.Tensor]) -> str:
    """Concise one-line summary: layer count, total neurons, total size."""
    n_layers = len(d)
    n_values = sum(t.nelement() for t in d.values())
    return f"{n_layers} layers, {n_values} values, {human_bytes(mapping_bytes(d))}"
