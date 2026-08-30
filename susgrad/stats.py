"""Descriptive statistics for a set of per-neuron values.

Used by ``scripts/value_stats.py`` to answer the question a reader asks first
about any metric: *what range do these numbers actually live in?* Ochiai is
bounded to [0, 1], tarantula sits around 0.5, and D* runs into the tens of
millions -- a table of min/median/mean/max per layer is what makes those scales
comparable in a paper.

Kept separate from :mod:`susgrad.viz.textmap` (whose ``correlation_summary`` is
about *correlations*, and reports things like ``% |r| > 0.5`` that make no sense
for a raw suspiciousness score).
"""

from typing import Dict, Iterable, Sequence

import numpy as np
import torch

# The columns every stats row carries, in report order.
STAT_FIELDS = ("n", "min", "p05", "median", "mean", "p95", "max", "std", "frac_zero")

#: The ``layer`` value of a row that pools EVERY layer's neurons into one
#: population ("the whole model"). A sentinel rather than a literal ``"all"``
#: spelled out in three files, because the population trajectory plots select
#: rows by exactly this value.
ALL_LAYERS = "all"


def describe_values(values, *, percentiles: Sequence[float] = (5, 95)) -> Dict[str, float]:
    """Min / percentiles / median / mean / max / std / zero-share of *values*.

    Non-finite entries are dropped (they are never legitimate suspiciousness or
    gradient values, and silently averaging a NaN would poison the whole row).
    ``frac_zero`` is reported separately because a layer whose neurons never
    activate produces a pile of exact zeros that drags the mean down without
    saying anything about the neurons that do fire.
    """
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {field: 0.0 for field in STAT_FIELDS} | {"n": 0}

    lo, hi = percentiles
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, lo)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, hi)),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "frac_zero": float((arr == 0.0).mean()),
    }


def describe_stack(tensors: Iterable[torch.Tensor], **kwargs) -> Dict[str, float]:
    """:func:`describe_values` over several tensors pooled together.

    The ensemble experiment has one tensor per instance for the same layer; the
    interesting range is the one across the whole population, not per instance.
    """
    flat = [
        (t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)).reshape(-1)
        for t in tensors
    ]
    if not flat:
        return describe_values(np.zeros(0), **kwargs)
    return describe_values(np.concatenate(flat), **kwargs)


def format_row(stats: Dict[str, float], *, precision: int = 6) -> Dict[str, str]:
    """Stats as strings for a CSV: ``n`` stays an integer, the rest is %g."""
    out = {"n": str(int(stats.get("n", 0)))}
    for field in STAT_FIELDS:
        if field == "n":
            continue
        out[field] = f"{stats.get(field, 0.0):.{precision}g}"
    return out
