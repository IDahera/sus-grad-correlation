"""Per-neuron correlation between suspiciousness and gradient across epochs."""

from susgrad.correlation.analysis import compute_correlations
from susgrad.correlation.metrics import (
    CORRELATIONS,
    pearson_correlation,
    spearman_correlation,
)

__all__ = [
    "CORRELATIONS",
    "pearson_correlation",
    "spearman_correlation",
    "compute_correlations",
]
