"""Correlation metrics, computed per neuron across the epoch axis.

Each function takes two tensors shaped ``(epochs, *layer)`` -- a per-neuron time
series of suspiciousness and of gradient -- and returns one correlation value per
neuron with shape ``(*layer,)``. A neuron whose series is constant (zero variance)
has undefined correlation; we return 0 for it.

Two standard coefficients are provided:
    * Pearson  -- linear correlation of the raw values.
    * Spearman -- Pearson correlation of the rank-transformed values (monotonic).
"""

import torch

_EPS = 1e-12


def _pearson_along_epochs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson r between *x* and *y* along axis 0 (epochs)."""
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    xm = x - x.mean(dim=0, keepdim=True)
    ym = y - y.mean(dim=0, keepdim=True)
    num = (xm * ym).sum(dim=0)
    den = torch.sqrt((xm**2).sum(dim=0) * (ym**2).sum(dim=0))
    r = torch.where(den > _EPS, num / den, torch.zeros_like(num))
    return r.clamp(-1.0, 1.0)


def _rank_along_epochs(t: torch.Tensor) -> torch.Tensor:
    """Mid-ranks along axis 0 (tied values share the average of their positions).

    Ties must NOT be broken by position: a neuron whose series is constant (a
    neuron that is never active, say -- common at epoch 0, where nothing has
    trained yet) would otherwise be handed the ranks 0, 1, 2, ... and come out
    perfectly correlated with anything, instead of the documented 0. With
    mid-ranks its rank series is constant too, so the zero-variance guard in
    :func:`_pearson_along_epochs` catches it, exactly like Pearson does.

    Axis 0 is short (epochs, or ensemble instances), so the pairwise comparison
    below costs ``len(axis 0)`` passes over the layer -- cheap, and it keeps the
    whole thing vectorised over neurons.
    """
    x = t.to(torch.float64)
    ranks = torch.empty_like(x)
    for i in range(x.shape[0]):
        value = x[i].unsqueeze(0)
        n_less = (x < value).sum(dim=0).to(torch.float64)
        n_equal = (x == value).sum(dim=0).to(torch.float64)
        ranks[i] = n_less + (n_equal - 1.0) / 2.0
    return ranks


def pearson_correlation(susp_series: torch.Tensor, grad_series: torch.Tensor) -> torch.Tensor:
    """Per-neuron Pearson correlation across epochs."""
    return _pearson_along_epochs(susp_series, grad_series)


def spearman_correlation(susp_series: torch.Tensor, grad_series: torch.Tensor) -> torch.Tensor:
    """Per-neuron Spearman correlation across epochs (Pearson of ranks)."""
    return _pearson_along_epochs(
        _rank_along_epochs(susp_series), _rank_along_epochs(grad_series)
    )


# Registry so scripts can iterate methods by name (and log them).
CORRELATIONS = {
    "pearson": pearson_correlation,
    "spearman": spearman_correlation,
}
