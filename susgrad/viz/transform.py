"""Turn per-layer tensors into bounded, plottable views.

These functions deliberately enforce upper/lower bounds and raise
:class:`VisualizationError` when a model cannot be sensibly visualised (a layer
with no neurons, an enormous layer, non-finite values, or a bounded metric whose
values fall outside [0, 1]).

They also guarantee the data points are preserved: flattening keeps every value,
and :func:`align_layers` requires the gradient and suspiciousness tensors to have
the **same number of neurons** per layer.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# --- bounds --------------------------------------------------------------------
MAX_LAYERS = 64
MIN_NEURONS_PER_LAYER = 1
# Hard ceiling: above this a layer is refused (memory / truly unrenderable). Real
# conv/MLP layers are far below it; large layers are downsampled, not rejected.
MAX_NEURONS_PER_LAYER = 2_000_000
# Heatmaps larger than this many cells are block-averaged down to an overview grid.
# The per-neuron data (LayerView.values) is kept full, so the sorted comparison
# and all alignment checks still use every neuron.
HEATMAP_MAX_CELLS = 16_384  # up to a 128x128 overview
# Tolerance when checking that bounded metrics (ochiai/tarantula) stay in [0, 1].
BOUND_TOL = 1e-6


class VisualizationError(ValueError):
    """Raised when data cannot be properly visualised."""


@dataclass(frozen=True)
class LayerView:
    """A single layer's values, flattened and arranged for a heatmap.

    ``values`` is always the full, order-preserved per-neuron vector (used by the
    comparison plot and alignment checks). ``display_values`` is what the heatmap
    draws: identical to ``values`` for normal layers, or a block-averaged overview
    for very large layers (``downsampled`` is then True).
    """

    name: str
    values: np.ndarray          # 1-D, length == n_neurons (order preserved)
    original_shape: Tuple[int, ...]
    display_values: np.ndarray  # 1-D, length <= HEATMAP_MAX_CELLS
    grid_shape: Tuple[int, int]  # (rows, cols) for display_values
    downsampled: bool
    reduction: int              # neurons averaged per displayed cell (1 = none)

    @property
    def n_neurons(self) -> int:
        return int(self.values.size)

    def heatmap_grid(self) -> np.ndarray:
        """Display values padded with NaN into the (rows, cols) grid."""
        rows, cols = self.grid_shape
        grid = np.full(rows * cols, np.nan, dtype=float)
        grid[: self.display_values.size] = self.display_values
        return grid.reshape(rows, cols)


def _grid_shape(n: int) -> Tuple[int, int]:
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = max(1, math.ceil(n / cols))
    return rows, cols


def _downsample(flat: np.ndarray, max_cells: int) -> Tuple[np.ndarray, int]:
    """Block-average *flat* down to <= max_cells values. Returns (values, reduction)."""
    n = flat.size
    if n <= max_cells:
        return flat, 1
    reduction = math.ceil(n / max_cells)
    pad = (-n) % reduction
    if pad:
        flat = np.concatenate([flat, np.full(pad, np.nan)])
    # Mean per block, ignoring the NaN padding in the final block.
    blocks = flat.reshape(-1, reduction)
    reduced = np.nanmean(blocks, axis=1)
    return reduced, reduction


def to_layer_view(
    name: str,
    tensor: torch.Tensor,
    *,
    value_range: Optional[Tuple[float, float]] = None,
    max_neurons: int = MAX_NEURONS_PER_LAYER,
    heatmap_max_cells: int = HEATMAP_MAX_CELLS,
) -> LayerView:
    """Flatten *tensor* to a :class:`LayerView`, downsampling the heatmap if large.

    Args:
        value_range: if given (e.g. ``(0.0, 1.0)`` for ochiai/tarantula), values
            must lie within it (± tolerance) or a :class:`VisualizationError` is
            raised.
        max_neurons: hard ceiling; above it the layer is refused.
        heatmap_max_cells: layers wider than this are block-averaged for the
            heatmap only (the full data is kept in ``values``).
    """
    original_shape = tuple(tensor.shape)
    flat = tensor.detach().cpu().reshape(-1).to(torch.float64).numpy()
    n = flat.size

    if n < MIN_NEURONS_PER_LAYER:
        raise VisualizationError(f"Layer {name!r} has no neurons to visualise.")
    if n > max_neurons:
        raise VisualizationError(
            f"Layer {name!r} has {n} neurons (> {max_neurons}); too large even to "
            "downsample. Raise --max-neurons or sub-sample the layer."
        )
    if not np.all(np.isfinite(flat)):
        raise VisualizationError(f"Layer {name!r} contains non-finite values.")

    if value_range is not None:
        lo, hi = value_range
        if flat.min() < lo - BOUND_TOL or flat.max() > hi + BOUND_TOL:
            raise VisualizationError(
                f"Layer {name!r} values [{flat.min():.4g}, {flat.max():.4g}] fall "
                f"outside the expected range [{lo}, {hi}]."
            )

    display, reduction = _downsample(flat, heatmap_max_cells)
    return LayerView(
        name=name,
        values=flat,
        original_shape=original_shape,
        display_values=display,
        grid_shape=_grid_shape(display.size),
        downsampled=reduction > 1,
        reduction=reduction,
    )


def build_layer_views(
    mapping: Dict[str, torch.Tensor],
    *,
    value_range: Optional[Tuple[float, float]] = None,
    max_neurons: int = MAX_NEURONS_PER_LAYER,
) -> Dict[str, LayerView]:
    """Build views for every layer, enforcing the layer-count bound."""
    if len(mapping) == 0:
        raise VisualizationError("No layers to visualise.")
    if len(mapping) > MAX_LAYERS:
        raise VisualizationError(
            f"{len(mapping)} layers (> {MAX_LAYERS}); too many to visualise."
        )
    return {
        name: to_layer_view(name, tensor, value_range=value_range, max_neurons=max_neurons)
        for name, tensor in mapping.items()
    }


def align_layers(
    susp_by_layer: Dict[str, torch.Tensor],
    grad_by_layer: Dict[str, torch.Tensor],
) -> List[str]:
    """Return the shared layer order, asserting matching neuron counts.

    Raises :class:`VisualizationError` if the two mappings cover different layers
    or any layer has a different neuron count between the metrics.
    """
    susp_layers = list(susp_by_layer)
    if set(susp_layers) != set(grad_by_layer):
        raise VisualizationError(
            "Suspiciousness and gradient layers differ: "
            f"{sorted(susp_by_layer)} vs {sorted(grad_by_layer)}."
        )
    for name in susp_layers:
        n_s = susp_by_layer[name].numel()
        n_g = grad_by_layer[name].numel()
        if n_s != n_g:
            raise VisualizationError(
                f"Layer {name!r}: suspiciousness has {n_s} neurons but gradient "
                f"has {n_g}; they must match."
            )
    return susp_layers


@dataclass(frozen=True)
class Heatmap:
    """A 2-D grid ready for imshow, plus a human-readable description."""

    grid: np.ndarray       # 2-D
    info: str              # e.g. "channel 0/6 · 28×28" or "256 neurons · 16×16 grid"
    downsampled: bool
    reduction: int
    # What the heatmap's axes mean (conv: spatial position; dense: wrapped index).
    xlabel: str = ""
    ylabel: str = ""


def _pad_to_grid(values_1d: np.ndarray) -> np.ndarray:
    rows, cols = _grid_shape(values_1d.size)
    grid = np.full(rows * cols, np.nan, dtype=float)
    grid[: values_1d.size] = values_1d
    return grid.reshape(rows, cols)


def build_heatmap(
    name: str,
    tensor: torch.Tensor,
    *,
    channel: int = 0,
    value_range: Optional[Tuple[float, float]] = None,
    max_neurons: int = MAX_NEURONS_PER_LAYER,
    heatmap_max_cells: int = HEATMAP_MAX_CELLS,
) -> Heatmap:
    """Build a structure-preserving heatmap for one layer.

    * 3-D ``(C, H, W)`` (conv): the chosen *channel* is shown as its true ``H×W``
      map, preserving spatial structure. The channel index is clamped to the
      available range and reported in ``info``.
    * 2-D ``(H, W)``: shown directly.
    * 1-D ``(N,)`` (dense): packed into a near-square grid (block-averaged if very
      large), since dense neurons have no spatial layout.
    """
    arr = tensor.detach().cpu().to(torch.float64).numpy()
    flat = arr.reshape(-1)
    n = flat.size

    if n < MIN_NEURONS_PER_LAYER:
        raise VisualizationError(f"Layer {name!r} has no neurons to visualise.")
    if n > max_neurons:
        raise VisualizationError(
            f"Layer {name!r} has {n} neurons (> {max_neurons}); too large."
        )
    if not np.all(np.isfinite(flat)):
        raise VisualizationError(f"Layer {name!r} contains non-finite values.")
    if value_range is not None:
        lo, hi = value_range
        if flat.min() < lo - BOUND_TOL or flat.max() > hi + BOUND_TOL:
            raise VisualizationError(
                f"Layer {name!r} values [{flat.min():.4g}, {flat.max():.4g}] fall "
                f"outside the expected range [{lo}, {hi}]."
            )

    if arr.ndim >= 3:
        # Treat the first axis as channels; flatten any trailing axes into width.
        c_count = arr.shape[0]
        c = int(channel) % c_count
        plane = arr[c]
        if plane.ndim == 1:
            grid = _pad_to_grid(plane)
            xlabel, ylabel = "grid column (wrapped index)", "grid row"
        else:
            grid = plane.reshape(plane.shape[0], -1)
            xlabel, ylabel = "neuron x-position (width)", "neuron y-position (height)"
        h, w = grid.shape
        return Heatmap(grid, f"channel {c}/{c_count} · {h}×{w}", False, 1,
                       xlabel=xlabel, ylabel=ylabel)

    if arr.ndim == 2:
        h, w = arr.shape
        return Heatmap(arr, f"{h}×{w}", False, 1, xlabel="column", ylabel="row")

    # 1-D dense layer.
    disp, reduction = _downsample(flat, heatmap_max_cells)
    grid = _pad_to_grid(disp)
    rows, cols = grid.shape
    note = f" (downsampled ×{reduction})" if reduction > 1 else ""
    unit = "block" if reduction > 1 else "neuron"
    return Heatmap(
        grid, f"{n} neurons · {rows}×{cols} grid{note}", reduction > 1, reduction,
        xlabel=f"grid column ({unit} index = row×{cols} + column)", ylabel="grid row",
    )


def sorted_descending(values: np.ndarray) -> np.ndarray:
    """Return *values* sorted from largest to smallest (copy)."""
    return np.sort(values)[::-1]


def correlation_stats(values) -> dict:
    """Summary stats for a layer's correlation values (non-finite excluded).

    Returns ``n`` (finite count), ``mean_abs`` (typical coupling strength,
    direction-agnostic), ``median`` (signed central tendency), ``frac_strong``
    (share with ``|r| > 0.5``) and ``frac_zero`` (share exactly 0 — i.e. the
    "undefined → 0" neurons, which a signed mean would otherwise hide).
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean_abs": 0.0, "median": 0.0, "frac_strong": 0.0, "frac_zero": 0.0}
    av = np.abs(v)
    return {
        "n": int(v.size),
        "mean_abs": float(av.mean()),
        "median": float(np.median(v)),
        "frac_strong": float((av > 0.5).mean()),
        "frac_zero": float((v == 0.0).mean()),
    }


def sorted_comparison(
    susp_values: np.ndarray, grad_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort both arrays by descending suspiciousness.

    Returns ``(order, susp_sorted, grad_sorted)``. The number of points is
    preserved and the same permutation is applied to both arrays, so a neuron's
    susp/grad pair stays together.
    """
    if susp_values.size != grad_values.size:
        raise VisualizationError(
            f"susp ({susp_values.size}) and grad ({grad_values.size}) differ in length."
        )
    order = np.argsort(susp_values)[::-1]
    return order, susp_values[order], grad_values[order]
