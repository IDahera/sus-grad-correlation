"""Plain-text (ASCII) renderings of the same grids the HTML report draws.

The HTML report is the interactive view; this module is the *diffable, greppable,
paste-into-the-thesis* view. Every heatmap becomes a block of characters with a
legend, so a run log or a markdown report carries the actual spatial pattern --
not just a number -- without any image files.

Two ramps, deliberately using disjoint glyph sets so the sign is never ambiguous:

    sequential  (suspiciousness / gradient, min..max)   " .:-=+*#%@"
    diverging   (correlation, fixed -1..+1)             negatives '@%;,',
                                                        positives ':=*#'

Grids wider/taller than ``max_cols``/``max_rows`` are block-averaged down (the
same idea as :func:`susgrad.viz.transform._downsample`, in 2-D), and the
reduction factor is stated in the caption so nothing silently pretends to be
per-neuron when it isn't.
"""

import math
from typing import Optional, Sequence, Tuple

import numpy as np

# --- ramps ---------------------------------------------------------------------

SEQ_RAMP = " .:-=+*#%@"

# (upper bound, glyph) -- a value belongs to the first bin whose bound it is below.
DIV_BINS: Tuple[Tuple[float, str], ...] = (
    (-0.75, "@"),   # strong negative
    (-0.50, "%"),
    (-0.25, ";"),
    (-0.05, ","),
    (0.05, "."),    # ~zero (includes the undefined-->0 neurons)
    (0.25, ":"),
    (0.50, "="),
    (0.75, "*"),
    (1.01, "#"),    # strong positive
)

NAN_GLYPH = " "

DIV_LEGEND = (
    "legend: '@' r<=-.75  '%' -.75..-.50  ';' -.50..-.25  ',' -.25..-.05  "
    "'.' -.05..+.05  ':' +.05..+.25  '=' +.25..+.50  '*' +.50..+.75  '#' r>=+.75"
)
SEQ_LEGEND = "legend: ' ' = min ... '@' = max (each map scaled to its own range)"


def _div_glyph(value: float) -> str:
    if value != value:  # NaN
        return NAN_GLYPH
    for bound, glyph in DIV_BINS:
        if value < bound:
            return glyph
    return DIV_BINS[-1][1]


def _seq_glyph(value: float, lo: float, hi: float) -> str:
    if value != value:
        return NAN_GLYPH
    span = (hi - lo) or 1.0
    t = min(1.0, max(0.0, (value - lo) / span))
    return SEQ_RAMP[min(len(SEQ_RAMP) - 1, int(t * len(SEQ_RAMP)))]


# --- 2-D downsampling -----------------------------------------------------------

def _block_reduce(grid: np.ndarray, max_rows: int, max_cols: int) -> Tuple[np.ndarray, int, int]:
    """Block-average *grid* so it fits in (max_rows, max_cols). NaNs are ignored."""
    rows, cols = grid.shape
    fr = max(1, math.ceil(rows / max_rows))
    fc = max(1, math.ceil(cols / max_cols))
    if fr == 1 and fc == 1:
        return grid, 1, 1

    pad_r = (-rows) % fr
    pad_c = (-cols) % fc
    if pad_r or pad_c:
        grid = np.pad(grid, ((0, pad_r), (0, pad_c)), constant_values=np.nan)
    r2, c2 = grid.shape
    blocks = grid.reshape(r2 // fr, fr, c2 // fc, fc)
    with np.errstate(invalid="ignore"):
        # An all-NaN block stays NaN (it is padding), which is what we want.
        reduced = np.nanmean(np.where(np.isnan(blocks), np.nan, blocks), axis=(1, 3))
    return reduced, fr, fc


# --- public API -----------------------------------------------------------------

def text_heatmap(
    grid,
    *,
    diverging: bool = False,
    max_rows: int = 48,
    max_cols: int = 72,
    indent: str = "    ",
) -> str:
    """Render a 2-D grid as a block of characters (with a trailing caption line).

    Args:
        grid: 2-D array-like; ``NaN`` cells are padding and render as spaces.
        diverging: True for correlation (fixed -1..+1 scale, signed glyphs);
            False for suspiciousness/gradient (scaled to the grid's own min/max).
        max_rows/max_cols: block-average anything larger down to this size.
        indent: prefix for every rendered line (keeps markdown code blocks tidy).
    """
    arr = np.asarray(grid, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"text_heatmap expects a 2-D grid, got shape {arr.shape}.")
    if arr.size == 0:
        return indent + "(empty)"

    shown, fr, fc = _block_reduce(arr, max_rows, max_cols)
    finite = shown[np.isfinite(shown)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 0.0

    lines = []
    for row in shown:
        if diverging:
            lines.append(indent + "".join(_div_glyph(v) for v in row))
        else:
            lines.append(indent + "".join(_seq_glyph(v, lo, hi) for v in row))

    caption = f"{arr.shape[0]}x{arr.shape[1]} cells"
    if fr > 1 or fc > 1:
        caption += f" -> shown {shown.shape[0]}x{shown.shape[1]} (block-averaged {fr}x{fc})"
    caption += f" · values [{_fmt(lo)}, {_fmt(hi)}]"
    lines.append(indent + f"({caption})")
    return "\n".join(lines)


def _fmt(v: float) -> str:
    if v != 0 and abs(v) < 1e-3:
        return f"{v:.2e}"
    return f"{v:.4f}"


def correlation_summary(values) -> dict:
    """Stats for a set of correlation values: n, mean, mean|r|, median, spreads.

    Extends :func:`susgrad.viz.transform.correlation_stats` with the *signed*
    mean and the negative share -- both matter here because a population of
    randomly-initialised instances can perfectly well be anti-correlated.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean": 0.0, "mean_abs": 0.0, "median": 0.0, "std": 0.0,
                "frac_strong": 0.0, "frac_pos": 0.0, "frac_neg": 0.0, "frac_zero": 0.0,
                "min": 0.0, "max": 0.0}
    av = np.abs(v)
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "mean_abs": float(av.mean()),
        "median": float(np.median(v)),
        "std": float(v.std()),
        "frac_strong": float((av > 0.5).mean()),
        "frac_pos": float((v > 0.05).mean()),
        "frac_neg": float((v < -0.05).mean()),
        "frac_zero": float((v == 0.0).mean()),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]],
                   align: Optional[Sequence[str]] = None) -> str:
    """A small markdown table builder (``align`` entries: 'l', 'r' or 'c')."""
    align = list(align) if align else ["l"] + ["r"] * (len(header) - 1)
    sep = {"l": " --- ", "r": " ---: ", "c": " :---: "}
    lines = [
        "| " + " | ".join(str(h) for h in header) + " |",
        "|" + "|".join(sep.get(a, " --- ") for a in align) + "|",
    ]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)
