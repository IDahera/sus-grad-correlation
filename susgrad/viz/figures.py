"""Standalone figure files (PNG/PDF) for inclusion in a LaTeX document.

The HTML report is for exploring; these are the files you ``\\includegraphics``.
They differ from the report's embedded figures in three ways that matter for
print:

* a **light** style (white background, dark text) instead of the report's dark
  theme -- a dark figure on a white page is unreadable in print;
* **vector PDF** alongside the PNG, so axis labels stay crisp at any zoom (the
  heatmap pixels themselves are raster either way);
* every figure is **self-labelled**: which epoch, which value range, which
  layer/metric/method -- so a figure pulled out of context still says what it is.

Two shapes are produced:

* :func:`save_heatmap` -- ONE panel (one epoch). Best for the thesis: LaTeX
  controls the layout, and each panel gets its own caption and ``\\ref``.
* :func:`save_heatmap_row` -- the captured epochs side by side in one file, on a
  SHARED colour scale. Convenient for drafts and slides, and honest because the
  scale is shared; the trade-off is that its inner labels are baked in at a fixed
  font size, so they will not match the surrounding document's typography.

Both are generated, and :func:`write_latex_index` emits a ready-to-use
``figures.tex`` with a subfigure macro for the single-panel route.
"""

import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from susgrad.utils.paths import ensure_dir  # noqa: E402

# Light, print-friendly style. Deliberately close to a LaTeX document's defaults:
# thin grey axes, no coloured background, small tick labels.
PAPER_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "#111111",
    "axes.labelcolor": "#333333",
    "axes.edgecolor": "#888888",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.titlecolor": "#111111",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 100,
    "pdf.fonttype": 42,   # embed TrueType, so the PDF text stays selectable
    "ps.fonttype": 42,
}

# Sequential for suspiciousness/gradient, diverging for correlation. Both are
# perceptually uniform and survive greyscale printing reasonably well.
CMAP_SEQ = "viridis"
CMAP_DIV = "coolwarm"

DEFAULT_FORMATS = ("png", "pdf")


# --- naming --------------------------------------------------------------------

def slug(text) -> str:
    """Filesystem-safe token: ``net.1`` -> ``net-1``, ``MLP × MNIST`` -> ``mlp-mnist``."""
    out = re.sub(r"[^0-9a-zA-Z]+", "-", str(text)).strip("-").lower()
    return out or "x"


def figure_stem(
    *,
    combo: str,
    kind: str,
    layer: Optional[str] = None,
    channel: Optional[int] = None,
    metric: Optional[str] = None,
    method: Optional[str] = None,
    instance: Optional[str] = None,
    epoch: Optional[int] = None,
    epochs: Optional[Sequence[int]] = None,
) -> str:
    """Build the filename stem, ordered so a directory listing groups usefully.

    ``<combo>__<layer>[__ch<NN>]__<kind>[__<metric>][__<method>][__<instance>]``
    ``__epoch-<NN>`` or ``__epochs-0-1-10``. Sorting the directory therefore
    walks combo -> layer -> channel -> kind -> metric -> method, which is exactly
    how you go looking for "the ochiai/pearson map of conv1".
    """
    parts = [slug(combo)]
    if layer is not None:
        parts.append(slug(layer))
    if channel is not None:
        parts.append(f"ch{int(channel):02d}")
    parts.append(slug(kind))
    if metric:
        parts.append(slug(metric))
    if method:
        parts.append(slug(method))
    if instance:
        parts.append(slug(instance))
    if epoch is not None:
        parts.append(f"epoch-{int(epoch):02d}")
    elif epochs:
        parts.append("epochs-" + "-".join(str(int(e)) for e in epochs))
    return "__".join(parts)


# --- shared drawing helpers ------------------------------------------------------

def _fmt(v: float) -> str:
    if v != 0 and abs(v) < 1e-3:
        return f"{v:.2e}"
    return f"{v:.3g}"


def range_label(grid) -> str:
    """``[min, max]`` of the finite cells -- the 'which values am I looking at' label."""
    arr = np.asarray(grid, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "[empty]"
    return f"[{_fmt(float(finite.min()))}, {_fmt(float(finite.max()))}]"


def _finite_range(grids) -> Tuple[float, float]:
    lo, hi = math.inf, -math.inf
    for grid in grids:
        arr = np.asarray(grid, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            lo = min(lo, float(finite.min()))
            hi = max(hi, float(finite.max()))
    if lo is math.inf:
        return 0.0, 0.0
    return lo, hi


def _save(fig, stem: Path, formats: Sequence[str], dpi: int) -> List[Path]:
    ensure_dir(stem.parent)
    written = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, bbox_inches="tight", dpi=dpi)
        written.append(path)
    plt.close(fig)
    return written


def _panel_size(grid) -> Tuple[float, float]:
    """Figure size that keeps cells roughly square without going extreme."""
    rows, cols = np.asarray(grid).shape
    aspect = cols / max(rows, 1)
    width = 2.6
    height = max(1.5, min(4.0, width / max(aspect, 0.35)))
    return width, height


# --- public API ------------------------------------------------------------------

def save_heatmap(
    stem: Path,
    grid,
    *,
    title: str,
    subtitle: str = "",
    diverging: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar_label: str = "",
    xlabel: str = "",
    ylabel: str = "",
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = 300,
) -> List[Path]:
    """Write ONE heatmap panel to ``<stem>.<fmt>`` for every requested format."""
    arr = np.asarray(grid, dtype=float)
    if diverging:
        vmin, vmax = (-1.0 if vmin is None else vmin), (1.0 if vmax is None else vmax)
    elif vmin is None or vmax is None:
        lo, hi = _finite_range([arr])
        vmin, vmax = (lo if vmin is None else vmin), (hi if vmax is None else vmax)

    with plt.rc_context(PAPER_STYLE):
        w, h = _panel_size(arr)
        fig, ax = plt.subplots(figsize=(w + 1.0, h + 0.7))
        im = ax.imshow(arr, cmap=CMAP_DIV if diverging else CMAP_SEQ,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{title}\n{subtitle}" if subtitle else title)
        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel or "")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        if cbar_label:
            cbar.set_label(cbar_label)
        return _save(fig, Path(stem), formats, dpi)


def save_heatmap_row(
    stem: Path,
    panels: Sequence[Tuple[str, object]],
    *,
    suptitle: str = "",
    diverging: bool = False,
    cbar_label: str = "",
    xlabel: str = "",
    ylabel: str = "",
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = 300,
) -> List[Path]:
    """Write several panels side by side, sharing ONE colour scale.

    Args:
        panels: ``[(label, grid), ...]`` -- typically one per captured epoch.
            Each panel is titled with its label and annotated underneath with its
            own value range, so a shared scale never hides what a panel contains.
    """
    if not panels:
        raise ValueError("save_heatmap_row needs at least one panel.")
    grids = [np.asarray(g, dtype=float) for _, g in panels]
    if diverging:
        vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = _finite_range(grids)

    with plt.rc_context(PAPER_STYLE):
        w, h = _panel_size(grids[0])
        fig, axes = plt.subplots(
            1, len(panels), figsize=(w * len(panels) + 1.2, h + 1.15), squeeze=False,
        )
        im = None
        for index, (ax, (label, grid)) in enumerate(zip(axes[0], panels)):
            arr = np.asarray(grid, dtype=float)
            im = ax.imshow(arr, cmap=CMAP_DIV if diverging else CMAP_SEQ,
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(label)
            # Only the per-panel range goes under each panel -- it is the one
            # number a shared colour scale cannot show. The (long) axis
            # description is stated once for the whole row instead of being
            # repeated under every panel, where the texts would collide.
            ax.set_xlabel(f"range {range_label(arr)}", fontsize=7)
            ax.set_ylabel(ylabel if index == 0 else "")
        if suptitle:
            fig.suptitle(suptitle, fontsize=9.5)
        if xlabel:
            fig.supxlabel(xlabel.replace("\n", " "), fontsize=7)
        cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, pad=0.02)
        scale_note = "shared scale −1…+1" if diverging else \
            f"shared scale {_fmt(vmin)}…{_fmt(vmax)}"
        cbar.set_label(f"{cbar_label} ({scale_note})" if cbar_label else scale_note)
        return _save(fig, Path(stem), formats, dpi)


#: A series spanning more than this many orders of magnitude is drawn on a
#: symmetric-log axis. D* routinely spans seven (median ~60, peak ~4e8): on a
#: linear axis the epoch-0 spike flattens every later epoch onto the baseline,
#: so the panel shows nothing at all.
LOG_SCALE_DECADES = 3.0


def _needs_log_scale(series: Iterable[Sequence[float]]) -> bool:
    values = np.concatenate([np.asarray(list(v), dtype=float).reshape(-1) for v in series])
    values = values[np.isfinite(values)]
    positive = values[values > 0]
    if positive.size == 0 or values.min() < 0:
        return False
    return float(positive.max() / positive.min()) > 10.0 ** LOG_SCALE_DECADES


def _apply_log_scale(ax, series) -> bool:
    """Switch *ax* to symlog if the data spans many decades. Returns whether it did."""
    if not _needs_log_scale(series):
        return False
    values = np.concatenate([np.asarray(list(v), dtype=float).reshape(-1) for v in series])
    positive = values[np.isfinite(values) & (values > 0)]
    # linthresh keeps the (legitimate) exact zeros visible instead of at -inf.
    ax.set_yscale("symlog", linthresh=max(float(positive.min()), 1e-30))
    return True


def draw_series_panel(
    ax,
    x,
    left_series: Dict[str, Sequence[float]],
    *,
    right_series: Optional[Dict[str, Sequence[float]]] = None,
    title: str = "",
    xlabel: str = "epoch",
    left_label: str = "",
    right_label: str = "",
    legend: bool = False,
    log_scale: str = "auto",
):
    """Draw one panel: left-axis series plus an optional right-axis reference.

    Suspiciousness metrics and gradients live on utterly different scales (ochiai
    is 0..1, D* runs into the millions, mean |gradient| is ~1e-5), so anything
    sharing one axis would flatten the smaller series onto the baseline. Each
    panel therefore carries ONE left-axis quantity, with the reference series
    (the gradient) dashed on its own right-hand axis, and switches to a log axis
    when the values span more than :data:`LOG_SCALE_DECADES` decades.

    No per-panel legend by default: the axis labels already name both series, and
    a legend box inside the axes lands on top of the curves. Callers that want
    one draw a single figure-level legend instead (see :func:`series_legend`).

    Returns ``{"left_log": bool, "right_log": bool}`` so a caller can say so in
    the axis label.
    """
    for name, values in left_series.items():
        ax.plot(x, values, marker="o", markersize=2.4, linewidth=1.3, label=name)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    # Epochs are whole numbers; matplotlib's default would offer 0.5 steps.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Headroom, so a peak never touches the frame (and any annotation fits).
    ax.margins(y=0.12)

    left_log = _apply_log_scale(ax, left_series.values()) if log_scale == "auto" else False
    ax.set_ylabel(f"{left_label} (log)" if left_log else left_label)
    handles, labels = ax.get_legend_handles_labels()

    right_log = False
    if right_series:
        ax_r = ax.twinx()
        for name, values in right_series.items():
            ax_r.plot(x, values, linestyle="--", linewidth=1.3, color="#333333", label=name)
        ax_r.margins(y=0.12)
        right_log = _apply_log_scale(ax_r, right_series.values()) if log_scale == "auto" else False
        ax_r.set_ylabel(f"{right_label} (log)" if right_log else right_label)
        h2, l2 = ax_r.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2

    if title:
        ax.set_title(title)
    if legend:
        ax.legend(handles, labels, fontsize=6.5, loc="best", framealpha=0.85)
    return {"left_log": left_log, "right_log": right_log, "handles": handles, "labels": labels}


def series_legend(fig, *, left_label: str = "suspiciousness (left axis)",
                  right_label: str = "mean |gradient| (right axis)") -> None:
    """One legend for the whole figure, below the panels -- never over the data.

    Deliberately explains the *line styles*, not the series names: each panel
    shows a different metric, so reusing one panel's handles would label every
    panel's solid line with the first panel's metric name. The metric itself is
    already on each panel's title and y-axis.
    """
    proxies = [Line2D([], [], color="#1f77b4", marker="o", markersize=3.2, linewidth=1.3)]
    labels = [left_label]
    if right_label:
        proxies.append(Line2D([], [], color="#333333", linestyle="--", linewidth=1.3))
        labels.append(right_label)
    fig.legend(proxies, labels, fontsize=7.5, loc="lower center",
               ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.0))


def save_metric_panels(
    stem: Path,
    x,
    left_by_name: Dict[str, Sequence[float]],
    *,
    right_series: Optional[Dict[str, Sequence[float]]] = None,
    suptitle: str = "",
    subtitle: str = "",
    xlabel: str = "epoch",
    right_label: str = "",
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = 300,
) -> List[Path]:
    """One panel per left-hand series (e.g. per suspiciousness metric), side by side.

    Every panel repeats the *right_series* (the gradient) so each metric can be
    read against it directly, each on a scale that suits it.
    """
    if not left_by_name:
        raise ValueError("save_metric_panels needs at least one series.")
    names = list(left_by_name)
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.4), squeeze=False)
        for ax, name in zip(axes[0], names):
            draw_series_panel(
                ax, x, {name: left_by_name[name]}, right_series=right_series,
                title=name, xlabel=xlabel, left_label=name, right_label=right_label,
            )
        if suptitle:
            fig.suptitle(f"{suptitle}\n{subtitle}" if subtitle else suptitle, fontsize=9.5)
        # Reserve the bottom strip BEFORE placing the legend, so it cannot land
        # on the x-axis labels.
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        series_legend(fig)
        return _save(fig, Path(stem), formats, dpi)


def new_grid_figure(nrows: int, ncols: int, *, figsize=None):
    """A blank (rows x cols) figure in the paper style, for multi-panel overviews."""
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (ncols * 5.2, nrows * 3.2),
                                 squeeze=False)
    return fig, axes


def save_figure(fig, stem: Path, formats: Sequence[str] = DEFAULT_FORMATS, dpi: int = 300,
                *, legend: bool = False) -> List[Path]:
    """Write an already-built figure to every requested format and close it.

    With ``legend=True`` a shared style legend is placed in a reserved strip at
    the bottom (see :func:`series_legend`).
    """
    with plt.rc_context(PAPER_STYLE):
        fig.tight_layout(rect=(0, 0.04, 1, 1) if legend else None)
        if legend:
            series_legend(fig)
    return _save(fig, Path(stem), formats, dpi)


# --- indexes --------------------------------------------------------------------

def write_manifest_csv(path: Path, records: Sequence[dict]) -> Path:
    """One row per generated figure, so the set is searchable without ``ls``."""
    path = Path(path)
    ensure_dir(path.parent)
    columns = ["combo", "layer", "channel", "kind", "metric", "method",
               "instance", "epochs", "panels", "file"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return path


_LATEX_PREAMBLE = r"""% Auto-generated by scripts/figures_ensemble.py -- do not edit by hand.
%
% Usage in your document:
%   \usepackage{graphicx}
%   \usepackage{subcaption}
%   \graphicspath{{<path to outputs/ensemble/figures/>}}
%   \input{figures.tex}
%
% Then place a single-panel figure with:
%   \susgradfig{FILE-STEM}{caption text}{fig:label}
% or the three captured epochs as one subfigure row (recommended for the thesis:
% each panel keeps its own caption and \ref, and the fonts match your document):
%   \susgradepochs{STEM-epoch-00}{STEM-epoch-01}{STEM-epoch-10}%
%     {overall caption}{fig:label}
%
% The pre-composed side-by-side files (…__epochs-0-1-10) are also here; they are
% handy for drafts and slides, but their labels are baked in at a fixed size.

\providecommand{\susgradfig}[3]{%
  \begin{figure}[htbp]
    \centering
    \includegraphics[width=0.55\linewidth]{#1}
    \caption{#2}
    \label{#3}
  \end{figure}%
}

\providecommand{\susgradepochs}[5]{%
  \begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.32\linewidth}
      \includegraphics[width=\linewidth]{#1}
      \caption{epoch 0 (untrained)}
    \end{subfigure}\hfill
    \begin{subfigure}[b]{0.32\linewidth}
      \includegraphics[width=\linewidth]{#2}
      \caption{after 1 epoch}
    \end{subfigure}\hfill
    \begin{subfigure}[b]{0.32\linewidth}
      \includegraphics[width=\linewidth]{#3}
      \caption{after 10 epochs}
    \end{subfigure}
    \caption{#4}
    \label{#5}
  \end{figure}%
}
"""


def write_latex_index(path: Path, records: Sequence[dict], *, root: Optional[Path] = None) -> Path:
    """Write ``figures.tex``: the two macros plus a commented list of every stem.

    The list is the point -- it is how you find the right ``\\susgradfig`` argument
    without leaving the editor.
    """
    path = Path(path)
    ensure_dir(path.parent)
    lines = [_LATEX_PREAMBLE, "", "% --- generated figures -------------------------------------------------"]

    by_combo: Dict[str, List[dict]] = {}
    for record in records:
        by_combo.setdefault(record["combo"], []).append(record)

    for combo in sorted(by_combo):
        lines.append(f"%\n% {combo}")
        for record in sorted(by_combo[combo], key=lambda r: r["file"]):
            file_path = Path(record["file"])
            stem = file_path.relative_to(root).with_suffix("") if root else file_path.with_suffix("")
            descr = " · ".join(
                str(record[key]) for key in ("layer", "kind", "metric", "method", "epochs")
                if record.get(key) not in (None, "")
            )
            lines.append(f"%   {stem}   ({descr})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
