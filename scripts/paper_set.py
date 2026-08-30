#!/usr/bin/env python3
"""Curate a small, documented figure set for the write-up.

The full export is ~640 files; a paper needs a dozen. This script SELECTS from
what ``figures_ensemble.py`` / ``plot_trajectory.py`` already produced -- it
draws nothing itself -- and groups the result into numbered folders, each with a
``README.md`` whose "Caption" line is written to be pasted straight into a LaTeX
``\\caption{}``.

    outputs/paper/
      01_method_what-is-correlated/   grad + susp + corr, one layer, epochs 0 & 1
      02_dense_first-layer/           corr epoch row, first dense layer, both MLPs
      03_conv_layers/                 corr epoch row, both conv layers, both LeNets
      04_value_ranges/                the condensed statistics table
      05_trajectories/                experiment 2: two figures + one condensed table
                                      showing where the values settle
      README.md                       what each folder is for

Selection happens through the figure manifest, so a figure that is not in the
manifest is reported as missing rather than silently skipped.

Note on pooling layers: LeNet-5 does contain ``MaxPool2d``, but suspiciousness
and gradients are only captured for ``Linear``/``Conv2d`` modules (see
``get_activations_with_hooks``), so there is nothing to plot for a pooling
stage. Folder 03 therefore shows the two CONVOLUTION stages instead.

Run AFTER figures-ensemble (+ value-stats, plot-trajectory):

    python scripts/paper_set.py
    make paper-figures
"""

import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from scripts._cli import (
    field,
    long_field,
    section,
    setup_logging,
)
from susgrad.utils import (
    ENSEMBLE_DIR,
    ENSEMBLE_FIGURES_DIR,
    OUTPUTS_DIR,
    TRAJECTORY_DIR,
    TRAJECTORY_FIGURES_DIR,
    ensure_dir,
)
from susgrad.registry import COMBINATIONS_BY_KEY, ENSEMBLE_COMBO_KEYS
from susgrad.sbfl import CORE_METRIC_NAMES
from susgrad.stats import ALL_LAYERS
from susgrad.viz.textmap import markdown_table

# Value-stats columns worth printing. min/p05/p95/std stay in the full CSV --
# on a paper page they add width without adding an argument.
PAPER_STAT_COLUMNS = ("median", "mean", "max", "frac_zero")


def _load_manifest(figures_dir: Path):
    path = figures_dir / "manifest.csv"
    if not path.exists():
        raise click.ClickException(
            f"No figure manifest at {path}. Run `make figures-ensemble` first."
        )
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _select(manifest, *, combo, layer, kind, epochs, metric=None, method=None, panels=None):
    """Manifest rows matching every given field ('epochs' is the '0+1+10' string)."""
    out = []
    for row in manifest:
        if row["combo"] != combo or row["layer"] != layer or row["kind"] != kind:
            continue
        if row["epochs"] != epochs:
            continue
        if metric is not None and row["metric"] != metric:
            continue
        if method is not None and row["method"] != method:
            continue
        if panels is not None and int(row["panels"]) != panels:
            continue
        out.append(row)
    return out


def _copy(row, target_dir: Path, suffix: str, log) -> Path:
    """Copy one manifest entry's figure (in *suffix* format) into *target_dir*."""
    source = Path(row["file"]).with_suffix(f".{suffix}")
    if not source.exists():
        log.warning("  missing: %s", source)
        return None
    target = ensure_dir(target_dir) / source.name
    shutil.copy2(source, target)
    return target


def _readme(folder: Path, title: str, purpose: str, caption: str, entries, notes=(),
            caption_label: str = "", extra_captions=()) -> Path:
    """Write the folder README: what it shows, a ready caption, and the file list.

    A folder holding more than one figure passes *extra_captions* as
    ``[(label, caption), ...]`` (and labels the first one via *caption_label*):
    each figure needs its OWN caption, since a shared one would have to be vague
    enough to cover both -- exactly what a caption must not be.
    """
    lines = [
        f"# {title}",
        "",
        "**What this shows.** " + purpose,
        "",
        (f"**Caption — {caption_label}** (paste into `\\caption{{}}`)." if caption_label
         else "**Caption (paste into `\\caption{}`).**"),
        "",
        "> " + caption,
        "",
    ]
    for label, text in extra_captions:
        lines += [f"**Caption — {label}.**", "", "> " + text, ""]
    lines += [
        "**Files.**",
        "",
    ]
    lines += [f"- `{name}` — {descr}" for name, descr in entries]
    if notes:
        lines += ["", "**Notes.**", ""] + [f"- {n}" for n in notes]
    lines.append("")
    path = folder / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --- folder 05: the condensed trajectory results table ---------------------------

#: Rows of the paper table, in the order a reader should meet them.
TRAJECTORY_TABLE_ORDER = ("mlp_mnist", "mlp_fmnist", "lenet_mnist", "lenet_fmnist")


def _read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _condensed_trajectory(trajectory_dir: Path, metrics):
    """ONE table carrying the whole experiment-2 claim, and nothing else.

    Six numbers per case, each answering one question a reader will ask:

        settled level   where the run actually ended up  (no model)
        asymptote a     where the fit says it is heading (model)
        tau             how fast it got there
        R2 free / R2=0  how the two exponentials compare
        dAIC if a=0     what pinning the limit to zero costs

    Everything else the pipeline computes -- BIC, held-out error, residual
    diagnostics, per-epoch values -- stays in outputs/trajectory/. It is method
    detail, and in this table it would bury the result.
    """
    def table(name):
        return _read_csv(trajectory_dir / name)

    fits = {(r["combo"], r["kind"]): r for r in table("population_fits.csv")}
    windows = table("population_windows.csv")
    comparison = table("population_model_comparison.csv")
    if not fits or not windows:
        return None, None

    # The LAST window per case: the level the run actually ended at.
    last = {}
    for row in windows:
        key = (row["combo"], row["kind"])
        if key not in last or int(row["window"]) > int(last[key]["window"]):
            last[key] = row
    scored = {(r["combo"], r["kind"], r["model"]): r for r in comparison}

    header = ["model × dataset", "metric", "settled level (last window)", "fitted a",
              "95% CI for a", "τ (epochs)", "R² (a free)", "R² (a = 0)", "ΔAIC if a = 0"]
    table_rows, csv_rows = [], []
    for combo_key in TRAJECTORY_TABLE_ORDER:
        for metric in metrics:
            fit = fits.get((combo_key, metric))
            window = last.get((combo_key, metric))
            if not fit or not window:
                continue
            free = scored.get((combo_key, metric, "exp"), {})
            pinned = scored.get((combo_key, metric, "exp0"), {})
            label = COMBINATIONS_BY_KEY[combo_key].label if combo_key in COMBINATIONS_BY_KEY \
                else combo_key
            level = f"{float(window['mean']):.4g} ± {float(window['std']):.2g}"
            # Four significant figures: the printed table has to be readable, and
            # the bootstrap interval does not support more than that anyway.
            ci = (f"[{float(fit['asymptote_ci_lo']):.4g}, {float(fit['asymptote_ci_hi']):.4g}]"
                  if fit.get("asymptote_ci_lo") else "—")
            table_rows.append([
                label, f"`{metric}`", level, f"{float(fit['asymptote']):.4g}", ci,
                f"{float(fit['tau']):.4g}", free.get("r2", "—"), pinned.get("r2", "—"),
                pinned.get("delta_aic", "—"),
            ])
            csv_rows.append({
                "combo": combo_key, "metric": metric,
                "settled_level_mean": window["mean"], "settled_level_std": window["std"],
                "asymptote_a": fit["asymptote"],
                "asymptote_ci_lo": fit.get("asymptote_ci_lo", ""),
                "asymptote_ci_hi": fit.get("asymptote_ci_hi", ""),
                "tau_epochs": fit["tau"],
                "r2_a_free": free.get("r2", ""), "r2_a_zero": pinned.get("r2", ""),
                "delta_aic_a_zero": pinned.get("delta_aic", ""),
            })
    if not table_rows:
        return None, None
    return markdown_table(header, table_rows), csv_rows


# --- folder 04: the condensed statistics table ----------------------------------

def _condensed_stats(stats_csv: Path, combos, layer, epochs):
    """Value ranges per model: rows = metric, columns = mean/median/max + zero share.

    The full table is combo x layer x metric x epoch x 9 statistics -- hundreds of
    lines. What an argument actually needs is two things: that the metrics live on
    incomparable scales, and that their level COLLAPSES between epoch 0 and 10.
    Both fit in one small table per model.

    **Which layer.** Layer names are architecture-specific (``net.1`` vs
    ``conv1``), so a single named layer cannot serve every model. The default is
    therefore the pooled ``layer = all`` row -- every neuron of that model -- which
    is the one row that means the same thing across architectures. Pass *layer* to
    pin a specific one; models that do not have it fall back to the pooled row, and
    the layer actually used is reported per section so the table never hides it.

    Returns ``([(label, combo, layer_used, table)], csv_rows)``.
    """
    with stats_csv.open(encoding="utf-8") as fh:
        all_rows = list(csv.DictReader(fh))
    if not all_rows:
        return None, None

    if isinstance(combos, str):
        combos = [combos]
    present = [c for c in ENSEMBLE_COMBO_KEYS if any(r["combo"] == c for r in all_rows)]
    wanted = [c for c in (combos or present) if c in present]

    sections, csv_rows = [], []
    for combo in wanted:
        mine = [r for r in all_rows if r["combo"] == combo]
        layers = {r["layer"] for r in mine}
        chosen = layer if layer in layers else (ALL_LAYERS if ALL_LAYERS in layers else None)
        if chosen is None:
            continue
        table, rows = _stats_table(mine, combo, chosen, epochs)
        if table:
            label = COMBINATIONS_BY_KEY[combo].label if combo in COMBINATIONS_BY_KEY else combo
            sections.append((label, combo, chosen, table))
            csv_rows += rows
    if not sections:
        return None, None
    return sections, csv_rows


def _stats_table(model_rows, combo: str, layer: str, epochs):
    """One model's value-range table (see :func:`_condensed_stats`)."""
    rows = [r for r in model_rows if r["layer"] == layer]
    by_kind = {}
    for row in rows:
        if row["epoch"] in {str(e) for e in epochs}:
            by_kind.setdefault(row["kind"], {})[int(row["epoch"])] = row
    if not by_kind:
        return None, None

    # `min` is deliberately absent: it is exactly 0 in ~70% of rows (inactive
    # neurons), so it would be a column of zeros -- `zeros` reports that fact
    # properly. mean AND median are both kept: they agree for the bounded metrics
    # and diverge ~6x for D*, which is what exposes its outlier-driven skew.
    header = ["metric"] + [f"mean @ e{e}" for e in epochs] + \
        [f"median @ e{e}" for e in epochs] + [f"max @ e{e}" for e in epochs] + \
        [f"zeros @ e{epochs[-1]}"]
    table_rows, csv_rows = [], []
    for kind, per_epoch in by_kind.items():
        if not all(e in per_epoch for e in epochs):
            continue
        table_rows.append(
            [f"`{kind}`"]
            + [f"{float(per_epoch[e]['mean']):.4g}" for e in epochs]
            + [f"{float(per_epoch[e]['median']):.4g}" for e in epochs]
            + [f"{float(per_epoch[e]['max']):.4g}" for e in epochs]
            + [f"{100 * float(per_epoch[epochs[-1]]['frac_zero']):.1f}%"]
        )
        csv_rows.append({
            "combo": combo, "layer": layer, "kind": kind,
            **{f"{stat}_e{e}": per_epoch[e][stat]
               for e in epochs for stat in PAPER_STAT_COLUMNS},
        })
    return markdown_table(header, table_rows), csv_rows


def _condensed_correlation(stats_csv: Path, combos, epochs, metrics, methods):
    """Rows = layer x metric, columns = mean |r| per method per epoch, PER MODEL.

    THE results table: unlike the value ranges (which describe the inputs), this
    describes the correlation itself. Both correlation methods are shown side by
    side -- Pearson as the reference, Spearman as the headline, because D* is
    exponentially scaled and Pearson then measures its non-linearity rather than
    the strength of the relationship.

    Every model gets its own table rather than one combined block: the layer
    names differ between architectures (`net.1` vs `conv1`), so a single table
    would either repeat the model on every row or silently mix layers that are
    not comparable. Returns ``([(label, table), ...], csv_rows)`` -- one table per
    model for reading, one flat CSV carrying the ``combo`` column for machines.
    """
    with stats_csv.open(encoding="utf-8") as fh:
        all_rows = list(csv.DictReader(fh))
    if not all_rows:
        return None, None

    # A bare string would iterate into characters and match nothing, returning
    # "no data" for what is actually a valid single-model request.
    if isinstance(combos, str):
        combos = [combos]
    present = [c for c in ENSEMBLE_COMBO_KEYS if any(r["combo"] == c for r in all_rows)]
    wanted = [c for c in (combos or present) if c in present]

    sections, csv_rows = [], []
    for combo in wanted:
        table, rows = _correlation_table(all_rows, combo, epochs, metrics, methods)
        if table:
            label = COMBINATIONS_BY_KEY[combo].label if combo in COMBINATIONS_BY_KEY else combo
            sections.append((label, combo, table))
            csv_rows += rows
    if not sections:
        return None, None
    return sections, csv_rows


def _correlation_table(all_rows, combo: str, epochs, metrics, methods):
    """One model's correlation table (see :func:`_condensed_correlation`)."""
    rows = [r for r in all_rows if r["combo"] == combo]
    if not rows:
        return None, None

    index = {(r["layer"], r["metric"], r["method"], int(r["epoch"])): r for r in rows}
    layers = sorted({r["layer"] for r in rows}, key=lambda l: [
        r["layer"] for r in rows].index(l))
    metrics = [m for m in metrics if any(r["metric"] == m for r in rows)]
    methods = [m for m in methods if any(r["method"] == m for r in rows)]

    header = ["layer", "metric"] + [f"{me} \\|r\\| @ e{e}" for me in methods for e in epochs]
    table_rows, csv_rows = [], []
    for layer in layers:
        for metric in metrics:
            cells, record = [], {"combo": combo, "layer": layer, "metric": metric}
            complete = True
            for method in methods:
                for epoch in epochs:
                    row = index.get((layer, metric, method, epoch))
                    if row is None:
                        complete = False
                        break
                    cells.append(f"{float(row['mean_abs_r']):.3f}")
                    record[f"{method}_mean_abs_r_e{epoch}"] = row["mean_abs_r"]
                    record[f"{method}_mean_r_e{epoch}"] = row["mean_r"]
            if complete:
                table_rows.append([f"`{layer}`", f"`{metric}`"] + cells)
                csv_rows.append(record)
    return markdown_table(header, table_rows), csv_rows


@click.command()
@click.option("--metric", default="ochiai", show_default=True,
              help="Suspiciousness metric for the curated figures.")
@click.option("--method", default="spearman", show_default=True,
              help="Correlation method for the curated figures.")
@click.option("--epochs", default="0,1,10", show_default=True,
              help="The epoch row shown in folders 02/03 (must match the export).")
@click.option("--method-epochs", default="0,1", show_default=True,
              help="Folder 01: which epochs to contrast. A SHARED-scale row for these epochs "
                   "is preferred over the per-epoch singles, whose independent colour scales "
                   "make a four-fold drop in level look like no change at all.")
@click.option("--format", "suffix", default="png", show_default=True,
              type=click.Choice(["png", "pdf"]), help="Which rendering to copy.")
@click.option("--dense-layer", default="net.1", show_default=True,
              help="First dense layer of the MLPs.")
@click.option("--conv-layers", default="conv1,conv2", show_default=True,
              help="LeNet convolution stages (there are no pooling activations to plot).")
@click.option("--stats-combos", default=None,
              help="Models in the VALUE-RANGE table, comma-separated. Default: every model "
                   "present in value_stats.csv, in registry order.")
@click.option("--stats-layer", default=None,
              help="Layer for the value-range table. Default: the pooled 'all' row (every "
                   "neuron of the model), which is comparable across architectures. A pinned "
                   "layer that a model does not have falls back to the pooled row.")
@click.option("--correlation-combos", default=None,
              help="Models in the correlation summary, comma-separated. Default: every model "
                   "present in correlation_stats.csv, in registry order.")
@click.option("--stats-methods", default="spearman,pearson", show_default=True,
              help="Correlation methods in the condensed correlation table "
                   "(Spearman as the headline, Pearson as the reference).")
@click.option("--instance", default=None,
              help="Folder 01: pin the sample instance (e.g. inst044). Default: the "
                   "lowest-numbered instance that has BOTH a suspiciousness and a gradient "
                   "figure -- they must be the SAME instance to be comparable.")
@click.option("--figures-dir", default=None, help="Where the full export lives.")
@click.option("--out-dir", default=None, help="Where to write the curated set.")
def main(metric, method, epochs, method_epochs, suffix, dense_layer, conv_layers,
         stats_combos, stats_layer, correlation_combos, stats_methods, instance,
         figures_dir, out_dir):
    log, logfile = setup_logging("paper_set")
    figures_dir = Path(figures_dir) if figures_dir else ENSEMBLE_FIGURES_DIR
    out_base = Path(out_dir) if out_dir else OUTPUTS_DIR / "paper"
    traj_figures = TRAJECTORY_FIGURES_DIR
    epoch_list = [int(e) for e in epochs.split(",") if e.strip()]
    method_epoch_list = [int(e) for e in method_epochs.split(",") if e.strip()]
    method_epochs_key = "+".join(str(e) for e in method_epoch_list)
    epochs_key = "+".join(str(e) for e in epoch_list)
    conv_list = [c.strip() for c in conv_layers.split(",") if c.strip()]

    manifest = _load_manifest(figures_dir)
    if out_base.exists():
        shutil.rmtree(out_base)
    ensure_dir(out_base)

    section(log, "Plan")
    field(log, "Metric / method", f"{metric} / {method}")
    field(log, "Epoch row", epochs_key)
    field(log, "Format", suffix)
    long_field(log, "Source", figures_dir)

    copied_total = 0

    # --- 01: what is actually being correlated -------------------------------
    section(log, "01 — method figure")
    folder = ensure_dir(out_base / "01_method_what-is-correlated")
    entries = []
    kinds = (("susp", f"{metric} suspiciousness"),
             ("grad", "mean |gradient|"),
             ("corr", f"{method}({metric}, gradient)"))

    # The suspiciousness and gradient panels MUST come from the same instance --
    # side by side they are read as one model's two quantities. Successive
    # exports pick their sample instance at random, so the manifest can hold
    # several; settle on one before selecting anything.
    def _instances(kind, metric_arg):
        return {row["instance"] for row in
                _select(manifest, combo="mlp_mnist", layer=dense_layer, kind=kind,
                        epochs=method_epochs_key, metric=metric_arg, method="")
                if row["instance"]}

    shared = _instances("susp", metric) & _instances("grad", "")
    sample_instance = instance if instance in shared else (sorted(shared)[0] if shared else None)
    if sample_instance:
        field(log, "Sample instance", f"{sample_instance} (of {len(shared)} available)")

    for kind, descr in kinds:
        metric_arg = metric if kind != "grad" else ""
        method_arg = method if kind == "corr" else ""
        # Preferred: ONE file holding the epochs side by side on a SHARED colour
        # scale. The per-epoch singles are each normalised to their own range, so
        # placing two of them next to each other hides exactly the change the
        # figure is meant to show.
        rows = _select(manifest, combo="mlp_mnist", layer=dense_layer, kind=kind,
                       epochs=method_epochs_key, metric=metric_arg, method=method_arg)
        if kind in ("susp", "grad") and sample_instance:
            rows = [r for r in rows if r["instance"] == sample_instance]
        if rows:
            for row in rows:
                path = _copy(row, folder, suffix, log)
                if path:
                    entries.append((path.name,
                                    f"{descr}, epochs {', '.join(map(str, method_epoch_list))} "
                                    "(shared colour scale)"))
                    copied_total += 1
            continue
        log.warning("  no shared-scale row for %s (epochs %s) -- falling back to singles; "
                    "run figures_ensemble.py --epochs %s to get it",
                    kind, method_epochs_key, ",".join(map(str, method_epoch_list)))
        for epoch in method_epoch_list:
            singles = _select(manifest, combo="mlp_mnist", layer=dense_layer, kind=kind,
                              epochs=str(epoch), panels=1,
                              metric=metric_arg, method=method_arg)
            if kind in ("susp", "grad") and sample_instance:
                singles = [r for r in singles if r["instance"] == sample_instance]
            for row in singles:
                path = _copy(row, folder, suffix, log)
                if path:
                    entries.append((path.name, f"{descr}, epoch {epoch} (own colour scale)"))
                    copied_total += 1
    _readme(
        folder,
        "01 — What is being correlated (method figure)",
        f"For a single hidden layer (`{dense_layer}` of the dense MLP on MNIST), the two "
        "quantities that enter the correlation and their result, at the untrained model "
        "(epoch 0) and after one epoch. The suspiciousness and gradient maps are ONE "
        "randomly chosen instance of the population; the correlation map is computed "
        "across all 100 instances.",
        f"The two quantities entering the correlation and their result, for the first hidden "
        f"layer of the dense MLP on MNIST: per-neuron {metric} suspiciousness, mean "
        f"|gradient|, and their {method} correlation across 100 independently initialised "
        f"models. Each figure contrasts the untrained model with the model after one epoch on "
        f"a shared colour scale. Each cell is one neuron; the dense layer has no spatial "
        f"structure, so neurons are wrapped row-major into a near-square grid. Suspiciousness "
        f"and gradient come from a single, randomly selected instance and are illustrative; "
        f"the correlation is the population result.",
        entries,
        notes=[
            "Belongs in the METHOD section, not the results: it explains the inputs, it is "
            "not itself a finding.",
            "The suspiciousness and gradient panels come from the SAME instance — check the "
            "instance id in the filenames if you rebuild the set.",
            "The suspiciousness/gradient files share ONE colour scale across the epochs, so "
            "the drop in level is visible; each panel still states its own range underneath. "
            "The correlation file is on the fixed −1…+1 scale.",
            "Do NOT build this comparison from the per-epoch single files: each of those is "
            "normalised to its own range, so epoch 0 (ochiai up to 0.93) and epoch 1 (up to "
            "0.21) look equally bright despite a four-fold difference.",
        ],
    )
    field(log, "Files", len(entries))

    # --- 02: the dense first layer, both MLPs --------------------------------
    section(log, "02 — dense first layer")
    folder = ensure_dir(out_base / "02_dense_first-layer")
    entries = []
    for combo, dataset in (("mlp_mnist", "MNIST"), ("mlp_fmnist", "Fashion-MNIST")):
        for row in _select(manifest, combo=combo, layer=dense_layer, kind="corr",
                           epochs=epochs_key, metric=metric, method=method):
            path = _copy(row, folder, suffix, log)
            if path:
                entries.append((path.name, f"dense MLP on {dataset}, layer `{dense_layer}`"))
                copied_total += 1
    _readme(
        folder,
        "02 — Correlation over training, first dense layer",
        f"The across-instance correlation for the first hidden layer of the dense MLP, on "
        f"both datasets, with epochs {', '.join(map(str, epoch_list))} side by side in one "
        "file on a shared colour scale.",
        f"Per-neuron {method} correlation between {metric} suspiciousness and mean |gradient|, "
        f"computed across 100 independently initialised models, for the first hidden layer of "
        f"the dense MLP on MNIST and Fashion-MNIST. Panels show the untrained model (epoch 0), "
        f"after one epoch, and after ten. The colour scale is fixed to −1…+1, so panels are "
        f"directly comparable; each panel states its own value range underneath.",
        entries,
        notes=[
            "This is the main result figure: the coupling is strongest in the UNTRAINED "
            "model and weakens once training starts.",
            "Both datasets are shown so the effect is visibly not an artefact of MNIST.",
        ],
    )
    field(log, "Files", len(entries))

    # --- 03: the convolution stages, both LeNets ------------------------------
    section(log, "03 — convolution stages")
    folder = ensure_dir(out_base / "03_conv_layers")
    entries = []
    for combo, dataset in (("lenet_mnist", "MNIST"), ("lenet_fmnist", "Fashion-MNIST")):
        for layer in conv_list:
            for row in _select(manifest, combo=combo, layer=layer, kind="corr",
                               epochs=epochs_key, metric=metric, method=method):
                path = _copy(row, folder, suffix, log)
                if path:
                    entries.append((path.name, f"LeNet-5 on {dataset}, layer `{layer}`"))
                    copied_total += 1
    _readme(
        folder,
        "03 — Correlation over training, convolution stages",
        f"The same view as folder 02, but for LeNet-5's two convolution stages "
        f"({', '.join(f'`{c}`' for c in conv_list)}) on both datasets. Conv neurons keep "
        "their spatial layout, so each pixel is a neuron at its true position in the feature "
        "map (one channel is shown, stated in the title).",
        f"Per-neuron {method} correlation between {metric} suspiciousness and mean |gradient| "
        f"across 100 independently initialised LeNet-5 models, for both convolution stages on "
        f"MNIST and Fashion-MNIST, at epochs {', '.join(map(str, epoch_list))}. Each pixel is "
        f"one neuron at its position in the feature map; the colour scale is fixed to −1…+1.",
        entries,
        notes=[
            "LeNet-5 pools between the convolutions, but pooling has no parameters and no "
            "hook, so there are no pooling activations to plot — the two CONVOLUTION stages "
            "are shown instead.",
            "Conv layers correlate markedly weaker than the dense layers of the same network "
            "(mean |r| ≈ 0.55–0.74 vs. 0.84–0.94) — that contrast is the point of this "
            "figure.",
            "The maps show spatial structure (a brighter border, a weaker centre). Before "
            "claiming that as a finding, check whether border neurons simply see mostly "
            "background: near-constant activations correlate easily.",
        ],
    )
    field(log, "Files", len(entries))

    # --- 04: the condensed statistics table ----------------------------------
    section(log, "04 — value ranges")
    folder = ensure_dir(out_base / "04_value_ranges")
    entries = []
    method_list = [m.strip() for m in stats_methods.split(",") if m.strip()]
    correlation_combos = ([c.strip() for c in correlation_combos.split(",") if c.strip()]
                          if correlation_combos else None)
    stats_combos = ([c.strip() for c in stats_combos.split(",") if c.strip()]
                    if stats_combos else None)
    stats_layer = stats_layer or ALL_LAYERS

    # (a) THE results table: the correlation itself, both methods side by side.
    corr_csv = ENSEMBLE_DIR / "correlation_stats.csv"
    if corr_csv.exists():
        sections, rows = _condensed_correlation(corr_csv, correlation_combos, epoch_list,
                                                list(CORE_METRIC_NAMES), method_list)
        if sections:
            body = [
                "# Correlation summary — all models",
                "",
                "Mean |r| between suspiciousness and mean |gradient|, per neuron and across "
                "100 independently initialised models, for every model/dataset pair. One table "
                "per model, because the layer names are architecture-specific and are not "
                "comparable across architectures.",
                "",
            ]
            for label, combo_key, table in sections:
                body += [f"## {label} (`{combo_key}`)", "", table, ""]
            (folder / "correlation_summary.md").write_text("\n".join(body), encoding="utf-8")
            with (folder / "correlation_summary.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            entries += [("correlation_summary.md", "THE results table — mean |r| per layer, "
                                                   f"metric and epoch, Spearman and Pearson, for "
                                                   f"all {len(sections)} models"),
                        ("correlation_summary.csv", "the same, flat, plus the signed mean r")]
            copied_total += 2
            field(log, "Correlation summary", f"{len(sections)} model(s), {len(rows)} rows")
    else:
        log.warning("  %s not found -- run `make correlate-ensemble` first.", corr_csv)

    # (b) supporting: the ranges of the values that went INTO the correlation.
    stats_csv = ENSEMBLE_DIR / "value_stats.csv"
    if stats_csv.exists():
        sections, rows = _condensed_stats(stats_csv, stats_combos, stats_layer, epoch_list)
        if sections:
            body = [
                "# Value ranges — all models",
                "",
                "These are the RAW suspiciousness and gradient values, **not** correlations. "
                "Unless a layer is pinned, each table pools every neuron of that model "
                f"(`layer = {ALL_LAYERS}`), which is the one row that means the same thing "
                "across architectures — layer names differ between the dense and the "
                "convolutional models.",
                "",
            ]
            for label, combo_key, layer_used, table in sections:
                body += [f"## {label} (`{combo_key}`), layer `{layer_used}`", "", table, ""]
            (folder / "value_ranges.md").write_text("\n".join(body), encoding="utf-8")
            with (folder / "value_ranges.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            entries += [("value_ranges.md", "supporting — the raw value ranges per metric, "
                                            f"for all {len(sections)} models"),
                        ("value_ranges.csv", "the same numbers, flat and machine-readable")]
            copied_total += 2
            field(log, "Value ranges", f"{len(sections)} model(s), "
                                       f"layer(s) {sorted({l for _, _, l, _ in sections})}")
        else:
            log.warning("  no value-range rows for the requested models/layer. If you expected "
                        "the pooled '%s' rows, regenerate with "
                        "`python scripts/value_stats.py --only ensemble`.", ALL_LAYERS)
    else:
        log.warning("  %s not found -- run `make value-stats` first.", stats_csv)
    _readme(
        folder,
        "04 — Tables (condensed)",
        f"Two tables. **`correlation_summary`** is the results table: mean |r| per layer, "
        f"metric and epoch, with Spearman and Pearson side by side, **for every model/dataset "
        f"pair** — one sub-table per model, since layer names differ between architectures. "
        f"**`value_ranges`** is supporting material: the RAW value ranges of the metrics and "
        f"the gradient (not correlations), **for every model**, pooled over all of that "
        f"model's neurons (`layer = {ALL_LAYERS}`) unless a layer is pinned — condensed from "
        "the full CSV.",
        f"Mean |r| between each suspiciousness metric and the mean |gradient|, per neuron and "
        f"across 100 independently initialised models, reported per model, layer and measured "
        f"epoch for both correlation coefficients. Spearman is the headline: D* is "
        f"exponentially scaled, so Pearson measures its non-linearity rather than the strength "
        f"of the relationship.",
        entries,
        notes=[
            "The correlation summary covers ALL models; if you quote a single number in the "
            "text, say which model and layer it came from — the dense and convolutional "
            "architectures differ substantially (mean |r| ~0.84-0.94 for dense layers vs. "
            "~0.55-0.74 for conv layers).",
            "`value_ranges` describes the INPUTS, not the correlation — do not read its "
            "columns as coupling strength. The correlation numbers are in "
            "`correlation_summary`.",
            f"Both tables now cover all four models. `value_ranges` pools every neuron of a "
            f"model (`layer = {ALL_LAYERS}`) so the four are comparable; the correlation "
            "summary keeps its per-layer breakdown, because that is where the dense/conv "
            "contrast lives. Pass `--stats-layer net.1` to pin a layer instead.",
            "In `value_ranges`, focus on two things: the SPREAD between metrics at epoch 0 "
            "(ochiai ~0.67 vs. D* ~1.9e7 — why they cannot be compared directly) and the DROP "
            "from epoch 0 to 10 (why a weaker correlation later is not automatically a weaker "
            "relationship).",
            "`min` is omitted on purpose: it is exactly 0 in ~70% of rows because inactive "
            "neurons score zero, so it would be a column of zeros — the `zeros` column states "
            "that properly. `mean` and `median` are both kept: they agree for ochiai, "
            "tarantula and the gradient, and differ ~6x for D*, which is what exposes its "
            "outlier-driven skew.",
            "p05 / p95 / std stay in the full CSV; they widen the table without changing any "
            "argument.",
            f"The complete table for every model, layer, metric and epoch ships with the "
            f"artefact as `outputs/ensemble/value_stats.csv`.",
        ],
    )
    field(log, "Files", len(entries))

    # --- 05: experiment 2 -----------------------------------------------------
    # Deliberately small. Experiment 2 makes ONE claim -- the values settle at a
    # level above zero -- and that claim needs two figures and one table. Every
    # other artefact the pipeline produces is method detail; it stays in
    # outputs/trajectory/ and is cited from there, so this folder can be dropped
    # into the paper without sorting through it first.
    section(log, "05 — trajectories")
    folder = ensure_dir(out_base / "05_trajectories")
    entries = []
    figures = (
        (f"all-combos__population-trajectories.{suffix}",
         "FIGURE 1 — the result: mean over all neurons per epoch, with the neuron spread and "
         "the fitted curve",
         "plot-trajectory-population"),
        (f"all-combos__convergence-windows.{suffix}",
         "FIGURE 2 — that it settles: the level per 50-epoch window, ±1 standard deviation",
         "plot-trajectory-population"),
        (f"all-combos__candidate-models.{suffix}",
         "FIGURE 3 (optional, appendix) — why an exponential with a free limit: all four "
         "candidates drawn over the data",
         "plot-trajectory-population"),
    )
    for name, descr, target in figures:
        source = traj_figures / name
        if source.exists():
            shutil.copy2(source, folder / name)
            entries.append((name, descr))
            copied_total += 1
        else:
            log.warning("  %s not found -- run `make %s` first.", source, target)

    table, table_rows = _condensed_trajectory(TRAJECTORY_DIR, list(CORE_METRIC_NAMES))
    if table:
        (folder / "results_table.md").write_text(
            "# Experiment 2 — where the values settle\n\n"
            "One row per model x metric. `settled level` is measured, not fitted: the mean over "
            "the last 50-epoch window. The two R-squared columns are the SAME exponential fitted "
            "twice -- once with its limit free, once with the limit pinned to zero -- and the "
            "last column is what pinning it to zero costs.\n\n"
            f"{table}\n", encoding="utf-8")
        with (folder / "results_table.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(table_rows[0]))
            writer.writeheader()
            writer.writerows(table_rows)
        entries += [("results_table.md", "TABLE 1 — the paper table: settled level, fitted "
                                         "limit, tau, and the free-limit vs limit-at-zero "
                                         "comparison"),
                    ("results_table.csv", "the same numbers, machine-readable")]
        copied_total += 2
    else:
        log.warning("  no trajectory tables -- run `make plot-trajectory-population` first.")

    _readme(
        folder,
        "05 — Where the suspiciousness values settle (experiment 2)",
        "Two figures and one table, in the order the argument runs: the curves (Fig. 1), the "
        "evidence that they settle (Fig. 2), and the numbers, including what happens if the "
        "limit is forced to zero (Table 1). Fig. 3 is optional support for the choice of "
        "function and belongs in an appendix if it is used at all.",
        "Per-epoch mean over ALL neurons of each model (202 for the dense MLPs, 6518 for the "
        "LeNets) of each suspiciousness metric, with the mean gradient repeated as a dashed "
        "reference on the right axis; the shaded band spans the 5th to 95th percentile of the "
        "neuron population. The red curve is a least-squares fit of a + b*exp(-(e-1)/tau), "
        "estimated from epoch 1 onwards because the untrained model's values are orders of "
        "magnitude larger.",
        entries,
        caption_label="Figure 1, population trajectories",
        extra_captions=[(
            "Figure 2, convergence windows",
            "The population mean per model and metric, averaged within four consecutive "
            "50-epoch windows; error bars are one standard deviation within each window and the "
            "dashed line marks the final window's level. Each window is annotated with its "
            "distance from the preceding window in units of that window's standard deviation. No "
            "model is fitted: the diminishing steps are the direct evidence that the values "
            "settle.",
        ), (
            "Figure 3, candidate models (appendix)",
            "Four candidate models fitted to each population-mean curve: a constant, a linear "
            "trend, an exponential constrained to decay to zero, and an exponential with a free "
            "limit. Each panel is annotated with the models' difference in AIC from the best "
            "model for that curve. Axis limits follow the data, so a rejected candidate may "
            "leave the frame.",
        ), (
            "Table 1, where the values settle",
            "Per model and metric: the level reached over the last 50 epochs (measured, not "
            "fitted), the limit estimated by the exponential fit with its bootstrap confidence "
            "interval, the time constant tau, and a comparison of the same exponential fitted "
            "with a free limit against one constrained to decay to zero.",
        )],
        notes=[
            "READ THE TABLE IN THIS ORDER. (1) `settled level` -- where the run actually ended "
            "up, measured over the last 50 epochs, no model involved. (2) `fitted a` and its "
            "confidence interval -- the same quantity estimated by the fit; it agrees with (1) "
            "to within a few percent. (3) `R2 (a free)` vs `R2 (a = 0)` and `dAIC if a = 0` -- "
            "the evidence that the level is ABOVE zero.",
            "THE CLAIM 'settles above zero' RESTS ON THE LAST THREE COLUMNS. Pinning the limit "
            "to zero costs at least 24 AIC units in every case (>10 is conventionally read as: "
            "no support), and for tarantula it drives R-squared negative -- the constrained fit "
            "is then worse than a flat line through the mean.",
            "The settled level is a measurement over epochs 150-200; the fitted limit is an "
            "estimate of where the curve is heading. Quote the first when stating where the "
            "values ended up, and the second only when discussing the trend.",
            "tau is the time constant in epochs: after tau epochs about 63% of the distance to "
            "the limit is closed, after 3*tau about 95%. It is the number that makes the metrics "
            "comparable -- D* stabilises within a few epochs, tarantula on LeNet/MNIST is still "
            "moving at epoch 200 (tau = 117).",
            "NOT in this folder, on purpose: BIC, held-out prediction error, residual "
            "diagnostics, the per-epoch values and the single-neuron figure. They are method "
            "detail and full data; they live in outputs/trajectory/ (population_fits.csv, "
            "population_model_comparison.csv, population_windows.csv, "
            "population_trajectories.md) and should be cited from there if a reviewer asks.",
            "Two limits to state in the text: the fits' residuals remain autocorrelated in 8 of "
            "12 cases, and no candidate predicts a held-out second half of the run within that "
            "half's own spread. So describe the values as settling to a level and then drifting "
            "slowly, rather than as converged.",
        ],
    )
    field(log, "Files", len(entries))

    # --- index ---------------------------------------------------------------
    index = [
        "# Curated figure set",
        "",
        "Five folders, each with a README stating what the figures show and a "
        "ready-to-paste LaTeX caption. Regenerate with `make paper-figures`.",
        "",
        "| Folder | Goes in | Shows |",
        "| --- | --- | --- |",
        "| `01_method_what-is-correlated` | Method | the two inputs and their correlation, one layer |",
        "| `02_dense_first-layer` | Results | the main result: correlation over training, both MLPs |",
        "| `03_conv_layers` | Results | the same for LeNet's convolution stages (dense vs. conv) |",
        "| `04_value_ranges` | Results | the condensed value-range table |",
        "| `05_trajectories` | Results | experiment 2: where the values settle — two figures "
        "and one table (plus one optional appendix figure) |",
        "",
        f"Source of the figures: `{figures_dir}` (full export, ~640 files) — see its "
        "`manifest.csv` to pick anything not curated here.",
        "",
    ]
    (out_base / "README.md").write_text("\n".join(index), encoding="utf-8")

    section(log, "Done")
    field(log, "Files curated", copied_total)
    long_field(log, "Curated set", out_base)
    long_field(log, "Run log", logfile)


if __name__ == "__main__":
    main()
