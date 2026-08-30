#!/usr/bin/env python3
"""EXPERIMENT 2 -- step T3: the WHOLE-MODEL trajectory, plus a fitted trend.

``plot_trajectory.py`` follows ONE randomly picked neuron, which shows what a
single unit does but says nothing about the model: a dead unit is flat at zero,
a lucky one is spiky, and neither generalises. This script asks the same
question of the entire population instead --

    mean suspiciousness over ALL neurons  vs  mean |gradient| over ALL neurons,
    at every captured epoch

-- in the same overview shape as the per-neuron figure: **one row per model, one
column per metric**, the gradient repeated as the dashed reference in every
panel. Around the mean, the 5th..95th percentile band shows how much of the
population actually follows it.

Three additions beyond a bigger sample:

* **A fitted trend.** Each panel's suspiciousness curve is fitted with
  ``y = a + b·exp(-(e - e0)/tau)`` (see :mod:`susgrad.trends`) and the fit is
  drawn on top. ``a`` is the value the curve is heading for, ``tau`` how many
  epochs it takes to get most of the way there -- i.e. the numeric answer to
  "does this converge?", instead of an eyeball verdict. ``--fit linear`` fits a
  trend line instead (a slope of ~0 being the other way to say "settled"), and
  ``--fit none`` skips fitting.
* **Epoch 0 is excluded from the fit by default** (``--fit-from``). The
  untrained model's values are an outlier of several orders of magnitude; left
  in, they dominate the least squares and the fit describes that one point
  rather than the run.
* **The loss, on a third axis** (``--loss``, default the test loss). The
  gradient plotted here is ``mean |d loss / d activation|`` measured on the TEST
  set, so it is a derivative of the green curve -- and on these runs the test
  loss climbs from ~epoch 20 onwards (the models overfit), which is why the
  gradient climbs with it instead of decaying to zero. Without the loss in the
  panel, a rising mean |gradient| reads as a bug; with it, the two lines make
  the reason visible. Drawn in its own colour, with its own dashes, on a spine
  pushed out past the gradient's, so no curve borrows another's scale. Pass
  ``--loss both`` to add the training loss (which does go to ~0), or
  ``--loss none`` for the old two-axis panel.

Reads ``outputs/trajectory/value_stats.csv`` -- the table ``value_stats.py``
already writes -- and uses its ``layer = all`` rows (every layer's neurons
pooled). Nothing is re-reduced from the tensors here, so the figures cannot
disagree with the value table. Pooling is over neurons, so a big layer weighs
proportionally more; pass ``--layer conv1`` to plot a single layer instead. The
loss comes from ``outputs/trajectory/<combo>__training.csv``, the table
``train_trajectory.py`` wrote during the same run, for the same reason.

Outputs:

    outputs/trajectory/figures/all-combos__population-trajectories.{png,pdf}
    outputs/trajectory/figures/<combo>__population__trajectory.{png,pdf}
    outputs/trajectory/population_trajectories.md    per-model tables + fits
    outputs/trajectory/population_trajectories.csv   every plotted value
    outputs/trajectory/population_fits.csv           one row per fitted case

Run AFTER train_trajectory.py and value_stats.py:

    python scripts/value_stats.py --only trajectory
    python scripts/plot_trajectory_population.py
    make plot-trajectory-population
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click
import torch

from scripts._cli import (
    collect_enabled_ensemble,
    ensemble_combo_options,
    field,
    long_field,
    section,
    select_ensemble_combinations,
    setup_logging,
)
from scripts.value_stats import STATS_NAME
from susgrad.correlation import pearson_correlation
from susgrad.sbfl import METRIC_NAMES
from susgrad.stats import ALL_LAYERS
from susgrad.trends import (
    DEFAULT_TAIL,
    DEFAULT_TOLERANCE,
    bootstrap_parameter_ci,
    compare_models,
    describe_convergence,
    fit_curve_values,
    fit_series,
    residual_diagnostics,
    window_levels,
)
from susgrad.utils import TRAJECTORY_DIR, TRAJECTORY_FIGURES_DIR, ensure_dir
from susgrad.viz.figures import (
    DEFAULT_FORMATS,
    candidate_legend,
    draw_candidate_panel,
    draw_series_panel,
    draw_window_panel,
    new_grid_figure,
    save_figure,
    save_metric_panels,
)
from susgrad.viz.textmap import markdown_table

GRADIENT_KIND = "gradient"

#: The loss curves ``train_trajectory.py`` wrote alongside the tensors, keyed by
#: the ``--loss`` choice that selects them. The TEST loss is the default because
#: it is the loss the captured gradient is a derivative of: both are measured on
#: the held-out split (see ``compute_gradient_spectrum``), so the gradient can
#: only be read against that one.
LOSS_COLUMNS = {"test": ("test_loss", "test loss"), "train": ("train_loss", "training loss")}
LOSS_CHOICES = ("test", "train", "both", "none")

#: The band drawn around the population curve: 90% of the neurons lie inside it.
BAND_FIELDS = ("p05", "p95")
BAND_LABEL = "5th–95th percentile of neurons"

TRAJECTORY_CSV_COLUMNS = ("combo", "layer", "kind", "epoch", "n", "mean", "median",
                          "p05", "p95", "std", "frac_zero", "fitted")

FIT_CSV_COLUMNS = ("combo", "layer", "kind", "stat", "model", "fit_from", "a", "b", "tau",
                   "tau_at_bound", "r2", "rmse", "asymptote", "asymptote_ci_lo",
                   "asymptote_ci_hi", "half_life", "settle_epoch",
                   "tail_epochs", "tail_mean", "tail_std", "tail_drift", "first_value",
                   "last_value", "converged")

# One row per (case, candidate model): the evidence FOR the model that is drawn,
# so the figure is a reported selection rather than an assertion.
COMPARISON_CSV_COLUMNS = ("combo", "layer", "kind", "stat", "fit_from", "model", "n", "k",
                          "r2", "rmse", "aic", "bic", "delta_aic", "holdout_rmse",
                          "holdout_ratio", "asymptote_full", "asymptote_first_half",
                          "asymptote_shift_pct", "lag1_autocorr", "runs_z",
                          "residuals_structured", "drift_pct_per_100ep", "preferred")

# The model-free convergence evidence: what the level actually was, window by
# window, and how far it moved measured in the previous window's own noise.
WINDOW_CSV_COLUMNS = ("combo", "layer", "kind", "stat", "window", "start_epoch", "end_epoch",
                      "n", "mean", "median", "std", "step_in_sd", "step_relative")


# --- reading the stats table ------------------------------------------------------

def load_stats_rows(path: Path, experiment: str = "trajectory"):
    """Every per-epoch row of *path* for one experiment, epoch parsed to int.

    The ``epoch = all`` rows (the pooled-over-training summary) are dropped:
    this script plots the epoch axis, and a row that is not on it would silently
    become a 'point' at the end of the curve.
    """
    with Path(path).open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("experiment") == experiment and r.get("epoch") != "all"]
    for row in rows:
        row["epoch"] = int(row["epoch"])
    return rows


def series_for(rows, *, combo: str, layer: str, kind: str, stat: str):
    """``(epochs, values, (lo, hi))`` for one (combo, layer, kind), epoch-ordered.

    Returns ``(None, None, None)`` when the combination is absent, so a caller
    can skip a model that was not captured instead of crashing on the first
    lookup.
    """
    picked = sorted(
        (r for r in rows if r["combo"] == combo and r["layer"] == layer and r["kind"] == kind),
        key=lambda r: r["epoch"],
    )
    if not picked:
        return None, None, None
    epochs = [r["epoch"] for r in picked]
    values = [float(r[stat]) for r in picked]
    band = ([float(r[BAND_FIELDS[0]]) for r in picked],
            [float(r[BAND_FIELDS[1]]) for r in picked])
    return epochs, values, band


def training_csv_path(base_dir: Path, combo_key: str) -> Path:
    """Where ``train_trajectory.py`` put this model's per-epoch loss."""
    return Path(base_dir) / f"{combo_key}__training.csv"


def loss_series(path: Path, epochs, kinds):
    """``{label: [value per epoch]}`` from a training CSV, aligned to *epochs*.

    An epoch the CSV does not cover -- or covers with a blank cell, as it does
    for the training loss at epoch 0, where no epoch has been trained yet --
    becomes ``nan``, which matplotlib draws as a gap. Padding it with a zero
    would invent a converged-looking point at exactly the epoch the reader is
    most likely to over-interpret.

    Returns ``{}`` when the file is missing, so a model trained before the loss
    was recorded still plots (without the third axis) instead of crashing.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        by_epoch = {int(r["epoch"]): r for r in csv.DictReader(fh) if r.get("epoch")}

    series = {}
    for kind in kinds:
        column, label = LOSS_COLUMNS[kind]
        values = []
        for epoch in epochs:
            cell = (by_epoch.get(epoch) or {}).get(column, "")
            values.append(float(cell) if cell not in ("", None) else float("nan"))
        if any(v == v for v in values):  # all-nan means the column was never filled
            series[label] = values
    return series


def series_pearson(epochs, a, b, *, from_epoch: int = 0) -> float:
    """Pearson r between two epoch series, over the epochs where both are finite.

    The same estimator the correlation experiment uses on per-neuron series
    (:func:`susgrad.correlation.pearson_correlation`), applied here to two whole-
    model curves -- so "r" means one thing across the write-up.

    *from_epoch* drops the head of the run for the same reason the fits do: at
    epoch 0 the model is untrained, so the loss sits at chance (~ln 10) while
    the gradient has not yet grown into the value the rest of the run lives at.
    That single point is far enough from both clouds to set the coefficient on
    its own -- it takes LeNet-5 x MNIST from +0.96 to +0.44 -- and it describes
    initialisation, not training.
    """
    pairs = [(x, y) for e, x, y in zip(epochs, a, b)
             if e >= from_epoch and x == x and y == y]
    if len(pairs) < 3:
        return float("nan")
    return float(pearson_correlation(
        torch.tensor([x for x, _ in pairs], dtype=torch.float64),
        torch.tensor([y for _, y in pairs], dtype=torch.float64),
    ))


def metrics_in(rows, combo: str, layer: str):
    """Which suspiciousness metrics this combo stored, in the canonical order."""
    present = {r["kind"] for r in rows if r["combo"] == combo and r["layer"] == layer}
    return [m for m in METRIC_NAMES if m in present]


# --- fitting ----------------------------------------------------------------------

def fit_case(epochs, values, *, model: str, fit_from: int, tol: float, tail: float):
    """Fit one curve from *fit_from* onwards; returns ``(fit, curve, summary)``.

    *curve* is the fitted values aligned with the FULL epoch list, with NaN
    before *fit_from* -- matplotlib leaves a gap there, so a plotted fit never
    draws over epochs it was not fitted on. Returns ``(None, None, None)`` when
    there are too few points to fit.
    """
    kept = [(e, v) for e, v in zip(epochs, values) if e >= fit_from]
    if model == "none" or len(kept) < 3:
        return None, None, None
    x = [e for e, _ in kept]
    y = [v for _, v in kept]

    fit = fit_series(x, y, model=model)
    by_epoch = dict(zip(x, fit_curve_values(x, fit)))
    curve = [by_epoch.get(e, float("nan")) for e in epochs]
    return fit, curve, describe_convergence(x, y, fit, tol=tol, tail=tail)


def compare_case(epochs, values, *, fit_from: int, models, holdout: float, preferred: str):
    """Score every candidate model on one curve; returns comparison CSV rows.

    ``drift_pct_per_100ep`` is carried on the linear row because that is the one
    number in the whole table a reader can interpret without knowing the metric's
    scale: the trend as a percentage of the curve's own level. Ochiai drifting
    -0.5%/100 epochs and D* drifting -2.6%/100 epochs are comparable statements;
    -3.5e-5 and -0.18 are not.
    """
    kept = [(e, v) for e, v in zip(epochs, values) if e >= fit_from]
    if len(kept) < 5:
        return []
    x = [e for e, _ in kept]
    y = [v for _, v in kept]
    level = sum(y) / len(y)

    scores = compare_models(x, y, models=models, holdout=holdout)
    rows = []
    for name, score in scores.items():
        diagnostics = residual_diagnostics(x, y, fit_series(x, y, model=name))
        drift = ""
        if name == "linear":
            slope = fit_series(x, y, model="linear").params["b"]
            if level:
                drift = f"{100.0 * slope * 100.0 / abs(level):.3f}"
        rows.append({
            "model": name, "n": score.n, "k": score.k,
            "r2": f"{score.r2:.4f}", "rmse": f"{score.rmse:.6g}",
            "aic": f"{score.aic:.2f}", "bic": f"{score.bic:.2f}",
            "delta_aic": f"{score.delta_aic:.2f}",
            "holdout_rmse": f"{score.holdout_rmse:.6g}",
            "holdout_ratio": f"{score.holdout_ratio:.3f}",
            "asymptote_full": _finite(score.asymptote, ".6g"),
            "asymptote_first_half": _finite(score.asymptote_train_only, ".6g"),
            "asymptote_shift_pct": _finite(score.asymptote_shift_pct, ".1f"),
            "lag1_autocorr": f"{diagnostics['lag1_autocorr']:.3f}",
            "runs_z": f"{diagnostics['runs_z']:.2f}",
            "residuals_structured": "yes" if diagnostics["structured"] else "no",
            "drift_pct_per_100ep": drift,
            "preferred": "yes" if name == preferred else "no",
        })
    return rows


def candidate_curves(epochs, values, *, fit_from: int, models):
    """``(x, y, [(model, curve), ...])`` -- every candidate fitted and sampled.

    The same fits the comparison table scores, kept as curves so the figure and
    the table cannot tell different stories.
    """
    kept = [(e, v) for e, v in zip(epochs, values) if e >= fit_from]
    if len(kept) < 5:
        return None, None, []
    x = [e for e, _ in kept]
    y = [v for _, v in kept]
    curves = []
    for name in models:
        try:
            curves.append((name, fit_curve_values(x, fit_series(x, y, model=name))))
        except ValueError:
            continue
    return x, y, curves


def window_rows(epochs, values, *, fit_from: int, windows: int):
    """Model-free convergence evidence: the level in successive equal windows."""
    kept = [(e, v) for e, v in zip(epochs, values) if e >= fit_from]
    if len(kept) < windows * 2:
        return []
    x = [e for e, _ in kept]
    y = [v for _, v in kept]
    rows = []
    for index, window in enumerate(window_levels(x, y, windows=windows), start=1):
        rows.append({
            "window": index,
            "start_epoch": f"{window['start']:.0f}", "end_epoch": f"{window['end']:.0f}",
            "n": window["n"], "mean": f"{window['mean']:.6g}",
            "median": f"{window['median']:.6g}", "std": f"{window['std']:.6g}",
            "step_in_sd": "" if "step_in_sd" not in window else f"{window['step_in_sd']:.2f}",
            "step_relative": "" if "step_relative" not in window
                             else f"{window['step_relative']:.4f}",
        })
    return rows


def _fitted_cell(curve, index) -> str:
    """The fitted value at *index* for a CSV cell -- empty where it was not fitted."""
    if curve is None:
        return ""
    value = curve[index]
    return "" if value != value else f"{value:.6g}"   # NaN: outside the fitted range


def _finite(value, spec=".6g") -> str:
    """Format a float, leaving the cell EMPTY when the model has no such quantity.

    A linear fit has no asymptote and a zero-asymptote exponential's is 0 by
    construction, so both would otherwise print 'nan' or a spurious 0 in a column
    a reader is meant to compare across models.
    """
    return "" if value is None or value != value else format(value, spec)


def _fmt(value, spec=".4g"):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return format(value, spec)


def _fit_row(combo_key, layer, kind, stat, fit, summary, fit_from, asymptote_ci=None):
    lo, hi = asymptote_ci if asymptote_ci else (None, None)
    return {
        "combo": combo_key, "layer": layer, "kind": kind, "stat": stat,
        "model": fit.model, "fit_from": fit_from,
        "asymptote_ci_lo": _fmt(lo, ".6g"), "asymptote_ci_hi": _fmt(hi, ".6g"),
        "a": _fmt(fit.params.get("a"), ".6g"), "b": _fmt(fit.params.get("b"), ".6g"),
        "tau": _fmt(fit.params.get("tau"), ".6g"),
        # A tau at an end of the searched range is a bound, not a measurement --
        # it has to travel with the number, not stay in the log.
        "tau_at_bound": _fmt(bool(fit.at_bound)),
        "r2": _fmt(summary["r2"], ".4f"), "rmse": _fmt(summary["rmse"], ".6g"),
        "asymptote": _fmt(summary["asymptote"], ".6g"),
        "half_life": _fmt(summary["half_life"], ".4g"),
        "settle_epoch": "" if summary["settle_epoch"] is None
                        else f"{summary['settle_epoch']:.1f}",
        "tail_epochs": summary["tail_epochs"],
        "tail_mean": _fmt(summary["tail_mean"], ".6g"),
        "tail_std": _fmt(summary["tail_std"], ".6g"),
        "tail_drift": _fmt(summary["tail_drift"], ".4f"),
        "first_value": _fmt(summary["first_value"], ".6g"),
        "last_value": _fmt(summary["last_value"], ".6g"),
        "converged": _fmt(summary["converged"]),
    }


# --- reporting ---------------------------------------------------------------------

def _fit_table(fit_rows):
    """The per-model fit summary, as a markdown table."""
    header = ["metric", "fit", "asymptote a", "95% CI (bootstrap)", "tau (epochs)", "R²",
              "within 5% at epoch", "tail drift"]
    rows = [[
        r["kind"], r["model"], r["a"],
        f"[{r['asymptote_ci_lo']}, {r['asymptote_ci_hi']}]" if r.get("asymptote_ci_lo") else "—",
        r["tau"] + (" (bound)" if r.get("tau_at_bound") == "yes" else ""), r["r2"],
        r["settle_epoch"] or "not within the run", r["tail_drift"],
    ] for r in fit_rows]
    return markdown_table(header, rows)


def _comparison_table(rows):
    """Why THIS model: fit quality charged per parameter, plus what it left over."""
    header = ["metric", "model", "R²", "ΔAIC", "held-out RMSE / spread",
              "asymptote (full run)", "asymptote (1st half)", "shift",
              "runs z", "structured?", "drift %/100 ep"]
    return markdown_table(header, [[
        r["kind"], r["model"] + (" ←" if r["preferred"] == "yes" else ""), r["r2"],
        r["delta_aic"], r["holdout_ratio"],
        r["asymptote_full"] or "—", r["asymptote_first_half"] or "—",
        f"{r['asymptote_shift_pct']}%" if r["asymptote_shift_pct"] else "—",
        r["runs_z"], r["residuals_structured"], r["drift_pct_per_100ep"] or "—",
    ] for r in rows])


def _window_table(rows):
    """The level window by window -- convergence with no model involved."""
    header = ["metric", "epochs", "mean", "std", "step (in previous std)", "step (% of level)"]
    return markdown_table(header, [[
        r["kind"], f"{r['start_epoch']}–{r['end_epoch']}", r["mean"], r["std"],
        r["step_in_sd"] or "—",
        f"{100 * float(r['step_relative']):.2f}%" if r["step_relative"] else "—",
    ] for r in rows])


def _markdown_section(combo, layer, epochs, stat, susp, gradient, fit_rows, figure, every,
                      comparison_rows=(), window_rows_=(), losses=None, loss_r=None):
    """One model's section: what was pooled, the evidence, the fit, the values."""
    head = [
        f"## {combo.label} (`{combo.key}`)",
        "",
        f"- **Population:** every neuron of {'the whole model' if layer == ALL_LAYERS else layer}"
        f" ({susp[next(iter(susp))]['n']} values per epoch)",
        f"- **Statistic:** {stat} over that population, band = {BAND_LABEL}",
        f"- **Epochs:** {epochs[0]}..{epochs[-1]} ({len(epochs)} snapshots)",
        f"- **Figure:** `{figure.name}`",
        "",
    ]
    if window_rows_:
        head += [
            "### Has it converged? (no model involved)",
            "",
            "The level in successive equal windows. A step much smaller than the previous "
            "window's own standard deviation means the curve has stopped moving in any sense "
            "a reader can dispute -- this is the evidence a convergence claim should rest on, "
            "with the fit below as a summary rather than as proof.",
            "",
            _window_table(window_rows_),
            "",
        ]
    if comparison_rows:
        head += [
            "### Why this fit and not another",
            "",
            "ΔAIC is measured from the best model on this curve and already charges each model "
            "for its parameters, so a lower value is not bought by flexibility (>10 is usually "
            "read as: the other model has essentially no support). The candidates are chosen so "
            "that winning means something: `const` = no trend at all, `linear` = a straight "
            "trend, `exp0` = the SAME exponential forced to decay to zero, `exp` = decay to a "
            "free level. **`exp` beating `exp0` is the measured statement that these values "
            "settle at a POSITIVE level rather than vanishing.** The held-out column refits on "
            "the first half of the run and predicts the second: below 1 means the extrapolation "
            "beats assuming the level simply stays at its average. `structured?` asks whether "
            "the residuals still contain the shape the model missed.",
            "",
            _comparison_table(comparison_rows),
            "",
        ]
    if fit_rows:
        head += ["### Fitted trend", "", _fit_table(fit_rows), ""]

    losses = losses or {}
    if losses:
        head += [
            "### The gradient against the loss it comes from",
            "",
            "The captured gradient is `mean |d loss / d activation|` on the held-out split, so "
            "it is a derivative of the test loss on the same data -- not of the training loss, "
            "and not of anything that gradient descent drives to zero. Stationarity kills the "
            "PARAMETER gradient (`sum_i (dL/da_i)·x_i = 0`); the per-neuron activation "
            "gradients it is built from cancel in that sum rather than vanish, and taking "
            "`|.|` before averaging removes even that cancellation. So the level it settles at "
            "is set by the loss, and a rising test loss (overfitting) is a rising mean "
            "|gradient|.",
            "",
            markdown_table(
                ["series", f"epoch {epochs[0]}", f"epoch {epochs[-1]}",
                 "r with mean \\|gradient\\|"],
                [[label,
                  f"{values[0]:.4g}" if values[0] == values[0] else "—",
                  f"{values[-1]:.4g}" if values[-1] == values[-1] else "—",
                  f"{(loss_r or {}).get(label, float('nan')):+.3f}"]
                 for label, values in losses.items()],
            ),
            "",
        ]

    header = ["epoch"] + list(susp) + [f"{stat} \\|gradient\\|"] + list(losses)
    rows = []
    for i, epoch in enumerate(epochs):
        if epoch % every and epoch not in (epochs[0], epochs[-1]):
            continue
        rows.append([epoch] + [f"{susp[m]['values'][i]:.4g}" for m in susp]
                    + [f"{gradient[i]:.3e}"]
                    + [f"{losses[label][i]:.4g}" if losses[label][i] == losses[label][i] else "—"
                       for label in losses])
    head += [f"### Values (every {every}th epoch)", "", markdown_table(header, rows), ""]
    return "\n".join(head)


# --- the script ---------------------------------------------------------------------

@click.command()
@click.option("--stats-csv", default=None,
              help="Stats table to read (default: outputs/trajectory/value_stats.csv).")
@click.option("--layer", default=ALL_LAYERS, show_default=True,
              help=f"Which population to plot: '{ALL_LAYERS}' (every neuron of the model) "
                   "or a single layer name.")
@click.option("--stat", type=click.Choice(["mean", "median"]), default="mean", show_default=True,
              help="Population statistic per epoch. median is the robust choice for D*, "
                   "whose mean is dominated by a handful of huge neurons.")
@click.option("--fit", "fit_model", type=click.Choice(["exp", "linear", "none"]),
              default="exp", show_default=True,
              help="Trend fitted to each suspiciousness curve (exp = a + b·exp(-e/tau)).")
@click.option("--fit-from", default=1, show_default=True,
              help="First epoch the fit uses. The default drops epoch 0 (the untrained "
                   "model), whose extreme values would otherwise dominate the fit.")
@click.option("--from-epoch", default=0, show_default=True,
              help="First epoch to PLOT. Epoch 0 is kept by default (it is the honest "
                   "starting point); pass 1 to drop the untrained spike that otherwise "
                   "compresses the rest of the curve into the bottom of the panel.")
@click.option("--fit-gradient/--no-fit-gradient", default=False, show_default=True,
              help="Also fit the gradient curve (drawn dotted). Off by default: it is "
                   "repeated in every panel, so its fit is drawn 3x per row.")
@click.option("--compare/--no-compare", default=True, show_default=True,
              help="Score every candidate model (AIC/BIC, held-out prediction, residual "
                   "diagnostics) so the drawn fit is a reported selection, not an assertion.")
@click.option("--compare-models", "comparison_models", default="const,linear,exp0,exp",
              show_default=True,
              help="Candidates scored against each other. The defaults are chosen so that "
                   "winning MEANS something: 'const' = no trend at all, 'linear' = a straight "
                   "trend, 'exp0' = the same exponential forced to decay to ZERO, 'exp' = decay "
                   "to a free (possibly non-zero) level. exp vs exp0 is the direct test of "
                   "whether the values head for zero or for a positive plateau.")
@click.option("--holdout", default=0.5, show_default=True,
              help="Fraction of the run held back when testing whether a model can predict "
                   "epochs it never saw.")
@click.option("--windows", default=4, show_default=True,
              help="Equal windows for the model-free convergence table (0 to skip).")
@click.option("--evidence-figures/--no-evidence-figures", default=True, show_default=True,
              help="Also draw the two figures that SHOW the evidence: every candidate model "
                   "over the data (with its ΔAIC), and the window levels with their spread.")
@click.option("--bootstrap", default=300, show_default=True,
              help="Resamples for the asymptote's confidence interval (0 to skip).")
@click.option("--tolerance", default=DEFAULT_TOLERANCE, show_default=True,
              help="How close to the asymptote counts as settled (fraction of the swing).")
@click.option("--tail", default=DEFAULT_TAIL, show_default=True,
              help="Fraction of the run treated as the tail when measuring drift.")
@click.option("--band/--no-band", default=True, show_default=True,
              help="Shade the 5th..95th percentile of the population around the suspiciousness curve.")
@click.option("--loss", "loss_choice", type=click.Choice(LOSS_CHOICES), default="test",
              show_default=True,
              help="Draw the loss on a third axis, offset past the gradient's. 'test' is the "
                   "loss the captured gradient is a derivative of (both are measured on the "
                   "held-out split), so it is the one that explains the gradient's shape -- "
                   "on these runs it RISES after ~epoch 20 as the models overfit, and the "
                   "gradient rises with it. 'both' adds the training loss, which does head "
                   "for zero. Read from <combo>__training.csv.")
@click.option("--gradient-band/--no-gradient-band", default=False, show_default=True,
              help="Also shade the gradient's percentile band. Off by default: near-dead "
                   "neurons put its 5th percentile at ~0, which stretches the log axis over "
                   "twenty decades and flattens every curve in the panel.")
@click.option("--table-every", default=10, show_default=True,
              help="Write every Nth epoch to the markdown value table (the CSV keeps all).")
@click.option("--formats", default=",".join(DEFAULT_FORMATS), show_default=True,
              help="Image formats to write.")
@click.option("--dpi", default=300, show_default=True)
@click.option("--out-dir", default=None, help="Base dir for the logs (and the figures subdir).")
@ensemble_combo_options
def main(stats_csv, layer, stat, fit_model, fit_from, from_epoch, fit_gradient, compare,
         comparison_models, holdout, windows, evidence_figures, bootstrap, tolerance, tail,
         loss_choice, band, gradient_band, table_every, formats, dpi, out_dir,
         **flags):
    log, logfile = setup_logging("plot_trajectory_population")
    combos = select_ensemble_combinations(collect_enabled_ensemble(flags))
    if not combos:
        log.warning("No combinations selected. Nothing to do.")
        return

    formats = [f.strip().lower() for f in formats.split(",") if f.strip()]
    out_base = Path(out_dir) if out_dir else TRAJECTORY_DIR
    fig_base = ensure_dir(out_base / TRAJECTORY_FIGURES_DIR.name)
    stats_path = Path(stats_csv) if stats_csv else out_base / STATS_NAME
    if not stats_path.exists():
        raise click.ClickException(
            f"{stats_path} not found -- this script plots that table.\n"
            f"  Capture first:  make train-trajectory\n"
            f"  Then summarise: make value-stats\n"
            f"(the table is written by value_stats.py, but it has nothing to summarise "
            f"until train_trajectory.py has captured the gradients)"
        )

    rows = [r for r in load_stats_rows(stats_path) if r["epoch"] >= from_epoch]
    if not rows:
        raise click.ClickException(
            f"{stats_path} has no rows for the trajectory experiment -- run "
            "`python scripts/value_stats.py --only trajectory` after train_trajectory.py."
        )

    section(log, "Plan")
    long_field(log, "Stats table", stats_path)
    field(log, "Population", f"layer '{layer}' ({'every neuron of the model' if layer == ALL_LAYERS else 'one layer'})")
    field(log, "Statistic", f"{stat} per epoch"
          + (f", band {BAND_FIELDS[0]}..{BAND_FIELDS[1]}" if band else ""))
    field(log, "Plotted from", f"epoch {from_epoch}")
    field(log, "Fit", "none" if fit_model == "none"
          else f"{fit_model}, from epoch {fit_from} onwards"
               + (" (gradient too)" if fit_gradient else ""))
    candidate_models = [m.strip() for m in comparison_models.split(",") if m.strip()]
    field(log, "Model selection", "off" if not compare or fit_model == "none"
          else f"{', '.join(candidate_models)} scored by AIC/BIC + a {holdout:.0%} held-out "
               f"tail + residual diagnostics")
    loss_kinds = {"both": ["test", "train"], "none": []}.get(loss_choice, [loss_choice])
    field(log, "Loss overlay", "off" if not loss_kinds
          else f"{', '.join(LOSS_COLUMNS[k][1] for k in loss_kinds)} on a third axis, "
               f"from <combo>__training.csv")
    field(log, "Convergence evidence", "none" if not windows
          else f"{windows} equal windows (model-free)"
               + (f" + {bootstrap} bootstrap resamples for the asymptote" if bootstrap else ""))
    field(log, "Combinations", f"{len(combos)} selected")

    sections, csv_rows, fit_rows, panels = [], [], [], []
    comparison_rows, window_csv_rows = [], []
    # Per (combo, metric): what the evidence FIGURES need, kept numeric.
    evidence = {}
    for combo in combos:
        section(log, combo.label)
        metrics = metrics_in(rows, combo.key, layer)
        epochs, gradient, grad_band = series_for(rows, combo=combo.key, layer=layer,
                                                 kind=GRADIENT_KIND, stat=stat)
        if not metrics or not epochs:
            log.warning("  Nothing captured for layer %r -- run train_trajectory.py and "
                        "value_stats.py first. Skipping.", layer)
            continue

        losses = {}
        if loss_kinds:
            loss_path = training_csv_path(out_base, combo.key)
            losses = loss_series(loss_path, epochs, loss_kinds)
            if not losses:
                log.warning("  %s not found (or has no loss column) -- drawing the panels "
                            "without the loss axis.", loss_path)
            for label, values in losses.items():
                # A training CSV from a SHORTER run pads with nan and draws a
                # curve that just stops -- which reads as "the loss ended", not
                # as "this table does not cover these epochs".
                covered = sum(1 for v in values if v == v)
                if covered < len(epochs):
                    log.warning("  %s covers %d of the %d plotted epochs (%s). The overlay "
                                "will stop early -- rerun train_trajectory.py for this "
                                "gradient kind to refresh it.",
                                loss_path.name, covered, len(epochs), label)

        susp, combo_fits, combo_comparison, combo_windows = {}, [], [], []
        for metric in metrics:
            m_epochs, values, m_band = series_for(rows, combo=combo.key, layer=layer,
                                                  kind=metric, stat=stat)
            if m_epochs != epochs:
                log.warning("  %s: epochs differ from the gradient's -- skipping the metric.",
                            metric)
                continue
            fit, curve, summary = fit_case(epochs, values, model=fit_model, fit_from=fit_from,
                                           tol=tolerance, tail=tail)
            n_values = next(r["n"] for r in rows if r["combo"] == combo.key
                            and r["layer"] == layer and r["kind"] == metric)
            susp[metric] = {"values": values, "band": m_band, "fit": curve, "n": n_values}
            if fit is not None:
                asymptote_ci = None
                if bootstrap and fit.model == "exp":
                    kept = [(e, v) for e, v in zip(epochs, values) if e >= fit_from]
                    asymptote_ci = bootstrap_parameter_ci(
                        [e for e, _ in kept], [v for _, v in kept], fit, "a",
                        resamples=bootstrap, seed=0,
                    )
                row = _fit_row(combo.key, layer, metric, stat, fit, summary, fit_from,
                               asymptote_ci)
                combo_fits.append(row)
                settled = ("not settled within the run" if summary["settle_epoch"] is None
                           else f"within {tolerance:.0%} of the asymptote by epoch "
                                f"{summary['settle_epoch']:.0f}")
                field(log, metric, f"{fit.describe()} · asymptote {summary['asymptote']:.4g}"
                                   + (f" (95% CI [{asymptote_ci[0]:.4g}, {asymptote_ci[1]:.4g}])"
                                      if asymptote_ci else "")
                                   + f" · {settled} · tail drift {summary['tail_drift']:.3f}"
                                   + (" · WARNING: tau hit the end of the searched range, so it "
                                      "is a bound, not a measurement" if fit.at_bound else ""))
            else:
                field(log, metric, f"{values[0]:.4g} → {values[-1]:.4g} (no fit)")

            common = {"combo": combo.key, "layer": layer, "kind": metric, "stat": stat}
            if compare and fit_model != "none":
                scored = compare_case(epochs, values, fit_from=fit_from,
                                      models=candidate_models, holdout=holdout,
                                      preferred=fit_model)
                combo_comparison += [{**common, "fit_from": fit_from, **r} for r in scored]
                chosen = next((r for r in scored if r["preferred"] == "yes"), None)
                rivals = [r for r in scored if r["preferred"] == "no"]
                if chosen and rivals:
                    field(log, f"{metric} — model choice",
                          f"{fit_model} preferred over "
                          + ", ".join(f"{r['model']} (ΔAIC +{r['delta_aic']})" for r in rivals)
                          + f" · held-out RMSE / spread {chosen['holdout_ratio']}"
                          + " · residuals "
                          + ("STRUCTURED (runs z " f"{chosen['runs_z']}) — the shape is not fully "
                             "captured, so report the fit as descriptive"
                             if chosen["residuals_structured"] == "yes"
                             else f"noise-like (runs z {chosen['runs_z']})"))
            if windows:
                combo_windows += [{**common, **r} for r in
                                  window_rows(epochs, values, fit_from=fit_from, windows=windows)]

            if evidence_figures:
                kept = [(e, v) for e, v in zip(epochs, values) if e >= fit_from]
                fit_x, fit_y, curves = candidate_curves(epochs, values, fit_from=fit_from,
                                                        models=candidate_models)
                entry = {"x": fit_x, "y": fit_y, "curves": curves,
                         "deltas": {r["model"]: r["delta_aic"] for r in
                                    (scored if compare and fit_model != "none" else [])},
                         "windows": window_levels([e for e, _ in kept], [v for _, v in kept],
                                                  windows=windows) if windows else []}
                evidence.setdefault(combo.key, {})[metric] = entry

        if not susp:
            log.warning("  No metric series available. Skipping.")
            continue

        grad_fit_curve = None
        if fit_gradient:
            g_fit, grad_fit_curve, g_summary = fit_case(epochs, gradient, model=fit_model,
                                                        fit_from=fit_from, tol=tolerance, tail=tail)
            if g_fit is not None:
                combo_fits.append(_fit_row(combo.key, layer, GRADIENT_KIND, stat, g_fit,
                                           g_summary, fit_from))
                field(log, "gradient", g_fit.describe())

        field(log, "Population size", f"{susp[metrics[0]]['n']} neurons per epoch")
        field(log, "Epochs", f"{epochs[0]}..{epochs[-1]} ({len(epochs)} snapshots)")
        field(log, "Gradient", f"epoch {epochs[0]}: {gradient[0]:.3e} → "
                               f"epoch {epochs[-1]}: {gradient[-1]:.3e}")
        # The gradient is d(loss)/d(activation) on the SAME split the loss is
        # measured on, so this r is the direct check on "is the gradient's shape
        # the loss's shape?" -- the question a rising mean |gradient| provokes.
        loss_r = {label: series_pearson(epochs, values, gradient, from_epoch=fit_from)
                  for label, values in losses.items()}
        for label, values in losses.items():
            finite = [v for v in values if v == v]
            field(log, label.capitalize(),
                  f"epoch {epochs[0]}: {values[0]:.4g} → epoch {epochs[-1]}: {values[-1]:.4g} "
                  f"(min {min(finite):.4g}) · r with the gradient {loss_r[label]:+.3f} "
                  f"(from epoch {fit_from})")

        stem = fig_base / f"{combo.key}__population__trajectory"
        written = save_metric_panels(
            stem, epochs, {m: susp[m]["values"] for m in susp},
            right_series={f"{stat} |gradient|": gradient},
            context_series=losses or None,
            context_label="cross-entropy loss",
            context_legend_labels=[f"{label} (far-right axis)" for label in losses],
            bands_by_name={m: susp[m]["band"] for m in susp} if band else None,
            fits_by_name={m: susp[m]["fit"] for m in susp if susp[m]["fit"]},
            right_band=grad_band if gradient_band else None,
            right_fit=grad_fit_curve,
            suptitle=f"{combo.label} — {stat} over ALL "
                     f"{'neurons' if layer == ALL_LAYERS else f'{layer} neurons'} "
                     f"({susp[metrics[0]]['n']} per epoch), {len(epochs)} snapshots",
            subtitle=f"band: {BAND_LABEL}" + ("" if fit_model == "none"
                                              else f" · fit: {fit_model}, from epoch {fit_from}"),
            left_legend_label=f"{stat} suspiciousness (left axis)",
            right_legend_label=f"{stat} |gradient| (right axis)",
            right_label=f"{stat} |gradient|",
            band_label=BAND_LABEL if band else "",
            fit_label="" if fit_model == "none" else f"fitted trend ({fit_model})",
            formats=formats, dpi=dpi,
        )
        long_field(log, "Figure", written[0])

        panels.append((combo, epochs, susp, gradient, grad_band, grad_fit_curve, losses,
                       list(susp)))
        fit_rows += combo_fits
        comparison_rows += combo_comparison
        window_csv_rows += combo_windows
        sections.append(_markdown_section(combo, layer, epochs, stat, susp, gradient,
                                          combo_fits, written[0], table_every,
                                          combo_comparison, combo_windows, losses, loss_r))

        for i, epoch in enumerate(epochs):
            for metric in susp:
                source = next(r for r in rows if r["combo"] == combo.key and r["layer"] == layer
                              and r["kind"] == metric and r["epoch"] == epoch)
                csv_rows.append({
                    "combo": combo.key, "layer": layer, "kind": metric, "epoch": epoch,
                    "n": source["n"], "mean": source["mean"], "median": source["median"],
                    "p05": source["p05"], "p95": source["p95"], "std": source["std"],
                    "frac_zero": source["frac_zero"],
                    "fitted": _fitted_cell(susp[metric]["fit"], i),
                })
            source = next(r for r in rows if r["combo"] == combo.key and r["layer"] == layer
                          and r["kind"] == GRADIENT_KIND and r["epoch"] == epoch)
            csv_rows.append({
                "combo": combo.key, "layer": layer, "kind": GRADIENT_KIND, "epoch": epoch,
                "n": source["n"], "mean": source["mean"], "median": source["median"],
                "p05": source["p05"], "p95": source["p95"], "std": source["std"],
                "frac_zero": source["frac_zero"],
                "fitted": _fitted_cell(grad_fit_curve, i),
            })

    if not panels:
        log.warning("Nothing plotted (run train_trajectory.py and value_stats.py first).")
        return

    # The overview: a row per model, a column per metric -- the same shape as the
    # per-neuron figure, so the two can be put side by side in the write-up.
    all_metrics = sorted({m for *_, metrics_here in panels for m in metrics_here},
                         key=lambda m: METRIC_NAMES.index(m))
    # Widen the panels when a third axis is drawn, or its offset spine is paid
    # for out of the plotting area.
    any_loss = any(panel[6] for panel in panels)
    fig, axes = new_grid_figure(len(panels), len(all_metrics),
                                figsize=((5.05 if any_loss else 4.2) * len(all_metrics),
                                         3.0 * len(panels)))
    for row, (combo, epochs, susp, gradient, grad_band, grad_curve, losses,
              _) in zip(axes, panels):
        for ax, metric in zip(row, all_metrics):
            if metric not in susp:
                ax.axis("off")
                continue
            draw_series_panel(
                ax, epochs, {metric: susp[metric]["values"]},
                right_series={f"{stat} |gradient|": gradient},
                left_band=susp[metric]["band"] if band else None,
                left_fit=susp[metric]["fit"],
                right_band=grad_band if gradient_band else None,
                right_fit=grad_curve,
                context_series=losses or None,
                title=f"{combo.label} · {metric}\n{stat} over all "
                      f"{'neurons' if layer == ALL_LAYERS else layer} "
                      f"({susp[metric]['n']})",
                left_label=f"{stat} {metric}", right_label=f"{stat} |gradient|",
                context_label="cross-entropy loss",
            )
    overview = save_figure(
        fig, fig_base / "all-combos__population-trajectories", formats, dpi, legend=True,
        legend_kwargs={
            "left_label": f"{stat} suspiciousness (left axis)",
            "right_label": f"{stat} |gradient| (right axis)",
            "band_label": BAND_LABEL if band else "",
            "fit_label": "" if fit_model == "none" else f"fitted trend ({fit_model})",
            "context_labels": [f"{label} (far-right axis)"
                               for label in (panels[0][6] if any_loss else {})],
        },
    )

    # --- the evidence, drawn ---------------------------------------------------
    # A table of AIC numbers convinces a statistician; a picture of the rejected
    # models lying visibly beside the data convinces everyone else, and the two
    # come from the same fits, so they cannot disagree.
    evidence_written = []
    if evidence_figures and evidence:
        ordered = [(panel[0], evidence[panel[0].key]) for panel in panels
                   if panel[0].key in evidence]

        fig, axes = new_grid_figure(len(ordered), len(all_metrics),
                                    figsize=(4.4 * len(all_metrics), 3.1 * len(ordered)))
        for row, (combo, per_metric) in zip(axes, ordered):
            for ax, metric in zip(row, all_metrics):
                entry = per_metric.get(metric)
                if not entry or not entry["curves"]:
                    ax.axis("off")
                    continue
                annotation = "\n".join(
                    f"{name:>6s} ΔAIC {float(entry['deltas'][name]):>8.1f}"
                    for name, _ in entry["curves"] if name in entry["deltas"]
                )
                draw_candidate_panel(
                    ax, entry["x"], entry["y"], entry["curves"],
                    title=f"{combo.label} · {metric}",
                    ylabel=f"{stat} {metric}", annotation=annotation,
                )
        candidate_written = save_figure(
            fig, fig_base / "all-combos__candidate-models", formats, dpi,
            # Four line styles to explain, which the default series legend does
            # not describe.
            legend_artist=lambda figure: candidate_legend(figure, candidate_models),
        )
        evidence_written.append(candidate_written[0])
        long_field(log, "Evidence figure (candidate models)", candidate_written[0])

        if windows:
            fig, axes = new_grid_figure(len(ordered), len(all_metrics),
                                        figsize=(4.0 * len(all_metrics), 2.8 * len(ordered)))
            for row, (combo, per_metric) in zip(axes, ordered):
                for ax, metric in zip(row, all_metrics):
                    entry = per_metric.get(metric)
                    if not entry or not entry["windows"]:
                        ax.axis("off")
                        continue
                    draw_window_panel(
                        ax, entry["windows"], title=f"{combo.label} · {metric}",
                        ylabel=f"{stat} {metric}",
                    )
            window_written = save_figure(
                fig, fig_base / "all-combos__convergence-windows", formats, dpi,
            )
            evidence_written.append(window_written[0])
            long_field(log, "Evidence figure (convergence windows)", window_written[0])

    md_path = out_base / "population_trajectories.md"
    md_path.write_text("\n".join([
        "# Population trajectories (experiment 2)",
        "",
        f"Every neuron of each model pooled into one population, summarised by its "
        f"**{stat}** at each captured epoch — the whole-model counterpart to the "
        "single-neuron curves in `neuron_trajectories.md`.",
        "",
        f"- Population: `layer = {layer}` rows of `value_stats.csv`",
        f"- Band: {BAND_LABEL}",
        f"- Loss: " + ("not drawn" if not loss_kinds
                       else f"{', '.join(LOSS_COLUMNS[k][1] for k in loss_kinds)} from "
                            f"`<combo>__training.csv`, on a third axis. The gradient is a "
                            f"derivative of the TEST loss (both are measured on the held-out "
                            f"split), which is why it tracks that curve rather than heading "
                            f"for zero."),
        f"- Fit: " + ("none" if fit_model == "none"
                      else f"`{fit_model}`, fitted from epoch {fit_from} onwards "
                           f"(epoch 0 is the untrained model and would dominate the "
                           f"least squares)"),
        f"- Model choice: " + ("not tested" if not comparison_rows
                               else f"`{fit_model}` was SELECTED, not assumed — every case is "
                                    f"scored against {', '.join(candidate_models)} by AIC/BIC, by "
                                    f"predicting a {holdout:.0%} held-out tail it was not fitted "
                                    f"on, and by whether its residuals still carry structure. "
                                    f"The per-model tables below carry that evidence."),
        f"- Settled = the fit is within {tolerance:.0%} of its asymptote AND the last "
        f"{tail:.0%} of the run drifts by no more than that. Treat the window table as the "
        f"primary evidence: it involves no model at all.",
        "",
        f"Overview figure: `{overview[0].name}`",
        "",
    ] + sections), encoding="utf-8")

    csv_path = out_base / "population_trajectories.csv"
    _write_csv(csv_path, TRAJECTORY_CSV_COLUMNS, csv_rows)
    fits_path = out_base / "population_fits.csv"
    if fit_rows:
        _write_csv(fits_path, FIT_CSV_COLUMNS, fit_rows)
    comparison_path = out_base / "population_model_comparison.csv"
    if comparison_rows:
        _write_csv(comparison_path, COMPARISON_CSV_COLUMNS, comparison_rows)
    windows_path = out_base / "population_windows.csv"
    if window_csv_rows:
        _write_csv(windows_path, WINDOW_CSV_COLUMNS, window_csv_rows)

    section(log, "Done")
    field(log, "Models plotted", len(panels))
    field(log, "Fits", f"{len(fit_rows)} ({sum(1 for r in fit_rows if r['converged'] == 'yes')} "
                       f"settled within the run)")
    if comparison_rows:
        chosen = [r for r in comparison_rows if r["preferred"] == "yes"]
        field(log, "Model selection",
              f"{fit_model} preferred in {sum(1 for r in chosen if float(r['delta_aic']) == 0)}"
              f"/{len(chosen)} cases by AIC · "
              f"{sum(1 for r in chosen if r['residuals_structured'] == 'yes')} of them still "
              f"leave structured residuals · "
              f"{sum(1 for r in chosen if float(r['holdout_ratio']) < 1)} predict the held-out "
              f"tail better than its own mean")
    long_field(log, "Overview figure", overview[0])
    long_field(log, "Value log (markdown)", md_path)
    long_field(log, "Value log (CSV)", csv_path)
    if fit_rows:
        long_field(log, "Fit parameters (CSV)", fits_path)
    if comparison_rows:
        long_field(log, "Model comparison (CSV)", comparison_path)
    if window_csv_rows:
        long_field(log, "Window levels (CSV)", windows_path)
    long_field(log, "Run log", logfile)


def _write_csv(path: Path, columns, rows) -> Path:
    ensure_dir(path.parent)
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    main()
