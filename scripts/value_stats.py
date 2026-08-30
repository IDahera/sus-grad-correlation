#!/usr/bin/env python3
"""Value ranges of every suspiciousness metric and the gradient, as CSV.

Answers "what scale do these numbers live on?" per **model x layer x metric x
epoch** -- min, 5th percentile, median, mean, 95th percentile, max, std and the
share of exact zeros. Ochiai is bounded to [0, 1], tarantula hovers around 0.5
and D* reaches tens of millions; without this table those numbers cannot be
compared or sanity-checked in a paper.

Reads the dumps BOTH experiments already wrote -- nothing is retrained, so this
is cheap to re-run:

    outputs/ensemble/value_stats.csv     pooled over all instances (experiment 1)
    outputs/trajectory/value_stats.csv   the single seeded run (experiment 2)

Each file has one row per (combo, layer, kind, epoch), plus a row with
``epoch = all`` per (combo, layer, kind) pooling every captured epoch -- so a
LaTeX table can be built from a single filter, without a second pass.

Every ``(kind, epoch)`` additionally gets a ``layer = all`` row: all of that
model's neurons pooled into ONE population. Those rows are the whole-model
"mean suspiciousness / mean gradient at epoch e" series that
``scripts/plot_trajectory_population.py`` plots -- it reads this CSV instead of
re-reducing the tensors, so the figure and the table can never disagree.

Run:

    python scripts/value_stats.py                 # both experiments
    python scripts/value_stats.py --only ensemble
    make value-stats
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from scripts._cli import (
    collect_enabled_ensemble,
    ensemble_combo_options,
    field,
    long_field,
    section,
    select_ensemble_combinations,
    setup_logging,
)
from susgrad.persistence import (
    list_epochs,
    list_instances,
    load_gradients,
    load_suspiciousness,
    seed_run_name,
)
from susgrad.sbfl import METRIC_NAMES
from susgrad.stats import ALL_LAYERS, STAT_FIELDS, describe_stack, format_row
from susgrad.utils import (
    ENSEMBLE_DIR,
    ENSEMBLE_GRADIENTS_DIR,
    ENSEMBLE_SUSPICIOUSNESS_DIR,
    TRAJECTORY_DIR,
    TRAJECTORY_GRADIENTS_DIR,
    TRAJECTORY_SUSPICIOUSNESS_DIR,
    ensure_dir,
)

COLUMNS = ["experiment", "combo", "layer", "kind", "epoch", "instances"] + list(STAT_FIELDS)


def _rows_for_epoch(experiment, combo_key, epoch, susp_dumps, grad_dumps, metrics, layers):
    """One row per (layer, kind) at this epoch, pooled over the given dumps.

    Plus a ``layer = all`` row per kind: every layer's neurons pooled into one
    population. That row is what a whole-model curve is drawn from -- see
    ``scripts/plot_trajectory_population.py`` -- so it is written here rather
    than re-derived there, and the figure can never disagree with the table.
    Pooling is over NEURONS, so a big layer contributes proportionally more
    (conv1's 4704 units outweigh net.1's 128); the per-layer rows are still
    there for anyone who wants them separated.
    """
    rows = []
    for layer in list(layers) + [ALL_LAYERS]:
        wanted = list(layers) if layer == ALL_LAYERS else [layer]
        for metric in metrics:
            stats = describe_stack(
                d[metric][name] for d in susp_dumps for name in wanted if name in d.get(metric, {})
            )
            rows.append({
                "experiment": experiment, "combo": combo_key, "layer": layer,
                "kind": metric, "epoch": epoch, "instances": len(susp_dumps),
                **format_row(stats),
            })
        stats = describe_stack(d[name] for d in grad_dumps for name in wanted if name in d)
        rows.append({
            "experiment": experiment, "combo": combo_key, "layer": layer,
            "kind": "gradient", "epoch": epoch, "instances": len(grad_dumps),
            **format_row(stats),
        })
    return rows


def _pooled_rows(experiment, combo_key, per_epoch_susp, per_epoch_grad, metrics, layers, n_instances):
    """The ``epoch = all`` rows: every captured epoch pooled together.

    Carries the same ``layer = all`` row as :func:`_rows_for_epoch`, so the
    ``(layer, epoch) = (all, all)`` cell -- the single number describing a whole
    model over its whole run -- exists too.
    """
    rows = []
    for layer in list(layers) + [ALL_LAYERS]:
        wanted = list(layers) if layer == ALL_LAYERS else [layer]
        for metric in metrics:
            stats = describe_stack(
                d[metric][name] for dumps in per_epoch_susp.values()
                for d in dumps for name in wanted if name in d.get(metric, {})
            )
            rows.append({
                "experiment": experiment, "combo": combo_key, "layer": layer,
                "kind": metric, "epoch": "all", "instances": n_instances, **format_row(stats),
            })
        stats = describe_stack(
            d[name] for dumps in per_epoch_grad.values()
            for d in dumps for name in wanted if name in d
        )
        rows.append({
            "experiment": experiment, "combo": combo_key, "layer": layer,
            "kind": "gradient", "epoch": "all", "instances": n_instances, **format_row(stats),
        })
    return rows


def _collect(log, experiment, combos, susp_base, grad_base, variants_for, epoch_limit=None):
    """Walk one experiment's dumps and build every stats row."""
    rows = []
    for combo in combos:
        variants = variants_for(combo)
        if not variants:
            log.warning("  %s: nothing captured -- skipping.", combo.key)
            continue

        # Epochs present in EVERY variant (instance/run), so a row always pools
        # the same population.
        epoch_sets = [
            set(list_epochs(susp_base, combo.key, v)) & set(list_epochs(grad_base, combo.key, v))
            for v in variants
        ]
        epochs = sorted(set.intersection(*epoch_sets)) if epoch_sets else []
        if epoch_limit:
            epochs = [e for e in epochs if e in epoch_limit]
        if not epochs:
            log.warning("  %s: no epoch captured in every run -- skipping.", combo.key)
            continue

        per_epoch_susp, per_epoch_grad = {}, {}
        for epoch in epochs:
            per_epoch_susp[epoch] = [
                load_suspiciousness(combo.key, v, epoch, base_dir=susp_base) for v in variants
            ]
            per_epoch_grad[epoch] = [
                load_gradients(combo.key, v, epoch, base_dir=grad_base) for v in variants
            ]

        first_susp = per_epoch_susp[epochs[0]][0]
        metrics = [m for m in METRIC_NAMES if m in first_susp]
        layers = list(per_epoch_grad[epochs[0]][0])

        for epoch in epochs:
            rows += _rows_for_epoch(experiment, combo.key, epoch, per_epoch_susp[epoch],
                                    per_epoch_grad[epoch], metrics, layers)
        rows += _pooled_rows(experiment, combo.key, per_epoch_susp, per_epoch_grad,
                             metrics, layers, len(variants))

        field(log, combo.key, f"{len(variants)} run(s) x {len(epochs)} epoch(s) x "
                              f"{len(layers)} layer(s) (+ '{ALL_LAYERS}') x "
                              f"{len(metrics)} metric(s) + gradient")
    return rows


def _write(path: Path, rows) -> Path:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


#: The one summary table this script writes, per experiment directory.
STATS_NAME = "value_stats.csv"


@click.command()
@click.option("--only", type=click.Choice(["both", "ensemble", "trajectory"]), default="both",
              show_default=True, help="Which experiment's dumps to summarise.")
@click.option("--seed", default=42, show_default=True,
              help="Experiment 2: which seeded run folder to read.")
@click.option("--epochs", default=None,
              help="Restrict to these epochs (comma-separated); default: everything captured.")
@click.option("--ensemble-out", default=None, help="Output CSV for experiment 1.")
@click.option("--trajectory-out", default=None, help="Output CSV for experiment 2.")
@ensemble_combo_options
def main(only, seed, epochs, ensemble_out, trajectory_out, **flags):
    log, logfile = setup_logging("value_stats")
    combos = select_ensemble_combinations(collect_enabled_ensemble(flags))
    if not combos:
        log.warning("No combinations selected. Nothing to do.")
        return

    epoch_limit = {int(e) for e in epochs.split(",") if e.strip()} if epochs else None

    section(log, "Plan")
    field(log, "Experiments", only)
    field(log, "Epochs", "all captured" if epoch_limit is None else sorted(epoch_limit))
    field(log, "Statistics", ", ".join(STAT_FIELDS))
    field(log, "Gradients", f"read from {Path('gradients')}/, written to {STATS_NAME}")

    ens_grads = ENSEMBLE_GRADIENTS_DIR
    traj_grads = TRAJECTORY_GRADIENTS_DIR
    # A missing directory means the CAPTURE step was never run -- a different
    # problem from "captured, but no epoch is common to every run", and the
    # per-combo skip below cannot tell them apart.
    for label, directory, step in (("ensemble", ens_grads, "train-ensemble"),
                                   ("trajectory", traj_grads, "train-trajectory")):
        if only in ("both", label) and not directory.exists():
            log.warning("%s: %s does not exist -- nothing was captured yet. "
                        "Run `make %s` first.", label, directory, step)

    written = []
    if only in ("both", "ensemble"):
        section(log, "Experiment 1 (across instances)")
        rows = _collect(
            log, "ensemble", combos, ENSEMBLE_SUSPICIOUSNESS_DIR, ens_grads,
            lambda c: sorted(set(list_instances(ENSEMBLE_SUSPICIOUSNESS_DIR, c.key))
                             & set(list_instances(ens_grads, c.key))),
            epoch_limit,
        )
        if rows:
            path = (Path(ensemble_out) if ensemble_out
                    else ENSEMBLE_DIR / STATS_NAME)
            written.append(_write(path, rows))
            field(log, "Rows", len(rows))

    if only in ("both", "trajectory"):
        section(log, "Experiment 2 (across epochs, one seeded run)")
        variant = seed_run_name(seed)
        rows = _collect(
            log, "trajectory", combos, TRAJECTORY_SUSPICIOUSNESS_DIR, traj_grads,
            lambda c, v=variant: [v] if list_epochs(TRAJECTORY_SUSPICIOUSNESS_DIR, c.key, v) else [],
            epoch_limit,
        )
        if rows:
            path = (Path(trajectory_out) if trajectory_out
                    else TRAJECTORY_DIR / STATS_NAME)
            written.append(_write(path, rows))
            field(log, "Rows", len(rows))

    section(log, "Done")
    if not written:
        log.warning("Nothing summarised -- run the experiments first.")
    for path in written:
        long_field(log, "CSV", path)
    long_field(log, "Run log", logfile)


if __name__ == "__main__":
    main()
