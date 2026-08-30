#!/usr/bin/env python3
"""MAIN (ensemble) PIPELINE -- step 3: render the interactive HTML report.

Builds ONE self-contained, interactive HTML with a tab per combo. Dropdowns for
layer / channel / suspiciousness metric / correlation method update the heatmaps
live in the browser (see ``susgrad.viz.ensemble_html`` for why: too many
combinations to pre-render as separate images).

Every captured epoch (base case: 0, 1, 10) is shown SIDE BY SIDE in one row --
one row for suspiciousness, one for gradient, one for correlation -- so the
before/after-1/after-10 comparison is a left-to-right read rather than a
dropdown you have to flip between. Within a row the colour scale is shared
across epochs, so the panels really are comparable.

The suspiciousness/gradient heatmaps are shown for ONE instance per combo --
picked at random (or pinned via ``--instance``) and clearly logged/labelled, so
it's traceable. The correlation heatmaps are already aggregated across ALL
instances (that's the whole point of the ensemble pipeline), so they are not
instance-specific.

Run AFTER train_ensemble.py + correlate_ensemble.py:

    python scripts/evaluate_ensemble.py --epochs 0,1,10
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from scripts._cli import (
    collect_enabled_ensemble,
    ensemble_combo_options,
    field,
    long_field,
    parse_epoch_list,
    section,
    select_ensemble_combinations,
    setup_logging,
)
from susgrad.correlation import CORRELATIONS
from susgrad.persistence import (
    correlation_path,
    epoch_snapshot_name,
    list_epochs,
    list_instances,
    load_correlation,
    load_gradients,
    load_suspiciousness,
)
from susgrad.sbfl import METRIC_NAMES
from susgrad.utils import (
    ENSEMBLE_CORRELATION_DIR,
    ENSEMBLE_GRADIENTS_DIR,
    ENSEMBLE_SUSPICIOUSNESS_DIR,
    VISUALIZATIONS_DIR,
    ensure_dir,
)
from susgrad.viz import build_heatmap
from susgrad.viz.ensemble_html import build_ensemble_report
from susgrad.viz.textmap import correlation_summary

CORR_METHODS = tuple(CORRELATIONS)


def _n_channels(tensor) -> int:
    return int(tensor.shape[0]) if tensor.dim() >= 3 else 1


def _grid_to_json(grid):
    """np.ndarray (possibly with NaN padding) -> JSON-safe nested list (NaN -> null)."""
    return [[(None if v != v else float(v)) for v in row] for row in grid.tolist()]


def _channel_grids(name, tensor, n_channels):
    return [_grid_to_json(build_heatmap(name, tensor, channel=c).grid) for c in range(n_channels)]


def _build_combo_payload(combo, instances, chosen_inst, candidate_epochs, susp_base, grad_base, corr_base, log):
    # Only keep epochs that have correlation data too, so every value the
    # dropdowns can select is fully populated (susp AND grad AND corr) -- a
    # partially-run correlate_ensemble.py must never produce a page that throws
    # mid-render for some epoch.
    epochs = [e for e in candidate_epochs if correlation_path(combo.key, epoch_snapshot_name(e), corr_base).exists()]
    if not epochs:
        log.warning("  No correlation data for %s -- run correlate_ensemble.py first. Skipping.", combo.key)
        return None

    # Epoch 0 (or the first available)'s gradient shapes give the layer order +
    # per-layer channel count.
    grad0 = load_gradients(combo.key, chosen_inst, epochs[0], base_dir=grad_base)
    layers = list(grad0)
    layer_channels = {name: _n_channels(grad0[name]) for name in layers}
    # For the axis explanation in the report: conv heatmaps show one channel's
    # true H×W spatial map; dense layers are wrapped row-major into a square.
    layer_kinds = {name: ("conv" if grad0[name].dim() >= 3 else "dense") for name in layers}

    metrics_present = set(METRIC_NAMES)
    per_epoch_susp, per_epoch_grad, per_epoch_corr = {}, {}, {}
    for epoch in epochs:
        susp_e = load_suspiciousness(combo.key, chosen_inst, epoch, base_dir=susp_base)
        grad_e = load_gradients(combo.key, chosen_inst, epoch, base_dir=grad_base)
        corr_e = load_correlation(combo.key, epoch_snapshot_name(epoch), base_dir=corr_base)
        metrics_present &= set(susp_e) & set(corr_e)
        per_epoch_susp[epoch] = susp_e
        per_epoch_grad[epoch] = grad_e
        per_epoch_corr[epoch] = corr_e

    metrics_present = [m for m in METRIC_NAMES if m in metrics_present]
    methods_present = [m for m in CORR_METHODS if all(m in per_epoch_corr[e].get(metrics_present[0], {}) for e in epochs)] if metrics_present else []

    susp, grad = {}, {}
    for name in layers:
        n_c = layer_channels[name]
        grad[name] = {epoch: _channel_grids(name, per_epoch_grad[epoch][name], n_c) for epoch in epochs}
        susp[name] = {
            metric: {epoch: _channel_grids(name, per_epoch_susp[epoch][metric][name], n_c) for epoch in epochs}
            for metric in metrics_present
        }

    # Correlation: aggregated across ALL instances already, by epoch snapshot.
    # ``corr_stats`` carries the whole-layer summary (mean |r| etc.) computed
    # from the FULL tensor -- the grids may be block-averaged for display, so
    # the browser must not try to derive these numbers from them.
    corr = {name: {metric: {method: {} for method in methods_present} for metric in metrics_present} for name in layers}
    corr_stats = {name: {metric: {method: {} for method in methods_present} for metric in metrics_present} for name in layers}
    for epoch in epochs:
        nested = per_epoch_corr[epoch]  # {metric: {method: {layer: tensor}}}
        for name in layers:
            n_c = layer_channels[name]
            for metric in metrics_present:
                for method in methods_present:
                    tensor = nested.get(metric, {}).get(method, {}).get(name)
                    if tensor is None:
                        continue
                    corr[name][metric][method][epoch] = _channel_grids(name, tensor, n_c)
                    corr_stats[name][metric][method][epoch] = correlation_summary(
                        tensor.detach().cpu().numpy()
                    )

    return {
        "key": combo.key,
        "label": combo.label,
        "instance_label": f"{chosen_inst} (random pick, logged)",
        "n_instances": len(instances),
        "epochs": epochs,
        "metrics": metrics_present,
        "corr_methods": methods_present,
        "layers": layers,
        "layer_channels": layer_channels,
        "layer_kinds": layer_kinds,
        "susp": susp,
        "grad": grad,
        "corr": corr,
        "corr_stats": corr_stats,
    }


@click.command()
@click.option("--instance", default=None, help="Pin a specific instance id (e.g. inst007); default: random.")
@click.option("--instance-seed", default=None, type=int, help="Seed for the random instance pick (default: unseeded).")
@click.option("--epochs", default="0,1,10", show_default=True,
              help="Epoch snapshots to show side by side. Only instances carrying ALL of them "
                   "are eligible. Use 'auto' for whatever the chosen instance has.")
@click.option("--grad-dir", default=None, help="Base dir for ensemble gradient dumps.")
@click.option("--susp-dir", default=None, help="Base dir for ensemble suspiciousness dumps.")
@click.option("--corr-dir", default=None, help="Base dir for ensemble correlation dumps.")
@click.option("--output", default=None, help="Output HTML path.")
@ensemble_combo_options
def main(instance, instance_seed, epochs, grad_dir, susp_dir, corr_dir, output,
         **flags):
    log, logfile = setup_logging("evaluate_ensemble")
    wanted_epochs = parse_epoch_list(epochs, allow_auto=True)
    combos = select_ensemble_combinations(collect_enabled_ensemble(flags))
    if not combos:
        log.warning("No combinations selected. Nothing to do.")
        return

    grad_base = Path(grad_dir) if grad_dir else ENSEMBLE_GRADIENTS_DIR
    susp_base = Path(susp_dir) if susp_dir else ENSEMBLE_SUSPICIOUSNESS_DIR
    corr_base = Path(corr_dir) if corr_dir else ENSEMBLE_CORRELATION_DIR
    rng = random.Random(instance_seed)

    section(log, "Plan")
    field(log, "Combinations", f"{len(combos)} selected")
    field(log, "Instance pick", instance or "random per combo (logged below)")
    field(log, "Epoch snapshots", "auto (whatever the chosen instance has)" if wanted_epochs is None
          else ", ".join(str(e) for e in wanted_epochs) + " — shown side by side in one row")

    payloads = []
    for combo in combos:
        section(log, combo.label)
        available = sorted(set(list_instances(susp_base, combo.key)) & set(list_instances(grad_base, combo.key)))
        if not available:
            log.warning("  No captured instances -- run train_ensemble.py first. Skipping.")
            continue

        # Eligible = carries every requested epoch, matching the population
        # correlate_ensemble.py used (it filters the same way).
        have = {
            inst: set(list_epochs(susp_base, combo.key, inst)) & set(list_epochs(grad_base, combo.key, inst))
            for inst in available
        }
        if wanted_epochs is None:
            instances = [i for i in available if have[i]]
        else:
            instances = [i for i in available if set(wanted_epochs).issubset(have[i])]
        if not instances:
            log.warning("  No instance carries epochs %s -- re-run train_ensemble.py "
                        "(or pass --epochs auto). Skipping.",
                        ", ".join(str(e) for e in (wanted_epochs or [])))
            continue
        if len(instances) < len(available):
            field(log, "Instances ignored",
                  f"{len(available) - len(instances)} of {len(available)} lack a requested epoch "
                  "(older/partial runs)")

        chosen = instance if instance in instances else rng.choice(instances)
        combo_epochs = sorted(have[chosen] & set(wanted_epochs)) if wanted_epochs else sorted(have[chosen])

        field(log, "Instances available", len(instances))
        field(log, "Chosen instance", f"{chosen} (of {len(instances)})")
        field(log, "Epoch snapshots", ", ".join(str(e) for e in combo_epochs))

        payload = _build_combo_payload(
            combo, instances, chosen, combo_epochs, susp_base, grad_base, corr_base, log
        )
        if payload is not None:
            payloads.append(payload)

    if not payloads:
        log.warning("Nothing to render (run train_ensemble.py + correlate_ensemble.py first).")
        return

    epoch_note = ", ".join(str(e) for e in payloads[0]["epochs"])
    out_path = Path(output) if output else ensure_dir(VISUALIZATIONS_DIR) / "ensemble_report.html"
    build_ensemble_report(
        out_path,
        combo_payloads=payloads,
        subtitle=(f"{len(payloads)} model/dataset pair(s) · epochs {epoch_note} side by side · "
                  "one randomly-chosen (logged) instance's susp/grad heatmaps · "
                  "correlation aggregated across all instances, per epoch."),
    )

    section(log, "Done")
    field(log, "Combos in report", len(payloads))
    long_field(log, "HTML report", out_path)
    long_field(log, "Run log", logfile)


if __name__ == "__main__":
    main()
