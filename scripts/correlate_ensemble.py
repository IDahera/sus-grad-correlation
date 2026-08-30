#!/usr/bin/env python3
"""MAIN (ensemble) PIPELINE -- step 2: correlate suspiciousness vs. gradient ACROSS INSTANCES.

The key difference from the secondary pipeline's ``correlate.py``: there, one
instance is trained once and the correlation is measured across the EPOCH axis.
Here, many instances are each trained once (see ``train_ensemble.py``), and the
correlation is measured, separately for each captured epoch snapshot (base case:
0, 1, 10), ACROSS THE POPULATION OF INSTANCES -- i.e. "do neurons that are
suspicious in one randomly-initialised model also tend to be the ones with high
gradient, at a given point in training, across many independent random
initialisations?"

This reuses ``susgrad.correlation.compute_correlations`` completely unchanged:
that function just stacks a list of ``{layer: tensor}`` dicts along a new axis
and correlates along it -- it doesn't care whether the list is ordered by epoch
(secondary pipeline) or by instance (here).

    outputs/ensemble/correlation/<combo>/epoch<EE>/correlation.pt
        -> {susp_metric: {corr_method: {layer: tensor}}}   (correlated across instances)
    outputs/ensemble/correlation/<combo>/epoch<EE>/population.json
        -> exactly which instances that file was correlated over, so a later run
           with a different population recomputes instead of silently mixing the two
    outputs/ensemble/correlation_stats.csv
        -> the same per-layer statistics as the report, flat: one row per
           (combo, layer, metric, method, epoch) -- what a LaTeX table is built from
    outputs/ensemble/correlation/<combo>/correlation_report.md
        -> the same thing in text: per-epoch statistics (mean |r|, median,
           % strong, % zero) plus ASCII heatmaps, with epochs printed one after
           another so 0 vs 1 vs 10 can be compared by eye.

Run AFTER train_ensemble.py:

    python scripts/correlate_ensemble.py --epochs 0,1,10
"""

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click
import numpy as np

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
from susgrad.correlation import CORRELATIONS, compute_correlations
from susgrad.persistence import (
    correlation_path,
    epoch_snapshot_name,
    list_epochs,
    list_instances,
    load_correlation,
    load_gradients,
    load_suspiciousness,
    save_correlation,
    seed_manifest_path,
)
from susgrad.sbfl import METRIC_NAMES
from susgrad.utils import (
    ENSEMBLE_CORRELATION_DIR,
    ENSEMBLE_GRADIENTS_DIR,
    ENSEMBLE_SEEDS_DIR,
    ENSEMBLE_SUSPICIOUSNESS_DIR,
    describe_mapping,
    ensure_dir,
)
from susgrad.viz import build_heatmap
from susgrad.viz.textmap import DIV_LEGEND, correlation_summary, markdown_table, text_heatmap


def _population_path(corr_base: Path, combo_key: str, variant: str) -> Path:
    """Sidecar recording WHICH instances a stored correlation was computed over."""
    return corr_base / combo_key / variant / "population.json"


def _reusable(path: Path, pop_path: Path, instances, methods) -> bool:
    """True if a stored correlation.pt was computed over exactly this population.

    Without this check, re-running with a different ``--instances`` count (or
    after adding instances) would silently mix populations: epoch 0 correlated
    over yesterday's 50 models, epoch 10 over today's 20. The sidecar is written
    next to every correlation this script saves; a missing one means "unknown
    provenance", which we treat as not reusable.
    """
    if not path.exists() or not pop_path.exists():
        return False
    try:
        meta = json.loads(pop_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return meta.get("instances") == list(instances) and set(methods).issubset(meta.get("methods", []))


def _write_population(pop_path: Path, instances, methods, epoch: int) -> None:
    ensure_dir(pop_path.parent)
    pop_path.write_text(json.dumps({
        "epoch": int(epoch),
        "n_instances": len(instances),
        "instances": list(instances),
        "methods": list(methods),
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2) + "\n", encoding="utf-8")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _f(x: float) -> str:
    return f"{x:+.4f}"


def _pooled(nested: dict, metric: str, method: str) -> np.ndarray:
    """All layers' correlation values for one metric+method, flattened into one vector."""
    layers = nested.get(metric, {}).get(method, {})
    if not layers:
        return np.zeros(0)
    return np.concatenate([t.detach().cpu().reshape(-1).numpy() for t in layers.values()])


def _summary_tables(log, epochs, methods, metrics, per_epoch):
    """Pooled mean |r| per (metric x epoch), one table per method -> markdown blocks.

    Also logged (compactly) to the run log, because this is the single number the
    experiment is actually about: how the susp/gradient coupling changes between
    the untrained model, one epoch in, and ten epochs in.
    """
    blocks = []
    for method in methods:
        rows = []
        for metric in metrics:
            stats = [correlation_summary(_pooled(per_epoch[e], metric, method)) for e in epochs]
            rows.append([f"`{metric}`"]
                        + [f"{s['mean_abs']:.4f}" for s in stats]
                        + [_f(s["mean"]) for s in stats]
                        + [_pct(s["frac_strong"]) for s in stats])
        header = (["metric"]
                  + [f"mean\\|r\\| @ e{e}" for e in epochs]
                  + [f"mean r @ e{e}" for e in epochs]
                  + [f"% \\|r\\|>0.5 @ e{e}" for e in epochs])
        blocks.append(f"### {method}\n\nAll layers pooled, correlated across instances.\n\n"
                      + markdown_table(header, rows) + "\n")
        log.info("\n**%s** — pooled mean |r| per metric (all layers, all neurons):\n", method)
        log.info("%s\n", markdown_table(["metric"] + [f"e{e}" for e in epochs],
                                        [[r[0]] + r[1:1 + len(epochs)] for r in rows]))
    return blocks


def _per_layer_blocks(epochs, methods, metrics, layers, per_epoch):
    """One mean|r| table per (method, metric): rows = layer, columns = epoch."""
    blocks = []
    for method in methods:
        for metric in metrics:
            rows = []
            for layer in layers:
                row = [f"`{layer}`"]
                n_shown = ""
                for epoch in epochs:
                    tensor = per_epoch[epoch].get(metric, {}).get(method, {}).get(layer)
                    if tensor is None:
                        row.append("—")
                        continue
                    stats = correlation_summary(tensor.detach().cpu().numpy())
                    n_shown = str(stats["n"])
                    row.append(f"{stats['mean_abs']:.4f}")
                rows.append(row + [n_shown])
            blocks.append(
                f"### {method} · `{metric}`\n\n"
                + markdown_table(["layer"] + [f"mean\\|r\\| @ e{e}" for e in epochs] + ["neurons"], rows)
                + "\n"
            )
    return blocks


def _detail_block(epochs, methods, metrics, layers, per_epoch):
    """One wide, greppable table per method: a row per (metric, layer, epoch)."""
    blocks = []
    header = ["metric", "layer", "epoch", "n", "mean r", "mean |r|", "median",
              "std", "% |r|>0.5", "% r>0", "% r<0", "% r=0", "min", "max"]
    for method in methods:
        rows = []
        for metric in metrics:
            for layer in layers:
                for epoch in epochs:
                    tensor = per_epoch[epoch].get(metric, {}).get(method, {}).get(layer)
                    if tensor is None:
                        continue
                    s = correlation_summary(tensor.detach().cpu().numpy())
                    rows.append([
                        f"`{metric}`", f"`{layer}`", epoch, s["n"], _f(s["mean"]),
                        f"{s['mean_abs']:.4f}", _f(s["median"]), f"{s['std']:.4f}",
                        _pct(s["frac_strong"]), _pct(s["frac_pos"]), _pct(s["frac_neg"]),
                        _pct(s["frac_zero"]), _f(s["min"]), _f(s["max"]),
                    ])
        blocks.append(f"### {method}\n\n" + markdown_table(header, rows) + "\n")
    return blocks


def _heatmap_blocks(epochs, methods, metrics, layers, per_epoch, *, channels, max_rows, max_cols):
    """ASCII correlation heatmaps, epochs printed consecutively for easy comparison."""
    blocks = [
        "Each map is one layer's per-neuron correlation, drawn on the fixed −1..+1 "
        "scale. The epochs are printed one after another (same layer, same metric, "
        "same method) so the change over training is readable straight down the page.\n",
        f"```text\n{DIV_LEGEND}\n```\n",
    ]
    for method in methods:
        blocks.append(f"### {method}\n")
        for metric in metrics:
            blocks.append(f"#### `{metric}` · {method}\n")
            for layer in layers:
                sample = next((per_epoch[e].get(metric, {}).get(method, {}).get(layer) for e in epochs), None)
                if sample is None:
                    continue
                n_channels = int(sample.shape[0]) if sample.dim() >= 3 else 1
                for channel in range(min(n_channels, channels) if n_channels > 1 else 1):
                    label = f"`{layer}`" + (f" · channel {channel}/{n_channels}" if n_channels > 1 else "")
                    blocks.append(f"**{label}**\n")
                    maps = []
                    for epoch in epochs:
                        tensor = per_epoch[epoch].get(metric, {}).get(method, {}).get(layer)
                        if tensor is None:
                            continue
                        grid = build_heatmap(layer, tensor, channel=channel).grid
                        stats = correlation_summary(tensor.detach().cpu().numpy())
                        maps.append(
                            f"epoch {epoch:>2}  mean|r|={stats['mean_abs']:.4f}  "
                            f"mean r={_f(stats['mean'])}  median={_f(stats['median'])}  "
                            f"|r|>0.5={_pct(stats['frac_strong'])}  r=0={_pct(stats['frac_zero'])}\n"
                            + text_heatmap(grid, diverging=True, max_rows=max_rows, max_cols=max_cols)
                        )
                    blocks.append("```text\n" + "\n\n".join(maps) + "\n```\n")
    return blocks


def _stats_rows(combo, instances, epochs, methods, metrics, layers, per_epoch):
    """The per-layer correlation statistics as flat rows (the markdown report's
    detail table, in a form a table generator or a stats tool can read)."""
    rows = []
    for method in methods:
        for metric in metrics:
            for layer in layers:
                for epoch in epochs:
                    tensor = per_epoch[epoch].get(metric, {}).get(method, {}).get(layer)
                    if tensor is None:
                        continue
                    s = correlation_summary(tensor.detach().cpu().numpy())
                    rows.append({
                        "combo": combo.key, "layer": layer, "metric": metric,
                        "method": method, "epoch": epoch, "instances": len(instances),
                        "n": s["n"], "mean_r": f"{s['mean']:.6f}",
                        "mean_abs_r": f"{s['mean_abs']:.6f}", "median_r": f"{s['median']:.6f}",
                        "std": f"{s['std']:.6f}", "frac_strong": f"{s['frac_strong']:.6f}",
                        "frac_pos": f"{s['frac_pos']:.6f}", "frac_neg": f"{s['frac_neg']:.6f}",
                        "frac_zero": f"{s['frac_zero']:.6f}",
                        "min_r": f"{s['min']:.6f}", "max_r": f"{s['max']:.6f}",
                    })
    return rows


STATS_COLUMNS = ["combo", "layer", "metric", "method", "epoch", "instances", "n",
                 "mean_r", "mean_abs_r", "median_r", "std", "frac_strong",
                 "frac_pos", "frac_neg", "frac_zero", "min_r", "max_r"]


def _write_stats_csv(path: Path, rows) -> Path:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=STATS_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_report(path, *, combo, instances, epochs, methods, metrics, layers, per_epoch,
                  seeds_file, log, heatmaps, channels, max_rows, max_cols):
    head = [
        f"# Correlation report — {combo.label} (`{combo.key}`)",
        "",
        f"- **Population:** {len(instances)} independently, randomly-initialised instances "
        f"(`{instances[0]}`..`{instances[-1]}`)",
        f"- **Epoch snapshots:** {', '.join(str(e) for e in epochs)}  (0 = before any training)",
        "- **Correlation axis:** across instances, separately per epoch snapshot",
        "- **Measured on:** the held-out evaluation split, per neuron",
        f"- **Suspiciousness metrics:** {', '.join(metrics)}",
        f"- **Correlation methods:** {', '.join(methods)}",
        f"- **Layers:** {', '.join(layers)}",
        f"- **Seeds:** `{seeds_file}`",
        "",
        "## Average correlation per epoch (all layers pooled)",
        "",
        "`mean |r|` is the direction-agnostic coupling strength; `mean r` is the "
        "signed average (near zero either means no coupling *or* positive and "
        "negative neurons cancelling out — read them together).",
        "",
    ]
    body = _summary_tables(log, epochs, methods, metrics, per_epoch)
    body += ["## Per-layer average correlation", ""] + _per_layer_blocks(
        epochs, methods, metrics, layers, per_epoch)
    body += ["## Full per-layer statistics", ""] + _detail_block(
        epochs, methods, metrics, layers, per_epoch)
    if heatmaps:
        body += ["## Correlation heatmaps (text)", ""] + _heatmap_blocks(
            epochs, methods, metrics, layers, per_epoch,
            channels=channels, max_rows=max_rows, max_cols=max_cols)

    path.write_text("\n".join(head + body), encoding="utf-8")
    return path


@click.command()
@click.option("--methods", default="pearson,spearman", show_default=True,
              help="Comma-separated correlation methods.")
@click.option("--epochs", default="0,1,10", show_default=True,
              help="Epoch snapshots to correlate; only instances that have ALL of them are "
                   "used, so the population is identical at every epoch. Use 'auto' for "
                   "whatever epochs every captured instance happens to share.")
@click.option("--report/--no-report", default=True, show_default=True,
              help="Write the text report (statistics + ASCII heatmaps) per combo.")
@click.option("--report-heatmaps/--no-report-heatmaps", default=True, show_default=True,
              help="Include the ASCII heatmaps in the report (statistics are always included).")
@click.option("--report-channels", default=1, show_default=True,
              help="Conv layers: how many channels to draw as text heatmaps.")
@click.option("--report-max-rows", default=48, show_default=True, help="Text heatmap height cap.")
@click.option("--report-max-cols", default=72, show_default=True, help="Text heatmap width cap.")
@click.option("--grad-dir", default=None, help="Base dir for ensemble gradient dumps.")
@click.option("--susp-dir", default=None, help="Base dir for ensemble suspiciousness dumps.")
@click.option("--output-dir", default=None, help="Base dir for ensemble correlation output.")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True,
              help="Recompute epoch snapshots that are already correlated.")
@ensemble_combo_options
def main(methods, epochs, report, report_heatmaps, report_channels, report_max_rows,
         report_max_cols, grad_dir, susp_dir, output_dir, overwrite, **flags):
    log, logfile = setup_logging("correlate_ensemble")
    methods = [m.strip() for m in methods.split(",") if m.strip()]
    unknown = set(methods) - set(CORRELATIONS)
    if unknown:
        raise click.BadParameter(f"Unknown methods {sorted(unknown)}; choose from {list(CORRELATIONS)}.")
    wanted_epochs = parse_epoch_list(epochs, allow_auto=True)

    combos = select_ensemble_combinations(collect_enabled_ensemble(flags))
    if not combos:
        log.warning("No combinations selected. Nothing to do.")
        return

    grad_base = Path(grad_dir) if grad_dir else ENSEMBLE_GRADIENTS_DIR
    susp_base = Path(susp_dir) if susp_dir else ENSEMBLE_SUSPICIOUSNESS_DIR
    corr_base = Path(output_dir) if output_dir else ENSEMBLE_CORRELATION_DIR
    # corr_base.parent is shared by both gradient kinds, so this summary table
    # needs the kind in its NAME or the second run overwrites the first's.
    stats_csv_name = "correlation_stats.csv"

    section(log, "Plan")
    field(log, "Suspiciousness metrics", ", ".join(METRIC_NAMES))
    field(log, "Correlation methods", ", ".join(methods))
    field(log, "Correlation axis", "instances (population), separately per epoch snapshot")
    field(log, "Epoch snapshots", "auto (whatever all instances share)" if wanted_epochs is None
          else ", ".join(str(e) for e in wanted_epochs))
    field(log, "Text report", f"{report} (heatmaps: {report_heatmaps})")
    field(log, "Overwrite", overwrite)
    field(log, "Combinations", f"{len(combos)} selected")

    stats_rows = []
    for combo in combos:
        section(log, combo.label)

        available = sorted(set(list_instances(susp_base, combo.key)) & set(list_instances(grad_base, combo.key)))
        if len(available) < 2:
            log.warning("  Need >=2 instances (have %d) -- run train_ensemble.py first. Skipping.", len(available))
            continue

        # Epochs captured per instance, in BOTH the gradient and suspiciousness dumps.
        have = {
            inst: set(list_epochs(susp_base, combo.key, inst)) & set(list_epochs(grad_base, combo.key, inst))
            for inst in available
        }
        if wanted_epochs is None:
            common_epochs = sorted(set.intersection(*have.values()))
            instances = available
        else:
            # Keep only instances that carry every requested epoch, so the
            # population being correlated is identical at each epoch snapshot --
            # a stale instance from an earlier run must not silently shrink the
            # epoch list for everyone else.
            instances = [i for i in available if set(wanted_epochs).issubset(have[i])]
            common_epochs = list(wanted_epochs) if instances else []
            dropped = len(available) - len(instances)
            if dropped:
                field(log, "Instances ignored",
                      f"{dropped} of {len(available)} lack one of epochs "
                      f"{', '.join(str(e) for e in wanted_epochs)} (older/partial runs)")

        if len(instances) < 2 or not common_epochs:
            log.warning("  No usable population (%d instances, epochs %s) -- skipping.",
                        len(instances), common_epochs or "none")
            continue

        field(log, "Instances", f"{len(instances)} ({instances[0]}..{instances[-1]})")
        field(log, "Epoch snapshots", ", ".join(str(e) for e in common_epochs))

        per_epoch, layers, metrics_present = {}, [], None
        for epoch in common_epochs:
            variant = epoch_snapshot_name(epoch)
            path = correlation_path(combo.key, variant, corr_base)
            pop_path = _population_path(corr_base, combo.key, variant)

            if not overwrite and _reusable(path, pop_path, instances, methods):
                result = load_correlation(combo.key, variant, base_dir=corr_base)
                field(log, f"epoch {epoch}", "already computed for this exact population "
                                             "(use --overwrite to redo); reused for the report")
            else:
                if path.exists():
                    field(log, f"epoch {epoch}", "recomputing: stored correlation was over a "
                                                 "different instance population or method set")
                susp_dumps = [load_suspiciousness(combo.key, inst, epoch, base_dir=susp_base) for inst in instances]
                grad_dumps = [load_gradients(combo.key, inst, epoch, base_dir=grad_base) for inst in instances]
                metrics_here = [m for m in METRIC_NAMES if m in susp_dumps[0]]
                t0 = time.perf_counter()
                result = {
                    metric: compute_correlations(
                        [d[metric] for d in susp_dumps], grad_dumps, methods=methods
                    )
                    for metric in metrics_here
                }
                compute_s = time.perf_counter() - t0
                path = save_correlation(result, combo.key, variant, base_dir=corr_base)
                _write_population(pop_path, instances, methods, epoch)
                any_metric = next(iter(result))
                log.info(
                    "  - epoch %2d (across %d instances): corr in %.3fs (%s) -> %s",
                    epoch, len(instances), compute_s,
                    describe_mapping(result[any_metric][next(iter(result[any_metric]))]),
                    path,
                )

            per_epoch[epoch] = result
            present = [m for m in METRIC_NAMES if m in result]
            metrics_present = present if metrics_present is None else [m for m in metrics_present if m in present]
            if not layers:
                first = result[present[0]]
                layers = list(first[next(iter(first))])

        if not report or not metrics_present:
            continue

        methods_present = [m for m in methods
                           if all(m in per_epoch[e].get(metrics_present[0], {}) for e in common_epochs)]
        stats_rows += _stats_rows(combo, instances, common_epochs, methods_present,
                                  metrics_present, layers, per_epoch)
        report_path = ensure_dir(corr_base / combo.key) / "correlation_report.md"
        _write_report(
            report_path, combo=combo, instances=instances, epochs=common_epochs,
            methods=methods_present, metrics=metrics_present, layers=layers,
            per_epoch=per_epoch, seeds_file=seed_manifest_path(combo.key, ENSEMBLE_SEEDS_DIR),
            log=log, heatmaps=report_heatmaps, channels=report_channels,
            max_rows=report_max_rows, max_cols=report_max_cols,
        )
        long_field(log, "Text report", report_path)

    section(log, "Done")
    if stats_rows:
        long_field(log, "Correlation statistics (CSV)",
                   _write_stats_csv(corr_base.parent / stats_csv_name, stats_rows))
    long_field(log, "Run log", logfile)


if __name__ == "__main__":
    main()
