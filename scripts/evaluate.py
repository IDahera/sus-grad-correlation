#!/usr/bin/env python3
"""Evaluate (visualise) suspiciousness, gradient and their correlation.

THE FINAL PIPELINE STEP. Produces ONE self-contained HTML with nested tabs:
  * a tab per model/dataset combination,
  * inside it, a tab per captured variant (window 1..10, 1..50, 1..100, ...), and
  * inside that, a sub-tab per suspiciousness metric (ochiai / tarantula / dstar).

Each metric sub-tab shows, per layer: suspiciousness / gradient / correlation
heatmaps (one conv channel at true resolution), a decrement-comparison plot, and
the correlation distribution histogram. Suspiciousness & gradient heatmaps are the
snapshot at the variant's final epoch; the correlation spans that whole window.

    python scripts/evaluate.py --channel 0
"""

import sys
from pathlib import Path

# Allow running directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from scripts._cli import (
    collect_enabled,
    combo_options,
    field,
    long_field,
    only_option,
    section,
    setup_logging,
)
from susgrad.persistence import (
    list_epochs,
    list_variants,
    load_correlation,
    load_gradients,
    load_suspiciousness,
    variant_stop,
)
from susgrad.registry import select_combinations
from susgrad.sbfl import BOUNDED_METRICS, METRIC_NAMES
from susgrad.utils import GRADIENTS_DIR, SUSPICIOUSNESS_DIR, VISUALIZATIONS_DIR, ensure_dir
from susgrad.viz.html import build_tabbed_report, colormap_legend_html
from susgrad.viz.render import render_combo_sections
from susgrad.viz.transform import VisualizationError

ALL_METRICS = METRIC_NAMES


@click.command()
@click.option("--epoch", default=0, show_default=True,
              help="Snapshot epoch for susp/grad heatmaps (0 = each variant's last).")
@click.option("--corr-method", type=click.Choice(["pearson", "spearman"]),
              default="pearson", show_default=True, help="Correlation method to show.")
@click.option("--channel", default=0, show_default=True,
              help="Conv channel to display (clamped per layer).")
@click.option("--max-neurons", default=2_000_000, show_default=True)
@click.option("--grad-dir", default=None)
@click.option("--susp-dir", default=None)
@click.option("--corr-dir", default=None)
@click.option("--output", default=None, help="Output HTML path (single combined file).")
@only_option
@combo_options
def main(epoch, corr_method, channel, max_neurons,
         grad_dir, susp_dir, corr_dir, output, only, **flags):
    log, logfile = setup_logging("evaluate")
    combos = select_combinations(collect_enabled(flags), only)
    if not combos:
        log.warning("No combinations selected. Nothing to do.")
        return

    susp_base = Path(susp_dir) if susp_dir else SUSPICIOUSNESS_DIR
    grad_base = Path(grad_dir) if grad_dir else GRADIENTS_DIR

    section(log, "Plan")
    field(log, "Snapshot epoch", epoch if epoch else "each variant's last")
    field(log, "Metrics (sub-tabs)", ", ".join(ALL_METRICS))
    field(log, "Correlation method", corr_method)
    field(log, "Conv channel", channel)
    field(log, "Combinations (tabs)", f"{len(combos)} selected")

    model_tabs = []
    for combo in combos:
        section(log, combo.label)
        variants = sorted(
            set(list_variants(susp_base, combo.key)) & set(list_variants(grad_base, combo.key)),
            key=variant_stop,
        )
        if not variants:
            log.warning("  Skipping %s: no captured variants found.", combo.key)
            continue

        variant_tabs = []
        for variant in variants:
            avail = list_epochs(susp_base, combo.key, variant)
            if not avail:
                continue
            snap = max(avail) if not epoch else min(epoch, max(avail))

            try:
                susp_all = load_suspiciousness(combo.key, variant, snap, base_dir=susp_dir)
                grad = load_gradients(combo.key, variant, snap, base_dir=grad_base)
            except FileNotFoundError as exc:
                log.warning("  %s/%s: %s", combo.key, variant, exc)
                continue

            corr_by_metric = None
            try:
                nested = load_correlation(combo.key, variant, base_dir=corr_dir)
                corr_by_metric = {
                    metric: methods[corr_method]
                    for metric, methods in nested.items() if corr_method in methods
                } or None
            except FileNotFoundError:
                pass

            try:
                sections_by_metric = render_combo_sections(
                    susp_all=susp_all, grad_by_layer=grad, corr_by_metric=corr_by_metric,
                    metrics=ALL_METRICS, bounded_metrics=BOUNDED_METRICS,
                    channel=channel, corr_method=corr_method, max_neurons=max_neurons,
                )
            except VisualizationError as exc:
                log.error("  %s/%s: %s", combo.key, variant, exc)
                continue

            variant_tabs.append({
                "label": f"1..{variant_stop(variant)} (snap e{snap})",
                "children": [{"name": m, "label": m, "html": sections_by_metric[m]}
                             for m in sections_by_metric],
            })
            log.info("  %s/%s: %d metric sub-tabs (snapshot epoch %d).",
                     combo.key, variant, len(sections_by_metric), snap)

        if variant_tabs:
            model_tabs.append({"label": combo.label, "children": variant_tabs})

    if not model_tabs:
        log.warning("Nothing could be rendered (run the capture steps first).")
        return

    out_path = Path(output) if output else ensure_dir(VISUALIZATIONS_DIR) / "report.html"
    build_tabbed_report(
        out_path,
        title="susgrad report",
        subtitle=(f"{len(model_tabs)} model(s) · variant tabs = epoch windows · "
                  f"metric sub-tabs · correlation: {corr_method} · conv channel: {channel}. "
                  f"Susp/grad heatmaps are a single-epoch snapshot; correlation spans the window."),
        tabs=model_tabs,
        intro_html=colormap_legend_html(),
    )

    section(log, "Done")
    field(log, "Models in report", len(model_tabs))
    long_field(log, "HTML report", out_path)
    long_field(log, "Run log", logfile)


if __name__ == "__main__":
    main()
