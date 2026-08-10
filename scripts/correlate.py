#!/usr/bin/env python3
"""Correlate each suspiciousness metric with the gradient, per neuron, per variant.

For every enabled combination and every captured variant (window 1..10, 1..50,
1..100, ...), this builds each neuron's suspiciousness and gradient series across
that variant's epochs and computes a SEPARATE correlation against the gradient for
ochiai, tarantula and dstar — with each requested method (Pearson, Spearman).

    outputs/correlation/<combo>/<variant>/correlation.pt
        ->  {susp_metric: {corr_method: {layer: tensor}}}

Run AFTER capture_gradients and capture_suspiciousness:

    python scripts/correlate.py
"""

import sys
import time
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
    overwrite_option,
    section,
    setup_logging,
)
from susgrad.correlation import CORRELATIONS, compute_correlations
from susgrad.persistence import (
    correlation_path,
    list_epochs,
    list_variants,
    load_gradients,
    load_suspiciousness,
    save_correlation,
)
from susgrad.registry import select_combinations
from susgrad.sbfl import METRIC_NAMES
from susgrad.utils import CORRELATION_DIR, GRADIENTS_DIR, SUSPICIOUSNESS_DIR, describe_mapping

SUSP_METRICS = METRIC_NAMES


@click.command()
@click.option("--methods", default="pearson,spearman", show_default=True,
              help="Comma-separated correlation methods.")
@click.option("--grad-dir", default=None, help="Base dir for gradient dumps.")
@click.option("--susp-dir", default=None, help="Base dir for suspiciousness dumps.")
@click.option("--output-dir", default=None, help="Base dir for correlation output.")
@overwrite_option
@only_option
@combo_options
def main(methods, grad_dir, susp_dir, output_dir, overwrite, only, **flags):
    log, logfile = setup_logging("correlate")
    methods = [m.strip() for m in methods.split(",") if m.strip()]
    unknown = set(methods) - set(CORRELATIONS)
    if unknown:
        raise click.BadParameter(f"Unknown methods {sorted(unknown)}; choose from {list(CORRELATIONS)}.")

    combos = select_combinations(collect_enabled(flags), only)
    if not combos:
        log.warning("No combinations selected. Nothing to do.")
        return

    susp_base = Path(susp_dir) if susp_dir else SUSPICIOUSNESS_DIR
    grad_base = Path(grad_dir) if grad_dir else GRADIENTS_DIR
    corr_base = Path(output_dir) if output_dir else CORRELATION_DIR

    section(log, "Plan")
    field(log, "Suspiciousness metrics", ", ".join(SUSP_METRICS))
    field(log, "Correlation methods", ", ".join(methods))
    field(log, "Overwrite", overwrite)
    field(log, "Combinations", f"{len(combos)} selected")

    for combo in combos:
        section(log, combo.label)
        variants = sorted(set(list_variants(susp_base, combo.key)) & set(list_variants(grad_base, combo.key)))
        if not variants:
            log.warning("  No shared variants for %s — run the capture steps first.", combo.key)
            continue

        for variant in variants:
            if not overwrite and correlation_path(combo.key, variant, corr_base).exists():
                field(log, f"{variant}", "already computed (use --overwrite to redo)")
                continue

            susp_epochs = list_epochs(susp_base, combo.key, variant)
            grad_epochs = list_epochs(grad_base, combo.key, variant)
            common = sorted(set(susp_epochs) & set(grad_epochs))
            if len(common) < 2:
                log.warning("  %s: need >=2 shared epochs (have %d) — skipping.", variant, len(common))
                continue

            susp_dumps = [load_suspiciousness(combo.key, variant, e, base_dir=susp_dir) for e in common]
            grad_per_epoch = [load_gradients(combo.key, variant, e, base_dir=grad_dir) for e in common]
            metrics_present = [m for m in SUSP_METRICS if m in susp_dumps[0]]

            t0 = time.perf_counter()
            result = {
                metric: compute_correlations(
                    [d[metric] for d in susp_dumps], grad_per_epoch, methods=methods
                )
                for metric in metrics_present
            }
            compute_s = time.perf_counter() - t0

            path = save_correlation(result, combo.key, variant, base_dir=output_dir)
            any_metric = next(iter(result))
            log.info("  - %s (epochs 1..%d): corr in %.3fs (%s) -> %s",
                     variant, common[-1], compute_s,
                     describe_mapping(result[any_metric][next(iter(result[any_metric]))]),
                     path)

    section(log, "Done")
    long_field(log, "Run log", logfile)


if __name__ == "__main__":
    main()
