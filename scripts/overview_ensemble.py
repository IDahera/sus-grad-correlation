#!/usr/bin/env python3
"""Render a standalone, data-free HTML overview of the MAIN (ensemble) pipeline.

Shows the 4 model/dataset pairs the ensemble pipeline covers, its 3 steps
(train_ensemble -> correlate_ensemble -> evaluate_ensemble), MNIST/Fashion-MNIST
dataset info, and the suspiciousness metrics listed in a pretty format --
WITHOUT paper citations (see susgrad/viz/html.py's "Formulas" tab on the
secondary pipeline's overview for the cited version).

    python scripts/overview_ensemble.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from scripts._cli import field, long_field, parse_epoch_list, setup_logging
from susgrad.datainspect import build_datasets_html
from susgrad.registry import domain_of, ensemble_combinations
from susgrad.sbfl import CORE_METRIC_NAMES
from susgrad.utils import VISUALIZATIONS_DIR, ensure_dir
from susgrad.viz.ensemble_html import build_ensemble_overview

_KIND = {"tabular": "dense · tabular", "mlp": "dense · image", "lenet": "convnet"}


@click.command()
@click.option("--instances", default=100, show_default=True, help="Instances per combo (for the overview text).")
@click.option("--epochs", default=10, show_default=True, help="Epochs per instance (for the overview text).")
@click.option("--capture-epochs", default="0,1,10", show_default=True,
              help="Captured epoch snapshots (for the overview text).")
@click.option("--samples/--no-samples", default=True, show_default=True,
              help="Include live example images in the Datasets tab.")
@click.option("--output", default=None, help="Output HTML path.")
def main(instances, epochs, capture_epochs, samples, output):
    log, logfile = setup_logging("overview_ensemble")
    captured = parse_epoch_list(capture_epochs, max_epoch=epochs)

    combos = [{
        "label": c.label,
        "dataset": c.dataset,
        "domain": domain_of(c),
        "kind": _KIND.get(c.model_kind, c.model_kind),
        "hidden": ", ".join(map(str, c.hidden)) if c.model_kind != "lenet" else "LeNet-5 (conv×2, fc×3)",
    } for c in ensemble_combinations()]

    datasets_html = build_datasets_html(["mnist", "fmnist"], with_samples=samples)

    out_path = Path(output) if output else ensure_dir(VISUALIZATIONS_DIR) / "ensemble_overview.html"
    build_ensemble_overview(
        out_path,
        combos=combos,
        instances=instances,
        epochs=epochs,
        metrics=CORE_METRIC_NAMES,
        capture_epochs=captured,
        datasets_html=datasets_html,
        subtitle=(f"{len(combos)} model/dataset pairs · {instances} instances × {epochs} epochs each · "
                  f"captured at epochs {', '.join(str(e) for e in captured)} · "
                  "suspiciousness (SBFL) vs gradient, correlated across instances, per epoch."),
    )
    field(log, "Models", len(combos))
    field(log, "Instances", instances)
    field(log, "Epochs", epochs)
    field(log, "Captured at epochs", ", ".join(str(e) for e in captured))
    long_field(log, "Overview HTML", out_path)
    long_field(log, "Run log", logfile)


if __name__ == "__main__":
    main()
