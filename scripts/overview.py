#!/usr/bin/env python3
"""Render a standalone, pretty HTML overview of the pipeline.

Data-free: it reads only the registry, so it always reflects the current
model/dataset pairs. Shows the models covered, the per-epoch training loop
(train -> compute gradient + suspiciousness -> snapshot at stops), and what the
final report evaluates/visualises.

    python scripts/overview.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from scripts._cli import field, long_field, setup_logging
from susgrad.datainspect import build_datasets_html
from susgrad.registry import COMBINATIONS, domain_of
from susgrad.sbfl import METRIC_NAMES
from susgrad.utils import VISUALIZATIONS_DIR, ensure_dir
from susgrad.viz.html import build_pipeline_overview

_KIND = {"tabular": "dense · tabular", "mlp": "dense · image", "lenet": "convnet"}


@click.command()
@click.option("--stops", default="5,10,20", show_default=True, help="Windows to depict.")
@click.option("--samples/--no-samples", default=True, show_default=True,
              help="Include live example rows / image samples in the Datasets tab.")
@click.option("--output", default=None, help="Output HTML path.")
def main(stops, samples, output):
    log, logfile = setup_logging("overview")
    stop_list = [int(s) for s in stops.split(",") if s.strip()]

    combos = [{
        "label": c.label,
        "dataset": c.dataset,
        "domain": domain_of(c),
        "kind": _KIND.get(c.model_kind, c.model_kind),
        "hidden": ", ".join(map(str, c.hidden)) if c.model_kind != "lenet" else "LeNet-5 (conv×2, fc×3)",
    } for c in COMBINATIONS]

    datasets_html = build_datasets_html(
        [c.dataset for c in COMBINATIONS], with_samples=samples
    )

    out_path = Path(output) if output else ensure_dir(VISUALIZATIONS_DIR) / "overview.html"
    build_pipeline_overview(
        out_path,
        combos=combos,
        metrics=METRIC_NAMES,
        corr_methods=("pearson", "spearman"),
        stops=stop_list,
        datasets_html=datasets_html,
        subtitle=(f"{len(combos)} model/dataset pairs · suspiciousness "
                  "(SBFL) vs gradient correlation across epoch windows."),
    )
    field(log, "Models", len(combos))
    field(log, "Live samples", samples)
    long_field(log, "Overview HTML", out_path)


if __name__ == "__main__":
    main()
