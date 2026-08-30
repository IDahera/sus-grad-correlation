#!/usr/bin/env python3
"""Capture per-neuron gradients after every epoch, for each enabled combination.

Trains ONCE up to the largest requested stop and stores each window ``1..stop``
**separately** as a variant (e005, e010, e020, ...). Only gradient tensors are
stored. Re-running skips complete variants unless --overwrite.

    python scripts/capture_gradients.py                      # windows 1..5, 1..10, 1..20
    python scripts/capture_gradients.py --epochs 100 --stops 10,50,100

    outputs/gradients/<combo>/<variant>/epoch_<NN>.pt   ({layer: tensor})
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from scripts._cli import (
    collect_enabled,
    combo_options,
    field,
    long_field,
    only_option,
    overwrite_option,
    parse_stops,
    run_capture,
    section,
    setup_logging,
)
from functools import partial

from susgrad.grads import compute_gradient_spectrum
from susgrad.persistence import save_gradients
from susgrad.registry import select_combinations
from susgrad.utils import GRADIENTS_DIR, describe_mapping


@click.command()
@click.option("--epochs", default=20, show_default=True, help="Default stop if --stops omitted.")
@click.option("--stops", default="5,10,20", show_default=True, help="Comma-separated window ends, e.g. 5,10,20.")
@click.option("--batch-size", default=64, show_default=True)
@click.option("--lr", default=1e-3, show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--max-samples", default=None, type=int, help="Cap samples (quick runs).")
@click.option("--output-dir", default=None, help="Base dir for gradient dumps.")
@overwrite_option
@only_option
@combo_options
def main(epochs, stops, batch_size, lr, seed, max_samples, output_dir, overwrite, only,
         **flags):
    log, logfile = setup_logging("capture_gradients")
    stop_list = parse_stops(stops, epochs)
    base_dir = Path(output_dir) if output_dir else GRADIENTS_DIR

    combos = select_combinations(collect_enabled(flags), only)
    if not combos:
        log.warning("No combinations selected. Nothing to do.")
        return

    section(log, "Plan")
    field(log, "Stops (windows)", ", ".join(f"1..{s}" for s in stop_list))
    field(log, "Train to", f"{max(stop_list)} epochs")
    field(log, "Overwrite", overwrite)
    field(log, "Combinations", f"{len(combos)} selected")

    run_capture(
        log, combos=combos, stop_list=stop_list, base_dir=base_dir, output_dir=output_dir,
        batch_size=batch_size, lr=lr, seed=seed, max_samples=max_samples, overwrite=overwrite,
        compute_fn=compute_gradient_spectrum,
        save_fn=lambda data, key, variant, epoch, out: save_gradients(data, key, variant, epoch, base_dir=out),
        summarize_fn=describe_mapping,
        kind_label="gradients",
    )

    section(log, "Done")
    long_field(log, "Run log", logfile)


if __name__ == "__main__":
    main()
