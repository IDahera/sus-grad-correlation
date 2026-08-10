#!/usr/bin/env python3
"""EXPERIMENT 2 -- step T1: one seeded instance per combo, captured every epoch.

Where the ensemble experiment asks "across many random initialisations, at a
fixed point in training", this one asks the opposite question: **how do a single
model's suspiciousness and gradient values move over a long training run?**

So: exactly ONE instance of each of the 4 combos (MLP and LeNet, on MNIST and
Fashion-MNIST), all initialised from the same ``--seed`` (42 by default, for
reproducibility), trained for ``--epochs`` (200) epochs. Gradients and
suspiciousness are captured on the held-out split **before the first epoch
(epoch 0) and after every single epoch** -- unlike experiment 1, nothing is
sub-sampled here, because the whole point is the shape of the curve.

Reuses the same computation and persistence functions as both other pipelines
(``compute_gradient_spectrum`` / ``compute_suspiciousness`` /
``save_gradients`` / ``save_suspiciousness``); the ``variant`` slot holds the
seed, so different seeds never overwrite each other:

    outputs/trajectory/gradients/<combo>/seed042/epoch_<NN>.pt
    outputs/trajectory/suspiciousness/<combo>/seed042/epoch_<NN>.pt
    outputs/trajectory/<combo>__training.csv      per-epoch accuracy + loss

A combo whose epochs are already all on disk is **skipped** -- captured values
are never regenerated unless you pass ``--overwrite``.

Run:

    python scripts/train_trajectory.py --epochs 200 --seed 42
    make train-trajectory
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from scripts._cli import (
    build_dataset_and_model,
    collect_enabled_ensemble,
    ensemble_combo_options,
    field,
    long_field,
    model_summary,
    section,
    select_ensemble_combinations,
    setup_logging,
)
from susgrad.grads import compute_gradient_spectrum
from susgrad.persistence import (
    list_epochs,
    save_gradients,
    save_suspiciousness,
    seed_run_name,
)
from susgrad.sbfl import CORE_METRIC_NAMES, METRIC_NAMES, compute_suspiciousness
from susgrad.training import evaluate, train_epochs
from susgrad.utils import (
    TRAJECTORY_DIR,
    TRAJECTORY_GRADIENTS_DIR,
    TRAJECTORY_SUSPICIOUSNESS_DIR,
    describe_mapping,
    ensure_dir,
    set_seed,
)


def _complete(grad_base: Path, susp_base: Path, combo_key: str, variant: str, epochs: int) -> bool:
    """True if epochs 0..epochs are already captured (both grad + susp)."""
    want = set(range(0, epochs + 1))
    have_grad = set(list_epochs(grad_base, combo_key, variant))
    have_susp = set(list_epochs(susp_base, combo_key, variant))
    return want.issubset(have_grad) and want.issubset(have_susp)


def _write_training_csv(path: Path, rows) -> Path:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "accuracy", "test_loss", "train_loss", "seconds"])
        for row in rows:
            writer.writerow([
                row["epoch"], f"{row['accuracy']:.4f}", f"{row['test_loss']:.6f}",
                "" if row["train_loss"] is None else f"{row['train_loss']:.6f}",
                f"{row['seconds']:.2f}",
            ])
    return path


@click.command()
@click.option("--epochs", default=200, show_default=True, help="Epochs to train the single instance.")
@click.option("--seed", default=42, show_default=True,
              help="Seed for the model init (one instance per combo, reproducible).")
@click.option("--metrics", default=",".join(CORE_METRIC_NAMES), show_default=True,
              help="Suspiciousness metrics to compute and store.")
@click.option("--batch-size", default=64, show_default=True)
@click.option("--lr", default=1e-3, show_default=True)
@click.option("--threshold", default=0.0, show_default=True, help="Suspiciousness activation threshold.")
@click.option("--dstar", "dstar_exponent", default=3, show_default=True, help="D* exponent.")
@click.option("--max-samples", default=None, type=int, help="Cap eval samples (quick runs).")
@click.option("--grad-dir", default=None, help="Base dir for gradient dumps.")
@click.option("--susp-dir", default=None, help="Base dir for suspiciousness dumps.")
@click.option("--out-dir", default=None, help="Base dir for the per-epoch training CSVs.")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True,
              help="Recompute a combo whose epochs are already fully captured.")
@ensemble_combo_options
def main(epochs, seed, metrics, batch_size, lr, threshold, dstar_exponent, max_samples,
         grad_dir, susp_dir, out_dir, overwrite, **flags):
    log, logfile = setup_logging("train_trajectory")
    metric_names = [m.strip() for m in metrics.split(",") if m.strip()]
    unknown = set(metric_names) - set(METRIC_NAMES)
    if unknown:
        raise click.BadParameter(f"Unknown metric(s) {sorted(unknown)}; choose from {list(METRIC_NAMES)}.")

    combos = select_ensemble_combinations(collect_enabled_ensemble(flags))
    if not combos:
        log.warning("No combinations selected. Nothing to do.")
        return

    grad_base = Path(grad_dir) if grad_dir else TRAJECTORY_GRADIENTS_DIR
    susp_base = Path(susp_dir) if susp_dir else TRAJECTORY_SUSPICIOUSNESS_DIR
    out_base = Path(out_dir) if out_dir else TRAJECTORY_DIR
    variant = seed_run_name(seed)

    section(log, "Plan")
    field(log, "Instances", f"1 per combo ({len(combos)} combos), all seeded with set_seed({seed})")
    field(log, "Epochs", f"{epochs} (captured at epoch 0 and after EVERY epoch -> {epochs + 1} snapshots)")
    field(log, "Measured on", "the held-out evaluation split (test_loader), not the training data")
    field(log, "Metrics", ", ".join(metric_names))
    field(log, "Run folder", variant)
    field(log, "Overwrite", overwrite)
    field(log, "Combinations", f"{len(combos)} selected: " + ", ".join(c.key for c in combos))

    for combo in combos:
        section(log, combo.label)
        if not overwrite and _complete(grad_base, susp_base, combo.key, variant, epochs):
            field(log, "Skipped", f"epochs 0..{epochs} already captured (use --overwrite to redo)")
            continue

        set_seed(seed)
        bundle, model = build_dataset_and_model(
            combo, batch_size=batch_size, seed=seed, max_samples=max_samples
        )
        field(log, "Model", model_summary(model))

        t_combo = time.perf_counter()
        rows = []

        def capture_at(epoch: int, train_loss, seconds: float) -> dict:
            grads = compute_gradient_spectrum(model, bundle.test_loader)
            susp = compute_suspiciousness(
                model, bundle.test_loader, metrics=metric_names,
                threshold=threshold, dstar_exponent=dstar_exponent,
            )
            save_gradients(grads, combo.key, variant, epoch, base_dir=grad_base)
            save_suspiciousness(susp, combo.key, variant, epoch, base_dir=susp_base)
            result = evaluate(model, bundle.test_loader)
            rows.append({
                "epoch": epoch, "accuracy": result.accuracy, "test_loss": result.loss,
                "train_loss": train_loss, "seconds": seconds,
            })
            return grads

        # Epoch 0: the untrained model, before a single gradient step.
        t0 = time.perf_counter()
        grads = capture_at(0, None, time.perf_counter() - t0)
        log.info("  - epoch  0 (untrained): acc %.2f%% (%s)", rows[-1]["accuracy"],
                 describe_mapping(grads))

        for result in train_epochs(model, bundle.train_loader, n_epochs=epochs, lr=lr, progress=False):
            capture_at(result.epoch, result.avg_loss, result.duration_s)
            log.info("  - epoch %2d/%d: train loss %.4f · acc %.2f%% · %.1fs",
                     result.epoch, epochs, result.avg_loss, rows[-1]["accuracy"],
                     result.duration_s)

        field(log, "Combo total time", f"{time.perf_counter() - t_combo:.1f}s")
        field(log, "Accuracy", f"epoch 0: {rows[0]['accuracy']:.2f}% → "
                               f"epoch {epochs}: {rows[-1]['accuracy']:.2f}%")
        long_field(log, "Training CSV",
                   _write_training_csv(out_base / f"{combo.key}__training.csv", rows))

    section(log, "Done")
    long_field(log, "Gradients dir", grad_base)
    long_field(log, "Suspiciousness dir", susp_base)
    field(log, "Next", "scripts/plot_trajectory.py (pick a neuron and plot its curves)")
    long_field(log, "Run log", logfile)


if __name__ == "__main__":
    main()
