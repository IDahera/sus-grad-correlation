#!/usr/bin/env python3
"""Train and evaluate every enabled model/dataset combination.

Reads the combinations from ``susgrad.registry``. Each pair has an on/off flag
(default on), so you can pick a subset, e.g.:

    python scripts/train_models.py --no-lenet-mnist          # skip MNIST
    python scripts/train_models.py --only tabular            # only the MLPs
    make train-models EPOCHS=20

Logs the planned work before iterating, then per-epoch metrics and a final
accuracy line per combination. Also appends a results CSV and writes an HTML
training report (per-epoch loss/accuracy charts + summary table) to
outputs/training_logs/.
"""

import csv
import sys
import time
from pathlib import Path

# Allow running directly (python scripts/train_models.py) before importing scripts.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

import torch.nn as nn

from scripts._cli import (
    build_dataset_and_model,
    collect_enabled,
    combo_options,
    field,
    long_field,
    model_summary,
    only_option,
    section,
    setup_logging,
)
from susgrad.persistence import save_model
from susgrad.registry import select_combinations
from susgrad.training import evaluate, train_epochs
from susgrad.utils import TRAINING_LOGS_DIR, ensure_dir
from susgrad.viz.html import build_training_report


def _count_weight_layers(model) -> int:
    return sum(1 for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d)))


@click.command()
@click.option("--epochs", default=20, show_default=True, help="Epochs per model.")
@click.option("--batch-size", default=64, show_default=True)
@click.option("--lr", default=1e-3, show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--max-samples", default=None, type=int, help="Cap samples (quick runs).")
@click.option("--output-dir", default=None, help="Where to save models.")
@only_option
@combo_options
def main(epochs, batch_size, lr, seed, max_samples, output_dir, only, **flags):
    log, logfile = setup_logging("train_models")
    combos = select_combinations(collect_enabled(flags), only)

    if not combos:
        log.warning("No combinations selected. Nothing to do.")
        return

    section(log, "Plan")
    field(log, "Combinations", f"{len(combos)} selected")
    for c in combos:
        log.info("  - %s  (dataset=`%s`, epochs=%d)", c.label, c.dataset, epochs)

    logs_dir = ensure_dir(TRAINING_LOGS_DIR)
    results_path = logs_dir / "training_results.csv"
    write_header = not results_path.exists()
    csv_file = open(results_path, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(
            ["timestamp", "combo", "dataset", "model", "weight_layers", "params",
             "epochs", "test_accuracy", "test_loss", "total_train_s", "avg_epoch_s"]
        )

    runs = []  # collected for the HTML report

    for combo in combos:
        section(log, f"{combo.label}")
        bundle, model = build_dataset_and_model(
            combo, batch_size=batch_size, seed=seed, max_samples=max_samples
        )
        n_layers = _count_weight_layers(model)
        n_params = sum(p.numel() for p in model.parameters())
        field(log, "Dataset", f"{combo.dataset} ({bundle.num_classes} classes, input {bundle.input_shape})")
        field(log, "Model", model_summary(model))

        log.info("")  # blank line before the epoch list
        history = []
        epoch_times = []
        for result in train_epochs(
            model, bundle.train_loader, n_epochs=epochs, lr=lr, progress=False
        ):
            # Evaluate after each epoch to record accuracy/loss curves.
            ev = evaluate(model, bundle.test_loader)
            epoch_times.append(result.duration_s)
            history.append({
                "epoch": result.epoch,
                "train_loss": result.avg_loss,
                "test_loss": ev.loss,
                "test_acc": ev.accuracy,
                "duration_s": result.duration_s,
            })
            log.info("  - epoch %2d/%d — train_loss=%.4f test_acc=%.2f%% "
                     "test_loss=%.4f (%.2fs)",
                     result.epoch, epochs, result.avg_loss, ev.accuracy,
                     ev.loss, result.duration_s)

        final_acc = history[-1]["test_acc"] if history else float("nan")
        final_loss = history[-1]["test_loss"] if history else float("nan")
        total = sum(epoch_times)
        avg = total / len(epoch_times) if epoch_times else 0.0
        log.info("")
        field(log, "Final test accuracy", f"{final_acc:.2f}%")
        field(log, "Final test loss", f"{final_loss:.4f}")
        field(log, "Architecture", f"{n_layers} weight layers, {n_params:,} params")
        field(log, "Training time", f"{total:.1f}s total, {avg:.2f}s/epoch ({epochs} epochs)")

        path = save_model(model, f"{combo.key}_e{epochs}", directory=output_dir)
        long_field(log, "Saved model", path)

        runs.append({
            "label": combo.label, "key": combo.key, "dataset": combo.dataset,
            "model_type": type(model).__name__, "weight_layers": n_layers,
            "params": n_params, "epochs": epochs,
            "final_accuracy": final_acc, "final_loss": final_loss,
            "history": history,
        })

        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), combo.key, combo.dataset,
            type(model).__name__, n_layers, n_params, epochs,
            f"{final_acc:.2f}", f"{final_loss:.4f}", f"{total:.1f}", f"{avg:.2f}",
        ])
        csv_file.flush()

    csv_file.close()

    # HTML training report (per-epoch loss/accuracy charts + summary table).
    report_path = logs_dir / f"training_report_{time.strftime('%Y%m%d_%H%M%S')}.html"
    build_training_report(
        report_path,
        runs=runs,
        title="Training report",
        subtitle=(
            f"{len(runs)} model/dataset variant(s) · per-epoch test metrics · "
            f"generated {time.strftime('%Y-%m-%d %H:%M:%S')}"
        ),
    )

    section(log, "Done")
    long_field(log, "Results CSV", results_path)
    long_field(log, "HTML report", report_path)
    long_field(log, "Run log", logfile)


if __name__ == "__main__":
    main()
