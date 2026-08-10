#!/usr/bin/env python3
"""MAIN (ensemble) PIPELINE -- step 1: train many freshly-initialised instances.

Unlike the secondary pipeline (one instance per combo, trained once over many
epochs), this trains ``--instances`` SEPARATE instances of each of the 4 fixed
combos (MLP and LeNet, separately, on MNIST and Fashion-MNIST -- see
``susgrad.registry.ensemble_combinations``). Each instance gets its own fresh
random initialisation (``set_seed(seed + instance_index)``, so instances differ
from each other but the whole run is reproducible from ``--seed``).

Base case: **100 instances, trained 10 epochs each, captured at epochs 0, 1, 10**
(``--capture-epochs 0,1,10``) -- a population that size is what makes the
across-instance correlation statistically meaningful:

    * epoch 0  -- before ANY training, i.e. the raw random initialisation,
    * epoch 1  -- after one epoch,
    * epoch 10 -- after ten.

Capturing is decoupled from training: each instance is trained ONCE, straight
through to the last requested epoch, and gradients + suspiciousness are computed
in between, only at the requested epochs. Nothing is trained twice and no
snapshot is computed that nobody asked for.

Both quantities are measured on the **held-out evaluation split**
(``bundle.test_loader``) -- never on the training data. Each instance's
**accuracy and loss on that same split** are recorded at every captured epoch,
so every correlation can be read against how well the models actually performed.

Capture reuses the exact same computation functions as the secondary pipeline
(``compute_gradient_spectrum`` / ``compute_suspiciousness``) and the exact same
save functions (``save_gradients`` / ``save_suspiciousness``) -- just with the
``variant`` slot repurposed as an *instance* id instead of an epoch window:

    outputs/ensemble/gradients/<combo>/<inst>/epoch_<NN>.pt       ({layer: tensor})
    outputs/ensemble/suspiciousness/<combo>/<inst>/epoch_<NN>.pt  ({metric: {layer: tensor}})

The seed that produced each instance -- and its per-epoch accuracy -- is recorded
separately, per combo:

    outputs/ensemble/seeds/<combo>.json    machine-readable manifest
    outputs/ensemble/seeds/seeds.md        the same, as markdown tables
    outputs/ensemble/seeds/accuracy.csv    one row per (combo, instance, epoch)

Run:

    python scripts/train_ensemble.py --instances 100 --epochs 10 --capture-epochs 0,1,10
    make train-ensemble
"""

import json
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
    parse_epoch_list,
    section,
    select_ensemble_combinations,
    setup_logging,
)
from susgrad.grads import compute_gradient_spectrum
from susgrad.persistence import (
    accuracy_summary,
    build_seed_manifest,
    instance_name,
    list_epochs,
    load_all_seed_manifests,
    save_gradients,
    save_seed_manifest,
    save_suspiciousness,
    seed_manifest_path,
    write_accuracy_csv,
    write_seeds_markdown,
)
from susgrad.sbfl import CORE_METRIC_NAMES, METRIC_NAMES, compute_suspiciousness
from susgrad.training import evaluate, train_epochs
from susgrad.utils import (
    ENSEMBLE_GRADIENTS_DIR,
    ENSEMBLE_SEEDS_DIR,
    ENSEMBLE_SUSPICIOUSNESS_DIR,
    describe_mapping,
    set_seed,
)


def _instance_complete(grad_base: Path, susp_base: Path, combo_key: str, i: int,
                       capture_epochs) -> bool:
    """True if every requested epoch is already captured (grad AND susp) for instance *i*."""
    inst = instance_name(i)
    want = set(capture_epochs)
    have_grad = set(list_epochs(grad_base, combo_key, inst))
    have_susp = set(list_epochs(susp_base, combo_key, inst))
    return want.issubset(have_grad) and want.issubset(have_susp)


def _stored_instance_results(seed_base: Path, combo_key: str) -> dict:
    """``{instance: entry}`` from an existing manifest, so a resumed run keeps
    the accuracies of instances it skips instead of blanking them."""
    path = seed_manifest_path(combo_key, seed_base)
    if not path.exists():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {e["instance"]: e for e in manifest.get("entries", [])}


@click.command()
@click.option("--instances", default=100, show_default=True,
              help="Number of freshly, randomly-initialised instances per combo.")
@click.option("--epochs", default=10, show_default=True, help="Epochs trained per instance.")
@click.option("--capture-epochs", default="0,1,10", show_default=True,
              help="Epoch snapshots to capture susp+grad at (0 = before any training). "
                   "Training runs straight through; only these epochs are measured.")
@click.option("--metrics", default=",".join(CORE_METRIC_NAMES), show_default=True,
              help="Suspiciousness metrics to compute and store.")
@click.option("--batch-size", default=64, show_default=True)
@click.option("--lr", default=1e-3, show_default=True)
@click.option("--seed", default=42, show_default=True,
              help="Base seed. Instance i is seeded with (seed + i), so instances "
                   "differ from each other but the whole run is reproducible.")
@click.option("--threshold", default=0.0, show_default=True, help="Suspiciousness activation threshold.")
@click.option("--dstar", "dstar_exponent", default=3, show_default=True, help="D* exponent.")
@click.option("--max-samples", default=None, type=int, help="Cap eval samples (quick runs).")
@click.option("--grad-dir", default=None, help="Base dir for gradient dumps.")
@click.option("--susp-dir", default=None, help="Base dir for suspiciousness dumps.")
@click.option("--seed-dir", default=None, help="Base dir for the per-instance seed manifests.")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True,
              help="Recompute instances that are already fully captured.")
@ensemble_combo_options
def main(instances, epochs, capture_epochs, metrics, batch_size, lr, seed, threshold,
          dstar_exponent, max_samples, grad_dir, susp_dir, seed_dir, overwrite, **flags):
    log, logfile = setup_logging("train_ensemble")
    capture = parse_epoch_list(capture_epochs, max_epoch=epochs)
    metric_names = [m.strip() for m in metrics.split(",") if m.strip()]
    unknown = set(metric_names) - set(METRIC_NAMES)
    if unknown:
        raise click.BadParameter(f"Unknown metric(s) {sorted(unknown)}; choose from {list(METRIC_NAMES)}.")
    combos = select_ensemble_combinations(collect_enabled_ensemble(flags))
    if not combos:
        log.warning("No combinations selected. Nothing to do.")
        return

    grad_base = Path(grad_dir) if grad_dir else ENSEMBLE_GRADIENTS_DIR
    susp_base = Path(susp_dir) if susp_dir else ENSEMBLE_SUSPICIOUSNESS_DIR
    seed_base = Path(seed_dir) if seed_dir else ENSEMBLE_SEEDS_DIR

    # Training stops at the last epoch anyone asked to measure -- epochs beyond
    # it would be pure waste (nothing is captured after them).
    train_to = max(capture)

    section(log, "Plan")
    field(log, "Instances per combo", instances)
    field(log, "Epochs per instance", train_to if train_to == epochs
          else f"{train_to} (--epochs {epochs}, but the last captured epoch is {train_to})")
    field(log, "Capture at epochs", ", ".join(str(e) for e in capture) + "  (0 = before any training)")
    field(log, "Capture schedule", "each instance trained ONCE, straight through; "
                                   "susp+grad computed in between, only at the epochs above")
    field(log, "Measured on", "the held-out evaluation split (test_loader), not the training data")
    field(log, "Seeding", f"instance i uses set_seed({seed} + i) -- reproducible, but distinct per instance")
    field(log, "Metrics", ", ".join(metric_names))
    field(log, "Accuracy", "held-out accuracy + loss recorded at every captured epoch")
    field(log, "Overwrite", overwrite)
    field(log, "Combinations", f"{len(combos)} selected: " + ", ".join(c.key for c in combos))

    manifests = []
    for combo in combos:
        section(log, combo.label)
        t_combo = time.perf_counter()
        n_trained, n_skipped = 0, 0
        seed_entries = []
        # Accuracies of instances this run skips are already on disk; keep them
        # rather than blanking the manifest on a resumed run.
        stored = _stored_instance_results(seed_base, combo.key)

        for i in range(instances):
            instance_seed = seed + i
            inst = instance_name(i)
            already_done = _instance_complete(grad_base, susp_base, combo.key, i, capture)

            if not overwrite and already_done:
                previous = stored.get(inst, {})
                seed_entries.append({
                    "index": i, "instance": inst, "seed": instance_seed,
                    "trained_this_run": False,
                    "accuracy": previous.get("accuracy", {}),
                    "loss": previous.get("loss", {}),
                })
                n_skipped += 1
                continue

            # Fresh, reproducible random draw for THIS instance's initial weights.
            set_seed(instance_seed)
            bundle, model = build_dataset_and_model(
                combo, batch_size=batch_size, seed=seed, max_samples=max_samples
            )
            if i == 0:
                field(log, "Model", model_summary(model))

            t0 = time.perf_counter()
            accuracy, loss = {}, {}

            def capture_at(epoch: int) -> dict:
                """Measure + store susp/grad (and accuracy) for this instance at *epoch*."""
                grads = compute_gradient_spectrum(model, bundle.test_loader)
                susp = compute_suspiciousness(
                    model, bundle.test_loader, metrics=metric_names,
                    threshold=threshold, dstar_exponent=dstar_exponent,
                )
                save_gradients(grads, combo.key, inst, epoch, base_dir=grad_base)
                save_suspiciousness(susp, combo.key, inst, epoch, base_dir=susp_base)
                # Same split, same moment in training as the tensors above.
                result = evaluate(model, bundle.test_loader)
                accuracy[str(epoch)] = result.accuracy
                loss[str(epoch)] = result.loss
                return grads

            # Epoch 0: BEFORE any training, i.e. the untrained snapshot.
            last_grad = capture_at(0) if 0 in capture else None

            for result in train_epochs(model, bundle.train_loader, n_epochs=train_to,
                                       lr=lr, progress=False):
                if result.epoch in capture:
                    last_grad = capture_at(result.epoch)

            seed_entries.append({
                "index": i, "instance": inst, "seed": instance_seed,
                "trained_this_run": True, "accuracy": accuracy, "loss": loss,
            })
            n_trained += 1
            log.info(
                "  - instance %s (%d/%d): trained %d epochs, captured at %s in %.2fs (%s) · acc %s",
                inst, i + 1, instances, train_to,
                ",".join(str(e) for e in capture), time.perf_counter() - t0,
                describe_mapping(last_grad) if last_grad else "n/a",
                " → ".join(f"e{e}: {accuracy[str(e)]:.2f}%" for e in capture if str(e) in accuracy),
            )

        field(log, "Trained this run", n_trained)
        field(log, "Skipped (already complete)", n_skipped)
        field(log, "Combo total time", f"{time.perf_counter() - t_combo:.1f}s")

        manifest = build_seed_manifest(
            combo_key=combo.key, label=combo.label, base_seed=seed, data_seed=seed,
            epochs_trained=train_to, capture_epochs=capture, seeds=seed_entries,
            metrics=metric_names,
        )
        manifests.append(manifest)
        for row in accuracy_summary(manifest):
            field(log, f"Accuracy @ epoch {row['epoch']}",
                  f"mean {row['mean']:.2f}% ± {row['std']:.2f} "
                  f"(min {row['min']:.2f}, max {row['max']:.2f}, n={row['n']})")
        long_field(log, "Seed manifest", save_seed_manifest(manifest, combo.key, base_dir=seed_base))

    section(log, "Done")
    long_field(log, "Gradients dir", grad_base)
    long_field(log, "Suspiciousness dir", susp_base)
    if manifests:
        # Rebuild the markdown/CSV from ALL manifests on disk, so a partial run
        # (e.g. --no-lenet-mnist) doesn't drop the combos it didn't touch.
        on_disk = load_all_seed_manifests(seed_base)
        long_field(log, "Seeds (markdown)", write_seeds_markdown(on_disk, base_dir=seed_base))
        long_field(log, "Accuracies (CSV)", write_accuracy_csv(on_disk, base_dir=seed_base))
    long_field(log, "Run log", logfile)


if __name__ == "__main__":
    main()
