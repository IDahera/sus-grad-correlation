#!/usr/bin/env python3
"""EXPERIMENT 2 -- step T2: follow ONE random neuron per model across training.

Builds on the dumps ``train_trajectory.py`` wrote: for each combo it picks one
neuron **at random from all of that model's neurons** (reproducible via
``--neuron-seed``, or pinned with ``--neuron mlp_mnist=net.1:57``) and plots how
that neuron's suspiciousness and gradient move over the captured epochs.

Suspiciousness metrics and gradients live on utterly different scales (ochiai is
0..1, D* runs into the thousands, mean |gradient| is ~1e-5), so each metric gets
its OWN panel with its own left axis, and the gradient is repeated as a dashed
line on each panel's right axis. That way every metric can be read against the
gradient without one curve flattening the others onto the baseline.

Outputs:

    outputs/trajectory/figures/<combo>__<layer>-n<idx>__trajectory.{png,pdf}
    outputs/trajectory/figures/all-combos__neuron-trajectories.{png,pdf}
    outputs/trajectory/neuron_trajectories.md   which neuron, and every value
    outputs/trajectory/neuron_trajectories.csv  the same values, one row per epoch

Run AFTER train_trajectory.py:

    python scripts/plot_trajectory.py --neuron-seed 42
    make plot-trajectory
"""

import csv
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
    section,
    select_ensemble_combinations,
    setup_logging,
)
from susgrad.neurons import (
    layer_shapes,
    neuron_series,
    neuron_value,
    parse_neuron_ref,
    pick_random_neuron,
)
from susgrad.persistence import (
    list_epochs,
    load_gradients,
    load_suspiciousness,
    seed_run_name,
)
from susgrad.sbfl import METRIC_NAMES
from susgrad.utils import (
    TRAJECTORY_DIR,
    TRAJECTORY_FIGURES_DIR,
    TRAJECTORY_GRADIENTS_DIR,
    TRAJECTORY_SUSPICIOUSNESS_DIR,
    ensure_dir,
)
from susgrad.viz.figures import (
    DEFAULT_FORMATS,
    draw_series_panel,
    new_grid_figure,
    save_figure,
    save_metric_panels,
    series_legend,
)
from susgrad.viz.textmap import markdown_table


def _parse_pins(values, log):
    """``--neuron mlp_mnist=net.1:57`` (repeatable) -> {combo_key: 'net.1:57'}."""
    pins = {}
    for item in values:
        if "=" not in item:
            raise click.BadParameter(f"--neuron expects <combo>=<layer>:<index>, got {item!r}")
        combo_key, ref = item.split("=", 1)
        pins[combo_key.strip()] = ref.strip()
    return pins


def _load_series(combo_key, variant, epochs, grad_base, susp_base, metrics, ref):
    """Per-epoch value of one neuron: every metric's suspiciousness + its gradient."""
    susp_per_epoch = {e: load_suspiciousness(combo_key, variant, e, base_dir=susp_base) for e in epochs}
    grad_per_epoch = {e: load_gradients(combo_key, variant, e, base_dir=grad_base) for e in epochs}

    _, gradient = neuron_series(grad_per_epoch, ref)
    susp = {}
    for metric in metrics:
        _, susp[metric] = neuron_series({e: susp_per_epoch[e][metric] for e in epochs}, ref)
    return susp, gradient


def _training_accuracy(out_base: Path, combo_key: str):
    """``{epoch: accuracy}`` from the CSV train_trajectory.py wrote (if present)."""
    path = out_base / f"{combo_key}__training.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {int(row["epoch"]): float(row["accuracy"]) for row in csv.DictReader(fh)}


def _is_inactive(gradient, susp) -> bool:
    """True if the neuron never fires: zero gradient AND zero suspiciousness throughout."""
    return (not any(g != 0.0 for g in gradient)
            and not any(v != 0.0 for values in susp.values() for v in values))


def _markdown_section(combo, ref, epochs, susp, gradient, accuracy, metrics, figure):
    head = [
        f"## {combo.label} (`{combo.key}`)",
        "",
        f"- **Neuron:** `{ref.describe()}`",
        f"- **Layer shape:** {tuple(ref.shape)} ({ref.n_neurons} neurons in that layer)",
        f"- **Flat index:** {ref.index} (row-major; coordinates {ref.coords})",
        f"- **Epochs:** {epochs[0]}..{epochs[-1]} ({len(epochs)} snapshots)",
        f"- **Figure:** `{figure.name}`",
    ]
    if _is_inactive(gradient, susp):
        head.append(
            "- **Note:** this neuron never activates on the evaluation split after "
            "training starts — a dead unit. Its curves are flat at zero by nature, "
            "not by error. Pass `--require-active` to draw only from neurons that "
            "still fire at the last captured epoch."
        )
    head.append("")
    # Escaped pipes: |gradient| inside a markdown table cell would end the column.
    header = ["epoch"] + [f"{m}" for m in metrics] + ["mean \\|gradient\\|"]
    if accuracy:
        header.append("model accuracy %")
    rows = []
    for i, epoch in enumerate(epochs):
        row = [epoch] + [f"{susp[m][i]:.4f}" for m in metrics] + [f"{gradient[i]:.3e}"]
        if accuracy:
            row.append(f"{accuracy.get(epoch, float('nan')):.2f}")
        rows.append(row)
    return "\n".join(head + [markdown_table(header, rows), ""])


@click.command()
@click.option("--seed", default=42, show_default=True,
              help="Which train_trajectory.py run to read (its --seed, i.e. the run folder).")
@click.option("--neuron-seed", default=42, show_default=True,
              help="Seed for the random neuron pick -- same seed, same neuron.")
@click.option("--neuron", "neuron_pins", multiple=True,
              help="Pin a neuron: --neuron mlp_mnist=net.1:57 (repeatable; coords also work: conv1:1,15,22).")
@click.option("--layer", default=None, help="Restrict the random pick to this layer name.")
@click.option("--require-active/--any-neuron", default=False, show_default=True,
              help="Re-draw until the neuron still fires at the last captured epoch. Biases the "
                   "pick (dead units are skipped), so it is opt-in.")
@click.option("--max-draws", default=50, show_default=True,
              help="How many re-draws --require-active may make before giving up.")
@click.option("--metrics", default=None, help="Comma-separated metrics (default: all stored).")
@click.option("--formats", default=",".join(DEFAULT_FORMATS), show_default=True,
              help="Image formats to write.")
@click.option("--dpi", default=300, show_default=True)
@click.option("--grad-dir", default=None, help="Base dir for trajectory gradient dumps.")
@click.option("--susp-dir", default=None, help="Base dir for trajectory suspiciousness dumps.")
@click.option("--out-dir", default=None, help="Base dir for the logs (and the figures subdir).")
@ensemble_combo_options
def main(seed, neuron_seed, neuron_pins, layer, require_active, max_draws, metrics,
         formats, dpi, grad_dir, susp_dir, out_dir, **flags):
    log, logfile = setup_logging("plot_trajectory")
    combos = select_ensemble_combinations(collect_enabled_ensemble(flags))
    if not combos:
        log.warning("No combinations selected. Nothing to do.")
        return

    formats = [f.strip().lower() for f in formats.split(",") if f.strip()]
    pins = _parse_pins(neuron_pins, log)
    grad_base = Path(grad_dir) if grad_dir else TRAJECTORY_GRADIENTS_DIR
    susp_base = Path(susp_dir) if susp_dir else TRAJECTORY_SUSPICIOUSNESS_DIR
    out_base = Path(out_dir) if out_dir else TRAJECTORY_DIR
    fig_base = ensure_dir(out_base / TRAJECTORY_FIGURES_DIR.name)
    variant = seed_run_name(seed)

    section(log, "Plan")
    field(log, "Run folder", f"{variant} (train_trajectory.py --seed {seed})")
    field(log, "Neuron pick", f"uniform over all neurons, rng seeded with {neuron_seed}"
          + (" (re-drawn until active)" if require_active else "")
                              + (f", restricted to layer {layer}" if layer else "")
          + (f"; pinned: {pins}" if pins else ""))
    field(log, "Combinations", f"{len(combos)} selected")

    sections, csv_rows, panels = [], [], []
    for combo in combos:
        section(log, combo.label)
        epochs = sorted(set(list_epochs(susp_base, combo.key, variant))
                        & set(list_epochs(grad_base, combo.key, variant)))
        if len(epochs) < 2:
            log.warning("  Need >=2 captured epochs (have %d) -- run train_trajectory.py first. Skipping.",
                        len(epochs))
            continue

        grad0 = load_gradients(combo.key, variant, epochs[0], base_dir=grad_base)
        susp0 = load_suspiciousness(combo.key, variant, epochs[0], base_dir=susp_base)
        shapes = layer_shapes(grad0)
        combo_metrics = [m for m in METRIC_NAMES if m in susp0]
        if metrics:
            wanted = {m.strip() for m in metrics.split(",") if m.strip()}
            combo_metrics = [m for m in combo_metrics if m in wanted]
        if not combo_metrics:
            log.warning("  No suspiciousness metrics stored. Skipping.")
            continue

        # Same neuron-seed => same neuron for this combo, run after run.
        if combo.key in pins:
            ref = parse_neuron_ref(pins[combo.key], shapes)
        else:
            rng = random.Random(neuron_seed)
            layer_filter = [layer] if layer else None
            ref = pick_random_neuron(shapes, rng, layers=layer_filter)
            if require_active:
                # A uniformly drawn neuron can be a dead unit whose curves are
                # flat at zero -- true, but not illustrative. Re-drawing until
                # one still fires at the LAST captured epoch biases the sample,
                # which is why it is opt-in and logged.
                last_grad = load_gradients(combo.key, variant, epochs[-1], base_dir=grad_base)
                attempts = 1
                while attempts <= max_draws and neuron_value(last_grad, ref) == 0.0:
                    ref = pick_random_neuron(shapes, rng, layers=layer_filter)
                    attempts += 1
                field(log, "Active-neuron draws", f"{attempts} (--require-active)"
                      + ("" if neuron_value(last_grad, ref) != 0.0
                         else f" — gave up after {max_draws}, keeping an inactive neuron"))

        susp, gradient = _load_series(combo.key, variant, epochs, grad_base, susp_base,
                                      combo_metrics, ref)
        accuracy = _training_accuracy(out_base, combo.key)

        field(log, "Chosen neuron", ref.describe())
        field(log, "Epochs", f"{epochs[0]}..{epochs[-1]} ({len(epochs)} snapshots)")
        field(log, "Gradient", f"epoch {epochs[0]}: {gradient[0]:.3e} → "
                               f"epoch {epochs[-1]}: {gradient[-1]:.3e}")
        for metric in combo_metrics:
            field(log, f"{metric}", f"epoch {epochs[0]}: {susp[metric][0]:.4f} → "
                                    f"epoch {epochs[-1]}: {susp[metric][-1]:.4f}")

        stem = fig_base / f"{combo.key}__{ref.slug()}__trajectory"
        written = save_metric_panels(
            stem, epochs, susp,
            right_series={"mean |gradient|": gradient},
            suptitle=f"{combo.label} — neuron {ref.layer}[{ref.index}] over {len(epochs)} snapshots",
            subtitle=f"{ref.describe()} · model seed {seed}",
            right_label="mean |gradient|", formats=formats, dpi=dpi,
        )
        long_field(log, "Figure", written[0])

        sections.append(_markdown_section(combo, ref, epochs, susp, gradient, accuracy,
                                          combo_metrics, written[0]))
        panels.append((combo, ref, epochs, susp, gradient, combo_metrics))
        for i, epoch in enumerate(epochs):
            row = {"combo": combo.key, "layer": ref.layer, "neuron_index": ref.index,
                   "coords": "-".join(str(c) for c in ref.coords), "epoch": epoch,
                   "gradient": f"{gradient[i]:.8e}",
                   "accuracy": f"{accuracy[epoch]:.4f}" if epoch in accuracy else ""}
            for metric in combo_metrics:
                row[metric] = f"{susp[metric][i]:.6f}"
            csv_rows.append(row)

    if not panels:
        log.warning("Nothing plotted (run train_trajectory.py first).")
        return

    # One overview figure: a row per model, a column per metric (the gradient is
    # repeated as the dashed reference in every panel).
    all_metrics = sorted({m for *_, metrics_here in panels for m in metrics_here},
                         key=lambda m: METRIC_NAMES.index(m))
    fig, axes = new_grid_figure(len(panels), len(all_metrics),
                                figsize=(4.2 * len(all_metrics), 3.0 * len(panels)))
    for row, (combo, ref, epochs, susp, gradient, metrics_here) in zip(axes, panels):
        for ax, metric in zip(row, all_metrics):
            if metric not in susp:
                ax.axis("off")
                continue
            draw_series_panel(
                ax, epochs, {metric: susp[metric]},
                right_series={"mean |gradient|": gradient},
                title=f"{combo.label} · {metric}\n{ref.layer}[{ref.index}]",
                left_label=metric, right_label="mean |gradient|",
            )
    overview = save_figure(fig, fig_base / "all-combos__neuron-trajectories", formats, dpi,
                           legend=True)

    md_path = out_base / "neuron_trajectories.md"
    md_path.write_text("\n".join([
        "# Neuron trajectories (experiment 2)",
        "",
        f"One randomly chosen neuron per model, followed across training. "
        f"Model seed **{seed}**, neuron-pick seed **{neuron_seed}** — re-running with the "
        "same two seeds reproduces exactly these neurons and values.",
        "",
        f"Overview figure: `{overview[0].name}`",
        "",
    ] + sections), encoding="utf-8")

    csv_path = out_base / "neuron_trajectories.csv"
    columns = ["combo", "layer", "neuron_index", "coords", "epoch"] + \
        [c for c in csv_rows[0] if c not in ("combo", "layer", "neuron_index", "coords", "epoch")]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    section(log, "Done")
    field(log, "Models plotted", len(panels))
    long_field(log, "Overview figure", overview[0])
    long_field(log, "Value log (markdown)", md_path)
    long_field(log, "Value log (CSV)", csv_path)
    long_field(log, "Run log", logfile)


if __name__ == "__main__":
    main()
