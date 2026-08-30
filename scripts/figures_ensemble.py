#!/usr/bin/env python3
"""MAIN (ensemble) PIPELINE -- step 4: export the heatmaps as standalone image files.

The HTML report is for exploring; this writes the same heatmaps as PNG **and**
PDF files you can ``\\includegraphics`` straight into a LaTeX document, in a
light print style with every figure self-labelled (epoch, value range, layer,
metric, method).

Three kinds of figure, each in two shapes:

    correlation   the across-instance corr(metric, gradient) -- the result
    suspiciousness \\ for ONE randomly chosen (and logged) instance, as an
    gradient        / illustrative sample of what goes into that correlation

    single panel  one epoch per file           -> best for the thesis: LaTeX
                                                  controls layout/captions
    epoch row     the captured epochs in one   -> convenient for drafts/slides;
                  file, SHARED colour scale       labels are baked in

Filenames sort into exactly the order you go looking for things:

    <combo>__<layer>[__ch<NN>]__<kind>[__<metric>][__<method>][__<inst>]__epoch(s)-...

Also written: ``manifest.csv`` (every figure, searchable) and ``figures.tex``
(ready-made ``\\susgradfig`` / ``\\susgradepochs`` macros plus a commented index
of every stem).

Run AFTER train_ensemble.py + correlate_ensemble.py:

    python scripts/figures_ensemble.py --epochs 0,1,10
    make figures-ensemble
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

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
from susgrad.correlation import CORRELATIONS
from susgrad.persistence import (
    correlation_path,
    epoch_snapshot_name,
    list_epochs,
    list_instances,
    load_correlation,
    load_gradients,
    load_suspiciousness,
)
from susgrad.sbfl import METRIC_NAMES
from susgrad.utils import (
    ENSEMBLE_CORRELATION_DIR,
    ENSEMBLE_FIGURES_DIR,
    ENSEMBLE_GRADIENTS_DIR,
    ENSEMBLE_SUSPICIOUSNESS_DIR,
    ensure_dir,
)
from susgrad.viz import build_heatmap
from susgrad.viz.figures import (
    figure_stem,
    merge_manifest,
    range_label,
    save_heatmap,
    save_heatmap_row,
    write_latex_index,
    write_manifest_csv,
)
from susgrad.viz.textmap import correlation_summary


def _epoch_label(epoch: int) -> str:
    return "epoch 0 (untrained)" if epoch == 0 else f"epoch {epoch}"


def _n_channels(tensor) -> int:
    return int(tensor.shape[0]) if tensor.dim() >= 3 else 1


def _record(path, *, combo, kind, layer, channel, epochs, panels,
            metric="", method="", instance=""):
    return {
        "combo": combo, "layer": layer, "channel": "" if channel is None else channel,
        "kind": kind, "metric": metric, "method": method, "instance": instance,
        "epochs": "+".join(str(e) for e in epochs), "panels": panels, "file": str(path),
    }


def _emit(stem_dir, *, base_kwargs, panels_by_epoch, epochs, title, subtitle_fn,
          suptitle, cbar_label, xlabel, ylabel, diverging, formats, dpi,
          singles, rows, records, record_kwargs, log):
    """Write the epoch-row file and/or one file per epoch for a single heatmap set."""
    written = 0
    if rows and len(epochs) > 1:
        stem = stem_dir / figure_stem(epochs=epochs, **base_kwargs)
        paths = save_heatmap_row(
            stem,
            [(_epoch_label(e), panels_by_epoch[e]) for e in epochs],
            suptitle=suptitle, diverging=diverging, cbar_label=cbar_label,
            xlabel=xlabel, ylabel=ylabel, formats=formats, dpi=dpi,
        )
        records.append(_record(paths[0], epochs=epochs, panels=len(epochs), **record_kwargs))
        written += 1

    if singles:
        for epoch in epochs:
            stem = stem_dir / figure_stem(epoch=epoch, **base_kwargs)
            paths = save_heatmap(
                stem, panels_by_epoch[epoch],
                title=title, subtitle=subtitle_fn(epoch), diverging=diverging,
                cbar_label=cbar_label, xlabel=xlabel, ylabel=ylabel,
                formats=formats, dpi=dpi,
            )
            records.append(_record(paths[0], epochs=[epoch], panels=1, **record_kwargs))
            written += 1
    return written


@click.command()
@click.option("--epochs", default="0,1,10", show_default=True,
              help="Epoch snapshots to export. Use 'auto' for everything available.")
@click.option("--instance", default=None,
              help="Instance for the suspiciousness/gradient samples (default: random, logged).")
@click.option("--instance-seed", default=None, type=int, help="Seed for that random pick.")
@click.option("--metrics", default=None, help="Comma-separated metrics (default: all stored).")
@click.option("--methods", default=None, help="Comma-separated correlation methods (default: all stored).")
@click.option("--layers", default=None,
              help="Comma-separated layer names to export (default: all). Names that a given "
                   "model does not have are simply skipped, so one list can serve several models, "
                   "e.g. --layers net.1,conv1,fc1.")
@click.option("--flat/--nested", default=False, show_default=True,
              help="Write every figure straight into --out-dir instead of <combo>/<kind>/ "
                   "subfolders. Handy for a small, curated set to \\includegraphics from.")
@click.option("--channels", default=1, show_default=True,
              help="Conv layers: how many channels to export per layer.")
@click.option("--formats", default="png,pdf", show_default=True,
              help="Image formats to write (png for previews, pdf for LaTeX).")
@click.option("--singles/--no-singles", default=True, show_default=True,
              help="One file per epoch (the recommended route for a LaTeX document).")
@click.option("--rows/--no-rows", default=True, show_default=True,
              help="The captured epochs side by side in one file, shared colour scale.")
@click.option("--samples/--no-samples", default=True, show_default=True,
              help="Also export suspiciousness/gradient maps for ONE sample instance.")
@click.option("--dpi", default=300, show_default=True, help="Raster resolution.")
@click.option("--grad-dir", default=None, help="Base dir for ensemble gradient dumps.")
@click.option("--susp-dir", default=None, help="Base dir for ensemble suspiciousness dumps.")
@click.option("--corr-dir", default=None, help="Base dir for ensemble correlation dumps.")
@click.option("--out-dir", default=None, help="Where to write the figures.")
@ensemble_combo_options
def main(epochs, instance, instance_seed, metrics, methods, layers, flat, channels, formats,
         singles, rows, samples, dpi, grad_dir, susp_dir, corr_dir, out_dir,
         **flags):
    log, logfile = setup_logging("figures_ensemble")
    wanted_epochs = parse_epoch_list(epochs, allow_auto=True)
    formats = [f.strip().lower() for f in formats.split(",") if f.strip()]
    layer_filter = {l.strip() for l in layers.split(",") if l.strip()} if layers else None
    combos = select_ensemble_combinations(collect_enabled_ensemble(flags))
    if not combos:
        log.warning("No combinations selected. Nothing to do.")
        return
    if not singles and not rows:
        raise click.BadParameter("Nothing to write: --no-singles and --no-rows together.")

    grad_base = Path(grad_dir) if grad_dir else ENSEMBLE_GRADIENTS_DIR
    susp_base = Path(susp_dir) if susp_dir else ENSEMBLE_SUSPICIOUSNESS_DIR
    corr_base = Path(corr_dir) if corr_dir else ENSEMBLE_CORRELATION_DIR
    out_base = ensure_dir(Path(out_dir) if out_dir
                          else ENSEMBLE_FIGURES_DIR)
    rng = random.Random(instance_seed)

    section(log, "Plan")
    field(log, "Combinations", f"{len(combos)} selected")
    field(log, "Epoch snapshots", "auto" if wanted_epochs is None
          else ", ".join(str(e) for e in wanted_epochs))
    field(log, "Shapes", ", ".join(x for x, on in (("single panel per epoch", singles),
                                                   ("epoch row (shared scale)", rows)) if on))
    field(log, "Formats", ", ".join(formats) + f" @ {dpi} dpi")
    field(log, "Layers", "all" if layer_filter is None else ", ".join(sorted(layer_filter)))
    field(log, "Layout", "flat (one directory)" if flat else "<combo>/<kind>/ subfolders")
    field(log, "Sample instance maps", samples)
    long_field(log, "Output dir", out_base)

    records, t_start = [], time.perf_counter()
    for combo in combos:
        section(log, combo.label)
        available = sorted(set(list_instances(susp_base, combo.key)) & set(list_instances(grad_base, combo.key)))
        if not available:
            log.warning("  No captured instances -- run train_ensemble.py first. Skipping.")
            continue

        have = {
            inst: set(list_epochs(susp_base, combo.key, inst)) & set(list_epochs(grad_base, combo.key, inst))
            for inst in available
        }
        eligible = ([i for i in available if set(wanted_epochs).issubset(have[i])]
                    if wanted_epochs else [i for i in available if have[i]])
        if not eligible:
            log.warning("  No instance carries the requested epochs. Skipping.")
            continue

        chosen = instance if instance in eligible else rng.choice(eligible)
        combo_epochs = sorted(have[chosen] & set(wanted_epochs)) if wanted_epochs else sorted(have[chosen])
        combo_epochs = [e for e in combo_epochs
                        if correlation_path(combo.key, epoch_snapshot_name(e), corr_base).exists()]
        if not combo_epochs:
            log.warning("  No correlated epochs -- run correlate_ensemble.py first. Skipping.")
            continue

        corr_by_epoch = {e: load_correlation(combo.key, epoch_snapshot_name(e), base_dir=corr_base)
                         for e in combo_epochs}
        susp_by_epoch = {e: load_suspiciousness(combo.key, chosen, e, base_dir=susp_base)
                         for e in combo_epochs}
        grad_by_epoch = {e: load_gradients(combo.key, chosen, e, base_dir=grad_base)
                         for e in combo_epochs}

        first_corr = corr_by_epoch[combo_epochs[0]]
        combo_metrics = [m for m in METRIC_NAMES if all(m in corr_by_epoch[e] for e in combo_epochs)]
        if metrics:
            wanted = {m.strip() for m in metrics.split(",") if m.strip()}
            combo_metrics = [m for m in combo_metrics if m in wanted]
        combo_methods = [m for m in CORRELATIONS
                         if all(m in corr_by_epoch[e].get(combo_metrics[0], {}) for e in combo_epochs)] \
            if combo_metrics else []
        if methods:
            wanted = {m.strip() for m in methods.split(",") if m.strip()}
            combo_methods = [m for m in combo_methods if m in wanted]
        if not combo_metrics or not combo_methods:
            log.warning("  Nothing to export (metrics=%s, methods=%s).", combo_metrics, combo_methods)
            continue

        layers = list(grad_by_epoch[combo_epochs[0]])
        if layer_filter is not None:
            layers = [l for l in layers if l in layer_filter]
            if not layers:
                log.warning("  None of the requested layers exist here (has: %s). Skipping.",
                            ", ".join(grad_by_epoch[combo_epochs[0]]))
                continue
        n_instances = len(eligible)

        field(log, "Sample instance", f"{chosen} (of {n_instances} eligible)")
        field(log, "Epochs", ", ".join(str(e) for e in combo_epochs))
        field(log, "Metrics × methods", f"{', '.join(combo_metrics)} × {', '.join(combo_methods)}")
        field(log, "Layers", ", ".join(layers))

        n_written = 0
        for layer in layers:
            n_c = _n_channels(grad_by_epoch[combo_epochs[0]][layer])
            for channel in range(min(n_c, channels)):
                ch_arg = channel if n_c > 1 else None
                sample_hm = build_heatmap(layer, grad_by_epoch[combo_epochs[0]][layer], channel=channel)
                xlabel, ylabel, info = sample_hm.xlabel, sample_hm.ylabel, sample_hm.info

                # --- correlation (the result: across ALL instances) ---
                for metric in combo_metrics:
                    for method in combo_methods:
                        grids, stats = {}, {}
                        for epoch in combo_epochs:
                            tensor = corr_by_epoch[epoch][metric][method][layer]
                            grids[epoch] = build_heatmap(layer, tensor, channel=channel).grid
                            stats[epoch] = correlation_summary(tensor.detach().cpu().numpy())
                        n_written += _emit(
                            out_base if flat else out_base / combo.key / "correlation",
                            base_kwargs=dict(combo=combo.key, kind="corr", layer=layer,
                                             channel=ch_arg, metric=metric, method=method),
                            panels_by_epoch=grids, epochs=combo_epochs,
                            title=f"{combo.label} · {layer} · {info}",
                            subtitle_fn=lambda e, m=metric, me=method, s=stats: (
                                f"{me}({m}, gradient) across {n_instances} instances · "
                                f"{_epoch_label(e)} · mean |r| = {s[e]['mean_abs']:.3f}"),
                            suptitle=(f"{combo.label} · {layer} · {info}\n"
                                      f"{method}({metric}, gradient) across {n_instances} instances"),
                            cbar_label="correlation r", xlabel=xlabel, ylabel=ylabel,
                            diverging=True, formats=formats, dpi=dpi,
                            singles=singles, rows=rows, records=records,
                            record_kwargs=dict(combo=combo.key, kind="corr", layer=layer,
                                               channel=ch_arg, metric=metric, method=method),
                            log=log,
                        )

                if not samples:
                    continue

                # --- suspiciousness + gradient for ONE sample instance ---
                for metric in combo_metrics:
                    grids = {e: build_heatmap(layer, susp_by_epoch[e][metric][layer], channel=channel).grid
                             for e in combo_epochs}
                    n_written += _emit(
                        out_base if flat else out_base / combo.key / "suspiciousness",
                        base_kwargs=dict(combo=combo.key, kind="susp", layer=layer,
                                         channel=ch_arg, metric=metric, instance=chosen),
                        panels_by_epoch=grids, epochs=combo_epochs,
                        title=f"{combo.label} · {layer} · {info}",
                        subtitle_fn=lambda e, m=metric, g=grids: (
                            f"{m} suspiciousness · instance {chosen} · {_epoch_label(e)} · "
                            f"range {range_label(g[e])}"),
                        suptitle=(f"{combo.label} · {layer} · {info}\n"
                                  f"{metric} suspiciousness · sample instance {chosen}"),
                        cbar_label=f"{metric} suspiciousness", xlabel=xlabel, ylabel=ylabel,
                        diverging=False, formats=formats, dpi=dpi,
                        singles=singles, rows=rows, records=records,
                        record_kwargs=dict(combo=combo.key, kind="susp", layer=layer,
                                           channel=ch_arg, metric=metric, instance=chosen),
                        log=log,
                    )

                grids = {e: build_heatmap(layer, grad_by_epoch[e][layer], channel=channel).grid
                         for e in combo_epochs}
                n_written += _emit(
                    out_base if flat else out_base / combo.key / "gradient",
                    base_kwargs=dict(combo=combo.key, kind="grad", layer=layer,
                                     channel=ch_arg, instance=chosen),
                    panels_by_epoch=grids, epochs=combo_epochs,
                    title=f"{combo.label} · {layer} · {info}",
                    subtitle_fn=lambda e, g=grids: (
                        f"mean |gradient| · instance {chosen} · {_epoch_label(e)} · "
                        f"range {range_label(g[e])}"),
                    suptitle=(f"{combo.label} · {layer} · {info}\n"
                              f"mean |gradient| · sample instance {chosen}"),
                    cbar_label="mean |gradient|", xlabel=xlabel, ylabel=ylabel,
                    diverging=False, formats=formats, dpi=dpi,
                    singles=singles, rows=rows, records=records,
                    record_kwargs=dict(combo=combo.key, kind="grad", layer=layer,
                                       channel=ch_arg, instance=chosen),
                    log=log,
                )

        field(log, "Figures written", f"{n_written} ({n_written * len(formats)} files)")

    if not records:
        log.warning("Nothing exported (run train_ensemble.py + correlate_ensemble.py first).")
        return

    section(log, "Done")
    field(log, "Figures", f"{len(records)} ({len(records) * len(formats)} files) "
                          f"in {time.perf_counter() - t_start:.1f}s")
    # Merge, never replace: a narrow re-run (one layer, one epoch) must not wipe
    # the index of everything exported before it.
    all_records = merge_manifest(out_base / "manifest.csv", records)
    if len(all_records) > len(records):
        field(log, "Manifest", f"{len(records)} written this run, {len(all_records)} in the directory")
    long_field(log, "Manifest", write_manifest_csv(out_base / "manifest.csv", all_records))
    long_field(log, "LaTeX index", write_latex_index(out_base / "figures.tex", all_records, root=out_base))
    long_field(log, "Run log", logfile)


if __name__ == "__main__":
    main()
