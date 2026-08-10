"""Grouped store/load functions for every artefact this project produces.

    models   -- whole trained/untrained models (save_model / load_model)
    spectra  -- per-epoch gradient and suspiciousness tensors
                (save_gradients / load_gradients / save_suspiciousness /
                 load_suspiciousness)
    seeds    -- per-instance manifests for the ensemble pipeline: which seed
                produced which instance, and its accuracy at each captured epoch
                (build_seed_manifest / save_seed_manifest / load_seed_manifest)
"""

from susgrad.persistence.models import load_model, save_model
from susgrad.persistence.seeds import (
    accuracy_by_epoch,
    accuracy_summary,
    build_seed_manifest,
    load_all_seed_manifests,
    load_seed_manifest,
    save_seed_manifest,
    seed_manifest_markdown,
    seed_manifest_path,
    write_accuracy_csv,
    write_seeds_markdown,
)
from susgrad.persistence.spectra import (
    correlation_path,
    epoch_snapshot_index,
    epoch_snapshot_name,
    gradients_path,
    instance_index,
    instance_name,
    list_epochs,
    list_instances,
    list_variants,
    load_correlation,
    load_gradients,
    load_suspiciousness,
    save_correlation,
    save_gradients,
    save_suspiciousness,
    seed_run_name,
    seed_run_seed,
    suspiciousness_path,
    variant_complete,
    variant_name,
    variant_stop,
)

__all__ = [
    "load_model",
    "save_model",
    "save_gradients",
    "load_gradients",
    "save_suspiciousness",
    "load_suspiciousness",
    "save_correlation",
    "load_correlation",
    "gradients_path",
    "suspiciousness_path",
    "correlation_path",
    "list_epochs",
    "list_variants",
    "variant_complete",
    "variant_name",
    "variant_stop",
    "instance_name",
    "instance_index",
    "list_instances",
    "epoch_snapshot_name",
    "epoch_snapshot_index",
    "seed_run_name",
    "seed_run_seed",
    "build_seed_manifest",
    "save_seed_manifest",
    "load_seed_manifest",
    "load_all_seed_manifests",
    "seed_manifest_path",
    "seed_manifest_markdown",
    "write_seeds_markdown",
    "write_accuracy_csv",
    "accuracy_by_epoch",
    "accuracy_summary",
]
