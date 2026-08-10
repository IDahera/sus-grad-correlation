"""Tests for the main (ensemble) pipeline's instance/epoch-snapshot naming.

Mirrors tests/test_variants.py's coverage of variant_name/variant_stop, but for
the ensemble pipeline's two naming schemes: an *instance* id (one randomly
initialised model, used as the ``variant`` slot for gradients/suspiciousness)
and an *epoch snapshot* id (used as the ``variant`` slot for correlation, since
the ensemble correlates across instances separately per epoch).
"""

import pytest
import torch

from susgrad.persistence import (
    epoch_snapshot_index,
    epoch_snapshot_name,
    instance_index,
    instance_name,
    list_instances,
    save_gradients,
    seed_run_name,
    seed_run_seed,
)


def test_instance_name_roundtrip():
    assert instance_name(0) == "inst000"
    assert instance_name(37) == "inst037"
    assert instance_index("inst000") == 0
    assert instance_index(instance_name(49)) == 49


def test_epoch_snapshot_name_roundtrip():
    assert epoch_snapshot_name(0) == "epoch00"
    assert epoch_snapshot_name(5) == "epoch05"
    assert epoch_snapshot_index("epoch00") == 0
    assert epoch_snapshot_index(epoch_snapshot_name(12)) == 12


def test_list_instances(tmp_path):
    g = {"fc": torch.zeros(4)}
    for i in (0, 1, 5):
        save_gradients(g, "combo", instance_name(i), epoch=0, base_dir=tmp_path)

    assert list_instances(tmp_path, "combo") == ["inst000", "inst001", "inst005"]
    # Instance-shaped folders must not be picked up by the (unrelated) variant lister.
    from susgrad.persistence import list_variants
    assert list_variants(tmp_path, "combo") == []


# --- experiment 2: the seeded single-instance run folder -------------------------

def test_seed_run_name_roundtrip():
    assert seed_run_name(42) == "seed042"
    assert seed_run_name(7) == "seed007"
    assert seed_run_seed("seed042") == 42
    assert seed_run_seed(seed_run_name(123)) == 123


def test_seed_run_name_rejects_other_variant_names():
    for bad in ("inst000", "e050", "epoch00", "nope"):
        with pytest.raises(ValueError):
            seed_run_seed(bad)


def test_different_seeds_never_share_a_run_folder(tmp_path):
    # Two runs of train_trajectory.py with different --seed must not overwrite
    # each other: the seed IS the variant slot.
    for seed in (42, 7):
        save_gradients({"fc": torch.tensor([float(seed)])}, "mlp_mnist",
                       seed_run_name(seed), 0, base_dir=tmp_path)
    assert sorted(p.name for p in (tmp_path / "mlp_mnist").iterdir()) == ["seed007", "seed042"]
