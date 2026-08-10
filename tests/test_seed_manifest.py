"""Tests for the per-instance seed manifests written by train_ensemble.py.

Each ensemble instance is a fresh random initialisation, so the manifest is the
only record of which seed produced which instance -- it has to round-trip, and
the markdown view has to list every instance (that is what makes a single
suspicious instance re-creatable months later).
"""

import pytest

from susgrad.persistence import (
    accuracy_by_epoch,
    accuracy_summary,
    build_seed_manifest,
    load_all_seed_manifests,
    load_seed_manifest,
    save_seed_manifest,
    seed_manifest_markdown,
    write_accuracy_csv,
    write_seeds_markdown,
)


def _manifest(combo_key="mlp_mnist", base_seed=42, n=3):
    return build_seed_manifest(
        combo_key=combo_key,
        label=f"label for {combo_key}",
        base_seed=base_seed,
        data_seed=base_seed,
        epochs_trained=10,
        capture_epochs=[0, 1, 10],
        metrics=["ochiai", "tarantula", "dstar"],
        seeds=[{"index": i, "instance": f"inst{i:03d}", "seed": base_seed + i,
                "trained_this_run": True,
                # Accuracy keys are strings, as they are after a JSON round-trip.
                "accuracy": {"0": 10.0 + i, "1": 80.0 + i, "10": 95.0 + i},
                "loss": {"0": 2.3, "1": 0.6, "10": 0.15}} for i in range(n)],
    )


def test_manifest_records_the_seed_of_every_instance():
    manifest = _manifest(n=20)
    assert manifest["instances"] == 20
    assert manifest["capture_epochs"] == [0, 1, 10]
    assert [e["seed"] for e in manifest["entries"]] == list(range(42, 62))
    assert manifest["entries"][7]["instance"] == "inst007"


def test_save_load_roundtrip(tmp_path):
    saved = save_seed_manifest(_manifest(), "mlp_mnist", base_dir=tmp_path)
    assert saved.name == "mlp_mnist.json"

    loaded = load_seed_manifest("mlp_mnist", base_dir=tmp_path)
    assert loaded["base_seed"] == 42
    assert loaded["entries"] == _manifest()["entries"]


def test_load_all_returns_every_combo(tmp_path):
    save_seed_manifest(_manifest("mlp_mnist"), "mlp_mnist", base_dir=tmp_path)
    save_seed_manifest(_manifest("lenet_mnist"), "lenet_mnist", base_dir=tmp_path)

    all_manifests = load_all_seed_manifests(tmp_path)
    assert [m["combo"] for m in all_manifests] == ["lenet_mnist", "mlp_mnist"]


def test_load_all_on_a_missing_dir_is_empty(tmp_path):
    assert load_all_seed_manifests(tmp_path / "nope") == []


def test_markdown_lists_every_instance_and_the_rule():
    text = seed_manifest_markdown(_manifest(n=4))
    assert "instance i is initialised with set_seed(base_seed + i)" in text
    for i in range(4):
        assert f"`inst{i:03d}`" in text and f"| {42 + i} |" in text


def test_write_seeds_markdown_covers_all_manifests(tmp_path):
    path = write_seeds_markdown([_manifest("mlp_mnist"), _manifest("lenet_mnist")],
                                base_dir=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert path.name == "seeds.md"
    assert "`mlp_mnist`" in text and "`lenet_mnist`" in text


# --- accuracy (recorded alongside the seed at every captured epoch) -------------

def test_manifest_records_accuracy_per_captured_epoch():
    manifest = _manifest(n=3)
    assert manifest["metrics"] == ["ochiai", "tarantula", "dstar"]
    assert accuracy_by_epoch(manifest) == {
        0: [10.0, 11.0, 12.0], 1: [80.0, 81.0, 82.0], 10: [95.0, 96.0, 97.0],
    }


def test_accuracy_summary_is_per_epoch_across_the_population():
    rows = {row["epoch"]: row for row in accuracy_summary(_manifest(n=3))}
    assert set(rows) == {0, 1, 10}
    assert rows[10]["mean"] == pytest.approx(96.0)
    assert rows[10]["n"] == 3
    assert rows[10]["min"] == 95.0 and rows[10]["max"] == 97.0
    assert rows[0]["std"] == pytest.approx(1.0)


def test_accuracy_summary_survives_instances_without_accuracy():
    manifest = _manifest(n=2)
    manifest["entries"][1]["accuracy"] = {}     # e.g. skipped before accuracy was recorded
    rows = {row["epoch"]: row for row in accuracy_summary(manifest)}
    assert rows[0]["n"] == 1 and rows[0]["std"] == 0.0


def test_accuracy_csv_has_a_row_per_instance_and_epoch(tmp_path):
    path = write_accuracy_csv([_manifest("mlp_mnist", n=2), _manifest("lenet_mnist", n=1)],
                              base_dir=tmp_path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()

    assert lines[0] == "combo,instance,seed,epoch,accuracy,loss"
    assert len(lines) == 1 + (2 + 1) * 3          # 3 instances x 3 epochs
    assert "mlp_mnist,inst000,42,0,10.0000,2.300000" in lines
    assert any(line.startswith("lenet_mnist,inst000,42,10,") for line in lines)


def test_markdown_shows_the_accuracy_columns_and_summary():
    text = seed_manifest_markdown(_manifest(n=2))
    assert "Held-out accuracy across the population" in text
    assert "acc @ e0" in text and "acc @ e10" in text
    assert "| 95.00 |" in text
