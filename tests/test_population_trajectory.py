"""Tests for the whole-model (population) trajectory step.

Two joints are worth pinning down. First, the ``layer = all`` rows
``value_stats.py`` writes: they pool every layer's NEURONS, which is what makes
"the mean over all trainable neurons" a defensible sentence in the write-up --
and they must not disturb the per-layer rows that were already there. Second,
the reader in ``plot_trajectory_population.py``, which turns those rows back
into curves; if it silently dropped or reordered epochs, the figure would still
look plausible.
"""

import csv

import pytest
import torch

from scripts.plot_trajectory_population import (
    fit_case,
    load_stats_rows,
    loss_series,
    metrics_in,
    series_for,
    series_pearson,
)
from scripts.value_stats import _pooled_rows, _rows_for_epoch
from susgrad.stats import ALL_LAYERS


@pytest.fixture
def dumps():
    """One 'instance': two layers of 4 and 2 neurons, one metric plus gradients."""
    susp = [{"ochiai": {"net.1": torch.tensor([0.0, 0.2, 0.4, 0.6]),
                        "net.3": torch.tensor([1.0, 1.0])}}]
    grad = [{"net.1": torch.tensor([0.0, 1.0, 2.0, 3.0]),
             "net.3": torch.tensor([4.0, 6.0])}]
    return susp, grad


def _row(rows, layer, kind):
    return next(r for r in rows if r["layer"] == layer and r["kind"] == kind)


def test_all_layers_row_pools_every_neuron(dumps):
    susp, grad = dumps
    rows = _rows_for_epoch("trajectory", "mlp_mnist", 3, susp, grad, ["ochiai"], ["net.1", "net.3"])

    pooled = _row(rows, ALL_LAYERS, "ochiai")
    assert pooled["n"] == "6"                          # 4 + 2 neurons, not 2 layers
    # Mean over NEURONS: (0 + .2 + .4 + .6 + 1 + 1) / 6, not the mean of the
    # two per-layer means -- the big layer has to weigh more.
    assert float(pooled["mean"]) == pytest.approx(3.2 / 6, rel=1e-5)
    assert float(pooled["max"]) == pytest.approx(1.0)
    assert float(pooled["frac_zero"]) == pytest.approx(1 / 6, rel=1e-5)


def test_all_layers_row_is_added_not_substituted(dumps):
    susp, grad = dumps
    rows = _rows_for_epoch("trajectory", "mlp_mnist", 3, susp, grad, ["ochiai"], ["net.1", "net.3"])

    layers = {r["layer"] for r in rows}
    assert layers == {"net.1", "net.3", ALL_LAYERS}
    assert float(_row(rows, "net.1", "ochiai")["mean"]) == pytest.approx(0.3)
    assert float(_row(rows, "net.3", "ochiai")["mean"]) == pytest.approx(1.0)
    # The gradient gets the same treatment as every metric.
    assert float(_row(rows, ALL_LAYERS, "gradient")["mean"]) == pytest.approx(16 / 6, rel=1e-5)


def test_epoch_all_rows_also_carry_the_pooled_layer(dumps):
    susp, grad = dumps
    rows = _pooled_rows("trajectory", "mlp_mnist", {0: susp, 1: susp}, {0: grad, 1: grad},
                        ["ochiai"], ["net.1", "net.3"], 1)

    pooled = _row(rows, ALL_LAYERS, "ochiai")
    assert pooled["epoch"] == "all"
    assert pooled["n"] == "12"                          # both epochs, all neurons
    assert float(pooled["mean"]) == pytest.approx(3.2 / 6, rel=1e-5)


# --- reading the table back --------------------------------------------------------

STATS_HEADER = ["experiment", "combo", "layer", "kind", "epoch", "instances", "n",
                "min", "p05", "median", "mean", "p95", "max", "std", "frac_zero"]


def _write_stats_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=STATS_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _stats_row(**overrides):
    row = {f: "0" for f in STATS_HEADER}
    row.update({"experiment": "trajectory", "combo": "mlp_mnist", "layer": ALL_LAYERS,
                "kind": "ochiai", "instances": "1", "n": "6"})
    row.update(overrides)
    return row


def test_load_drops_the_pooled_epoch_and_other_experiments(tmp_path):
    path = _write_stats_csv(tmp_path / "value_stats.csv", [
        _stats_row(epoch="0", mean="0.9"),
        _stats_row(epoch="1", mean="0.5"),
        _stats_row(epoch="all", mean="0.7"),          # a summary, not a point in time
        _stats_row(epoch="0", mean="0.1", experiment="ensemble"),
    ])

    rows = load_stats_rows(path)

    assert [r["epoch"] for r in rows] == [0, 1]
    assert all(isinstance(r["epoch"], int) for r in rows)


def test_series_is_epoch_ordered_regardless_of_file_order(tmp_path):
    path = _write_stats_csv(tmp_path / "value_stats.csv", [
        _stats_row(epoch="2", mean="0.3", p05="0.1", p95="0.5"),
        _stats_row(epoch="0", mean="0.9", p05="0.7", p95="1.0"),
        _stats_row(epoch="1", mean="0.5", p05="0.3", p95="0.8"),
    ])

    epochs, values, band = series_for(load_stats_rows(path), combo="mlp_mnist",
                                      layer=ALL_LAYERS, kind="ochiai", stat="mean")

    assert epochs == [0, 1, 2]
    assert values == [0.9, 0.5, 0.3]
    assert band == ([0.7, 0.3, 0.1], [1.0, 0.8, 0.5])


def test_missing_combination_is_reported_not_raised(tmp_path):
    path = _write_stats_csv(tmp_path / "value_stats.csv", [_stats_row(epoch="0")])

    assert series_for(load_stats_rows(path), combo="lenet_mnist", layer=ALL_LAYERS,
                      kind="ochiai", stat="mean") == (None, None, None)


def test_metrics_are_listed_in_the_canonical_order(tmp_path):
    path = _write_stats_csv(tmp_path / "value_stats.csv", [
        _stats_row(epoch="0", kind="dstar"),
        _stats_row(epoch="0", kind="ochiai"),
        _stats_row(epoch="0", kind="gradient"),        # not a suspiciousness metric
    ])

    assert metrics_in(load_stats_rows(path), "mlp_mnist", ALL_LAYERS) == ["ochiai", "dstar"]


# --- fitting through the script's helper ---------------------------------------------

def test_fit_case_leaves_a_gap_before_the_first_fitted_epoch():
    epochs = list(range(11))
    values = [9.9] + [0.5 + 0.5 * (0.5 ** e) for e in range(1, 11)]   # epoch 0 is an outlier

    fit, curve, summary = fit_case(epochs, values, model="exp", fit_from=1, tol=0.05, tail=0.25)

    assert len(curve) == len(epochs)
    assert curve[0] != curve[0]                       # NaN: epoch 0 was not fitted
    assert all(v == v for v in curve[1:])
    # The outlier at epoch 0 must not drag the asymptote up.
    assert fit.asymptote == pytest.approx(0.5, abs=0.05)
    assert summary["first_value"] == pytest.approx(values[1])


def test_fit_case_declines_when_asked_for_no_fit_or_given_too_little():
    epochs, values = [0, 1, 2, 3], [1.0, 0.9, 0.8, 0.7]

    assert fit_case(epochs, values, model="none", fit_from=1, tol=0.05, tail=0.25) == (None, None, None)
    assert fit_case(epochs, values, model="exp", fit_from=3, tol=0.05, tail=0.25) == (None, None, None)


# --- the loss overlay ----------------------------------------------------------------

def _write_training_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "accuracy", "test_loss",
                                                "train_loss", "seconds"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_loss_series_aligns_to_the_plotted_epochs(tmp_path):
    # The training CSV is a separate file from value_stats.csv, so the two can
    # cover different epochs; the loss must follow the PLOTTED epochs or a curve
    # would be drawn against the wrong x values.
    path = _write_training_csv(tmp_path / "mlp_mnist__training.csv", [
        {"epoch": 0, "accuracy": 9.8, "test_loss": 2.3, "train_loss": "", "seconds": 0},
        {"epoch": 1, "accuracy": 96.0, "test_loss": 0.13, "train_loss": 0.27, "seconds": 4},
        {"epoch": 2, "accuracy": 97.0, "test_loss": 0.09, "train_loss": 0.11, "seconds": 4},
    ])

    series = loss_series(path, [0, 1, 2], ["test", "train"])

    assert series["test loss"] == [2.3, 0.13, 0.09]
    # Epoch 0 has no training loss -- a gap, not a zero.
    assert series["training loss"][0] != series["training loss"][0]
    assert series["training loss"][1:] == [0.27, 0.11]


def test_loss_series_leaves_a_gap_for_an_epoch_the_csv_lacks(tmp_path):
    path = _write_training_csv(tmp_path / "t.csv", [
        {"epoch": 0, "accuracy": 9.8, "test_loss": 2.3, "train_loss": "", "seconds": 0},
    ])

    values = loss_series(path, [0, 1], ["test"])["test loss"]

    assert values[0] == 2.3 and values[1] != values[1]


def test_missing_training_csv_is_reported_not_raised(tmp_path):
    assert loss_series(tmp_path / "absent.csv", [0, 1], ["test"]) == {}


def test_series_pearson_drops_the_untrained_epoch(tmp_path):
    # Epoch 0 is an outlier in BOTH series (chance-level loss, un-grown
    # gradient). Left in, that single point sets the coefficient; the default
    # fit_from drops it, exactly as the curve fits do.
    epochs = list(range(6))
    loss = [2.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    gradient = [1.2, 1.0, 2.0, 3.0, 4.0, 5.0]   # chance-level loss, un-grown gradient

    assert series_pearson(epochs, loss, gradient, from_epoch=1) == pytest.approx(1.0, abs=1e-6)
    assert series_pearson(epochs, loss, gradient, from_epoch=0) < 0.5


def test_series_pearson_ignores_epochs_either_series_is_missing():
    epochs = [0, 1, 2, 3]
    loss = [float("nan"), 0.4, 0.5, 0.6]
    gradient = [1.0, 1.0, 2.0, 3.0]

    assert series_pearson(epochs, loss, gradient) == pytest.approx(1.0, abs=1e-6)
