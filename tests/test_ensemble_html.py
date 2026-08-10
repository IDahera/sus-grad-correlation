"""Tests for the ensemble report's side-by-side epoch layout.

The report's whole point is that the captured epochs (base case 0, 1, 10) sit in
ONE ROW per quantity, on a shared colour scale -- not behind an epoch dropdown.
These tests pin that contract at the level the browser actually consumes: the
embedded JSON payload and the runtime that turns it into rows.
"""

import json
import re

from susgrad.viz.ensemble_html import build_ensemble_report


def _payload(epochs=(0, 1, 10)):
    layer, metric, method = "fc1", "ochiai", "pearson"
    grids = {e: [[[0.1, 0.2], [0.3, 0.4]]] for e in epochs}   # one channel each
    return {
        "key": "mlp_mnist",
        "label": "MLP × MNIST (dense)",
        "instance_label": "inst003 (random pick, logged)",
        "n_instances": 20,
        "epochs": list(epochs),
        "metrics": [metric],
        "corr_methods": [method],
        "layers": [layer],
        "layer_channels": {layer: 1},
        "layer_kinds": {layer: "dense"},
        "susp": {layer: {metric: grids}},
        "grad": {layer: grids},
        "corr": {layer: {metric: {method: grids}}},
        "corr_stats": {layer: {metric: {method: {
            e: {"mean_abs": 0.5, "median": 0.4, "frac_strong": 0.25, "frac_zero": 0.1}
            for e in epochs}}}},
    }


def _embedded_payload(html):
    match = re.search(r'<script class="payload" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(match.group(1))


def test_report_embeds_every_captured_epoch(tmp_path):
    out = build_ensemble_report(tmp_path / "r.html", combo_payloads=[_payload()])
    data = _embedded_payload(out.read_text(encoding="utf-8"))

    assert data["epochs"] == [0, 1, 10]
    # Each quantity carries a grid for every epoch, so one row can show them all.
    assert sorted(data["grad"]["fc1"]) == ["0", "1", "10"]
    assert sorted(data["corr"]["fc1"]["ochiai"]["pearson"]) == ["0", "1", "10"]


def test_epochs_are_not_behind_a_dropdown(tmp_path):
    html = build_ensemble_report(tmp_path / "r.html", combo_payloads=[_payload()]).read_text(encoding="utf-8")

    assert 'data-role="epoch"' not in html          # no epoch <select>
    assert 'data-role="rows"' in html               # the row container instead
    # The remaining dropdowns still exist.
    for role in ("layer", "channel", "metric", "method"):
        assert f'data-role="{role}"' in html


def test_rows_are_built_per_epoch_on_a_shared_scale(tmp_path):
    html = build_ensemble_report(tmp_path / "r.html", combo_payloads=[_payload()]).read_text(encoding="utf-8")

    # One card per epoch, created from data.epochs by the runtime...
    assert "data.epochs.map(function (e)" in html
    assert "epoch-card" in html and "row-cards" in html
    # ...and susp/grad share one lo/hi across those epochs (correlation is fixed
    # to -1..+1), which is what makes the comparison honest.
    assert "shared colour scale across the epochs" in html
    assert "drawHeatmap(canvas, grids[i], row.diverging, lo, hi)" in html


def test_epoch_zero_is_labelled_as_untrained(tmp_path):
    html = build_ensemble_report(tmp_path / "r.html", combo_payloads=[_payload()]).read_text(encoding="utf-8")
    assert "before training" in html


def test_correlation_cards_show_the_precomputed_stats(tmp_path):
    out = build_ensemble_report(tmp_path / "r.html", combo_payloads=[_payload()])
    html = out.read_text(encoding="utf-8")
    data = _embedded_payload(html)

    assert data["corr_stats"]["fc1"]["ochiai"]["pearson"]["10"]["mean_abs"] == 0.5
    assert "mean |r| " in html


def test_any_number_of_epochs_works(tmp_path):
    out = build_ensemble_report(tmp_path / "r.html", combo_payloads=[_payload(epochs=(0, 2))])
    assert _embedded_payload(out.read_text(encoding="utf-8"))["epochs"] == [0, 2]


def test_empty_report_still_renders(tmp_path):
    out = build_ensemble_report(tmp_path / "empty.html", combo_payloads=[])
    assert "Nothing to show" in out.read_text(encoding="utf-8")
