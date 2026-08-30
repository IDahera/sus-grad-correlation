"""Tests for the condensed tables in the curated paper set.

Two tables that are easy to confuse and must not be: ``correlation_summary``
describes the CORRELATION (mean |r|), ``value_ranges`` describes the RAW
suspiciousness/gradient values that go into it. These pin the columns each one
carries, including the two deliberate omissions.
"""

import csv

import pytest

from scripts.paper_set import _condensed_correlation, _condensed_stats, _readme
from susgrad.stats import ALL_LAYERS


def _write(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _value_stats(tmp_path):
    columns = ["experiment", "combo", "layer", "kind", "epoch", "instances", "n",
               "min", "p05", "median", "mean", "p95", "max", "std", "frac_zero"]
    rows = []
    for kind, scale in (("ochiai", 0.6), ("dstar", 1.9e7)):
        for epoch, factor in ((0, 1.0), (1, 0.2), (10, 0.1)):
            rows.append({
                "experiment": "ensemble", "combo": "mlp_mnist", "layer": "net.1",
                "kind": kind, "epoch": epoch, "instances": 100, "n": 12800,
                "min": 0.0, "p05": 0.0, "median": scale * factor,
                "mean": scale * factor * 1.05, "p95": scale * factor * 2,
                "max": scale * factor * 3, "std": 0.1, "frac_zero": 0.065,
            })
    return _write(tmp_path / "value_stats.csv", columns, rows)


def _correlation_stats(tmp_path):
    columns = ["combo", "layer", "metric", "method", "epoch", "instances", "n",
               "mean_r", "mean_abs_r", "median_r", "std", "frac_strong",
               "frac_pos", "frac_neg", "frac_zero", "min_r", "max_r"]
    rows = []
    for layer in ("net.1", "net.3"):
        for metric in ("ochiai", "dstar"):
            for method, base in (("spearman", 0.9), ("pearson", 0.7)):
                for epoch in (0, 1, 10):
                    rows.append({
                        "combo": "mlp_mnist", "layer": layer, "metric": metric,
                        "method": method, "epoch": epoch, "instances": 100, "n": 128,
                        "mean_r": base, "mean_abs_r": base, "median_r": base,
                        "std": 0.02, "frac_strong": 1.0, "frac_pos": 1.0,
                        "frac_neg": 0.0, "frac_zero": 0.0, "min_r": 0.5, "max_r": 0.99,
                    })
    return _write(tmp_path / "correlation_stats.csv", columns, rows)


# --- the results table ----------------------------------------------------------

def test_correlation_table_shows_both_methods_per_epoch(tmp_path):
    sections, rows = _condensed_correlation(
        _correlation_stats(tmp_path), ["mlp_mnist"], [0, 1, 10],
        ["ochiai", "dstar"], ["spearman", "pearson"],
    )
    # One (label, combo, table) section per model requested.
    assert len(sections) == 1 and sections[0][1] == "mlp_mnist"
    table = sections[0][2]
    header = table.splitlines()[0]

    # Spearman first (the headline), Pearson as the reference, three epochs each.
    assert header.index("spearman") < header.index("pearson")
    assert header.count("@ e0") == 2 and header.count("@ e10") == 2
    assert len(rows) == 4          # 2 layers x 2 metrics
    assert rows[0]["spearman_mean_abs_r_e0"] == "0.9"
    assert rows[0]["pearson_mean_abs_r_e0"] == "0.7"
    # The signed mean travels in the CSV even though the table prints |r|.
    assert "spearman_mean_r_e0" in rows[0]


def test_correlation_table_is_empty_for_an_unknown_model(tmp_path):
    table, rows = _condensed_correlation(
        _correlation_stats(tmp_path), "lenet_mnist", [0], ["ochiai"], ["spearman"])
    assert table is None and rows is None


# --- the supporting table -------------------------------------------------------

def test_value_table_keeps_mean_and_median_but_drops_min(tmp_path):
    sections, rows = _condensed_stats(_value_stats(tmp_path), ["mlp_mnist"], "net.1", [0, 1, 10])
    assert len(sections) == 1 and sections[0][1:3] == ("mlp_mnist", "net.1")
    header = sections[0][3].splitlines()[0]

    assert "mean @ e0" in header and "median @ e0" in header and "max @ e0" in header
    # min is exactly 0 for most rows (inactive neurons); `zeros` states that better.
    assert "min" not in header
    assert "zeros @ e10" in header
    assert len(rows) == 2          # ochiai + dstar


def test_value_table_reports_the_zero_share_as_a_percentage(tmp_path):
    sections, _ = _condensed_stats(_value_stats(tmp_path), ["mlp_mnist"], "net.1", [0, 1, 10])
    assert "6.5%" in sections[0][3]


def test_value_table_skips_kinds_missing_an_epoch(tmp_path):
    path = _value_stats(tmp_path)
    kept = [r for r in csv.DictReader(path.open(encoding="utf-8"))
            if not (r["kind"] == "dstar" and r["epoch"] == "10")]
    _write(path, list(kept[0]), kept)

    _, rows = _condensed_stats(path, ["mlp_mnist"], "net.1", [0, 1, 10])
    assert [r["kind"] for r in rows] == ["ochiai"]


def test_an_unknown_layer_falls_back_to_the_pooled_row_or_reports_nothing(tmp_path):
    """Layer names are architecture-specific, so a pinned layer WILL be missing."""
    path = _value_stats(tmp_path)
    # No pooled row in the fixture: nothing to fall back to.
    assert _condensed_stats(path, ["mlp_mnist"], "nope", [0]) == (None, None)

    # With a pooled row present, a missing layer falls back to it rather than
    # dropping the model out of the table silently.
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    pooled = [{**r, "layer": ALL_LAYERS} for r in rows]
    _write(path, list(rows[0]), rows + pooled)

    sections, _ = _condensed_stats(path, ["mlp_mnist"], "nope", [0, 1, 10])
    assert [s[2] for s in sections] == [ALL_LAYERS]


def test_value_ranges_cover_every_model_present(tmp_path):
    path = _value_stats(tmp_path)
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    extra = [{**r, "combo": "lenet_mnist", "layer": "conv1"} for r in rows]
    _write(path, list(rows[0]), rows + extra)

    sections, flat = _condensed_stats(path, None, "net.1", [0, 1, 10])

    # lenet has no net.1 and no pooled row here, so it is reported as absent
    # rather than silently borrowing the MLP's layer.
    assert [s[1] for s in sections] == ["mlp_mnist"]
    assert {r["combo"] for r in flat} == {"mlp_mnist"}


# --- folder READMEs ------------------------------------------------------------

def test_readme_gives_each_figure_its_own_caption(tmp_path):
    """Folder 05 holds two figures; one shared caption could not describe both."""
    path = _readme(
        tmp_path, "05 — Trajectories", "purpose text", "the single-neuron caption",
        [("a.png", "one neuron"), ("b.png", "the population")],
        caption_label="single neuron",
        extra_captions=[("population", "the population caption")],
    )
    text = path.read_text(encoding="utf-8")

    assert "**Caption — single neuron** (paste into `\\caption{}`)." in text
    assert "> the single-neuron caption" in text
    assert "**Caption — population.**" in text
    assert "> the population caption" in text
    # The file list still follows the captions, not the other way round.
    assert text.index("the population caption") < text.index("**Files.**")


def test_readme_without_extra_captions_is_unchanged(tmp_path):
    text = _readme(tmp_path, "04 — Tables", "purpose", "only caption",
                   [("t.md", "a table")]).read_text(encoding="utf-8")

    assert "**Caption (paste into `\\caption{}`).**" in text
    assert "Caption —" not in text


def test_correlation_summary_covers_every_model_present(tmp_path):
    """The summary is the results table, so it must not silently show one model."""
    path = _correlation_stats(tmp_path)
    # The fixture writes mlp_mnist only; add a second model to the same file.
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    extra = [{**r, "combo": "lenet_mnist", "layer": "conv1"} for r in rows]
    _write(path, list(rows[0]), rows + extra)

    sections, flat = _condensed_correlation(
        path, None, [0, 1, 10], ["ochiai", "dstar"], ["spearman", "pearson"],
    )

    assert [combo for _, combo, _ in sections] == ["mlp_mnist", "lenet_mnist"]
    assert {r["combo"] for r in flat} == {"mlp_mnist", "lenet_mnist"}
    # Every row carries its model, so the flat CSV stays usable on its own.
    assert all(r["combo"] for r in flat)


def test_a_single_model_may_still_be_requested_as_a_bare_string(tmp_path):
    sections, _ = _condensed_correlation(
        _correlation_stats(tmp_path), "mlp_mnist", [0, 1, 10],
        ["ochiai"], ["spearman"],
    )
    assert len(sections) == 1
