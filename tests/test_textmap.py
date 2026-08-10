"""Tests for the plain-text heatmaps and correlation statistics.

These back the text report ``correlate_ensemble.py`` writes: the ASCII map must
preserve the grid's shape and the SIGN of every correlation (a positive and a
negative neuron must never render as the same glyph), and the statistics must
treat "undefined -> 0" neurons visibly rather than hiding them in a signed mean.
"""

import numpy as np
import pytest

from susgrad.viz.textmap import (
    DIV_BINS,
    correlation_summary,
    markdown_table,
    text_heatmap,
)


def _map_lines(text):
    """The rendered rows (everything but the trailing caption line)."""
    return [line for line in text.splitlines() if not line.strip().startswith("(")]


def test_diverging_map_keeps_shape_and_sign():
    grid = np.array([[1.0, -1.0, 0.0], [0.6, -0.6, 0.3]])
    lines = _map_lines(text_heatmap(grid, diverging=True, indent=""))

    assert len(lines) == 2 and all(len(line) == 3 for line in lines)
    strong_pos, strong_neg = lines[0][0], lines[0][1]
    assert strong_pos == "#" and strong_neg == "@"
    assert strong_pos != strong_neg
    # Zero is its own glyph, distinct from both tails.
    assert lines[0][2] == "." and lines[0][2] not in (strong_pos, strong_neg)


def test_positive_and_negative_glyph_sets_are_disjoint():
    positives = {g for bound, g in DIV_BINS if bound > 0.05}
    negatives = {g for bound, g in DIV_BINS if bound <= -0.05}
    assert positives.isdisjoint(negatives)


def test_nan_padding_renders_as_blank_not_as_a_value():
    grid = np.array([[0.9, np.nan]])
    line = _map_lines(text_heatmap(grid, diverging=True, indent=""))[0]
    assert line[0] == "#" and line[1] == " "


def test_large_grids_are_block_averaged_and_say_so():
    grid = np.random.default_rng(0).uniform(-1, 1, size=(200, 200))
    text = text_heatmap(grid, diverging=True, max_rows=20, max_cols=20, indent="")
    lines = _map_lines(text)

    assert len(lines) == 20 and all(len(line) == 20 for line in lines)
    assert "block-averaged" in text and "200x200" in text


def test_sequential_map_spans_its_own_min_to_max():
    grid = np.array([[0.0, 0.5, 1.0]])
    line = _map_lines(text_heatmap(grid, diverging=False, indent=""))[0]
    # Low end is the sparsest glyph, high end the densest, and they differ.
    assert line[0] == " " and line[2] == "@" and line[0] != line[1] != line[2]


def test_indent_is_applied_to_every_line():
    text = text_heatmap(np.zeros((2, 2)), diverging=True, indent="    ")
    assert all(line.startswith("    ") for line in text.splitlines())


def test_text_heatmap_rejects_non_2d():
    with pytest.raises(ValueError):
        text_heatmap(np.zeros(4), diverging=True)


def test_correlation_summary_separates_magnitude_from_sign():
    values = np.array([1.0, -1.0, 0.0, 0.0])
    s = correlation_summary(values)

    assert s["n"] == 4
    assert s["mean"] == pytest.approx(0.0)      # +1 and -1 cancel...
    assert s["mean_abs"] == pytest.approx(0.5)  # ...but the coupling is not zero
    assert s["frac_strong"] == pytest.approx(0.5)
    assert s["frac_zero"] == pytest.approx(0.5)
    assert s["frac_pos"] == pytest.approx(0.25)
    assert s["frac_neg"] == pytest.approx(0.25)
    assert (s["min"], s["max"]) == (-1.0, 1.0)


def test_correlation_summary_ignores_non_finite_and_empty():
    s = correlation_summary(np.array([np.nan, 0.5, np.inf]))
    assert s["n"] == 1 and s["mean"] == pytest.approx(0.5)
    assert correlation_summary(np.array([]))["n"] == 0


def test_markdown_table_has_a_header_separator_row():
    table = markdown_table(["layer", "mean"], [["`fc1`", "0.5000"]]).splitlines()
    assert table[0].startswith("| layer") and set(table[1]) <= set("|-: ")
    assert "`fc1`" in table[2]
