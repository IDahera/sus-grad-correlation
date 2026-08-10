"""Tests for the value-range statistics behind ``scripts/value_stats.py``.

The table these feed is what lets a reader compare metrics whose scales differ by
seven orders of magnitude, so two things have to be right: the numbers
themselves, and the fact that inactive neurons (exact zeros) are reported
separately rather than silently dragging the mean down.
"""

import numpy as np
import pytest
import torch

from susgrad.stats import STAT_FIELDS, describe_stack, describe_values, format_row


def test_describe_values_reports_the_full_five_number_summary():
    stats = describe_values(np.arange(101, dtype=float))   # 0..100

    assert stats["n"] == 101
    assert stats["min"] == 0.0 and stats["max"] == 100.0
    assert stats["median"] == pytest.approx(50.0)
    assert stats["mean"] == pytest.approx(50.0)
    assert stats["p05"] == pytest.approx(5.0)
    assert stats["p95"] == pytest.approx(95.0)
    assert set(stats) == set(STAT_FIELDS)


def test_zero_share_is_reported_not_hidden():
    # Half the "neurons" never activate -- the mean alone would not say so.
    stats = describe_values(np.array([0.0, 0.0, 0.5, 0.5]))
    assert stats["frac_zero"] == pytest.approx(0.5)
    assert stats["mean"] == pytest.approx(0.25)
    assert stats["min"] == 0.0


def test_non_finite_values_are_dropped():
    stats = describe_values(np.array([1.0, np.nan, np.inf, 3.0]))
    assert stats["n"] == 2
    assert stats["mean"] == pytest.approx(2.0)


def test_empty_input_is_all_zeros_not_a_crash():
    stats = describe_values(np.array([]))
    assert stats["n"] == 0
    assert all(stats[f] == 0.0 for f in STAT_FIELDS if f != "n")


def test_accepts_torch_tensors_of_any_shape():
    stats = describe_values(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))
    assert stats["n"] == 24 and stats["max"] == 23.0


def test_describe_stack_pools_every_instance():
    # One tensor per "instance", as the ensemble experiment stores them.
    tensors = [torch.full((5,), float(i)) for i in range(4)]
    stats = describe_stack(tensors)

    assert stats["n"] == 20
    assert stats["min"] == 0.0 and stats["max"] == 3.0
    assert stats["mean"] == pytest.approx(1.5)


def test_describe_stack_of_nothing_is_empty():
    assert describe_stack([])["n"] == 0


def test_format_row_keeps_n_integral_and_the_rest_significant():
    row = format_row(describe_values(np.array([1e-9, 4.55e8])))
    assert row["n"] == "2"
    # %g keeps the magnitude readable instead of collapsing to 0.000000.
    assert "e" in row["max"].lower() or float(row["max"]) == 4.55e8
    assert set(row) == set(STAT_FIELDS)
