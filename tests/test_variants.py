"""Tests for variant naming, discovery and completeness."""

import torch

from susgrad.persistence import (
    list_epochs,
    list_variants,
    save_gradients,
    variant_complete,
    variant_name,
    variant_stop,
)
from scripts._cli import parse_stops


def test_variant_name_roundtrip():
    assert variant_name(10) == "e010"
    assert variant_name(100) == "e100"
    assert variant_stop("e010") == 10
    assert variant_stop(variant_name(50)) == 50


def test_parse_stops_sorts_dedupes_and_defaults():
    assert parse_stops("100,10,50,10", epochs=999) == [10, 50, 100]
    assert parse_stops("", epochs=10) == [10]   # default to a single stop at epochs


def test_list_variants_and_completeness(tmp_path):
    g = {"fc": torch.zeros(4)}
    # Write a complete 1..3 window and a partial 1..3 window (missing epoch 3).
    for e in (1, 2, 3):
        save_gradients(g, "combo", variant_name(3), epoch=e, base_dir=tmp_path)
    for e in (1, 2):
        save_gradients(g, "combo", variant_name(5), epoch=e, base_dir=tmp_path)

    assert list_variants(tmp_path, "combo") == ["e003", "e005"]
    assert list_epochs(tmp_path, "combo", "e003") == [1, 2, 3]
    assert variant_complete(tmp_path, "combo", 3) is True
    assert variant_complete(tmp_path, "combo", 5) is False  # missing epochs 3..5
