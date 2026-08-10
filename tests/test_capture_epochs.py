"""Tests for the ensemble pipeline's epoch-selection parsing.

``parse_epoch_list`` is what turns ``--capture-epochs 0,1,10`` (train), and the
matching ``--epochs`` of the correlate/evaluate steps, into the actual snapshot
list. Epoch 0 is legal here (it means "before any training"), which is the one
place this differs from the secondary pipeline's 1-based windows -- so that,
plus the ``max_epoch`` guard that stops you asking for a snapshot the training
run never reaches, is what these tests pin down.
"""

import click
import pytest

from scripts._cli import parse_epoch_list


def test_parses_the_base_case():
    assert parse_epoch_list("0,1,10") == [0, 1, 10]


def test_sorts_deduplicates_and_ignores_whitespace():
    assert parse_epoch_list(" 10, 0 ,1, 10 ") == [0, 1, 10]


def test_epoch_zero_is_allowed_but_negatives_are_not():
    assert parse_epoch_list("0") == [0]
    with pytest.raises(click.BadParameter):
        parse_epoch_list("-1,0")


def test_rejects_epochs_beyond_the_training_length():
    assert parse_epoch_list("0,1,10", max_epoch=10) == [0, 1, 10]
    with pytest.raises(click.BadParameter):
        parse_epoch_list("0,1,20", max_epoch=10)


def test_rejects_non_integers_and_empty_input():
    with pytest.raises(click.BadParameter):
        parse_epoch_list("0,one,10")
    with pytest.raises(click.BadParameter):
        parse_epoch_list("")


def test_auto_only_when_allowed():
    assert parse_epoch_list("auto", allow_auto=True) is None
    with pytest.raises(click.BadParameter):
        parse_epoch_list("auto")
