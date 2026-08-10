"""Regression tests for prepare_dataset / compatibility checks (input & output dims)."""

import pytest

from susgrad.models import TabularMLP
from susgrad.training import DimensionMismatchError
from susgrad.training._checks import check_model_dataset_compatible


def test_input_output_dims_match(model, normalized_loader):
    # Should not raise: 6 features in, 2 classes out, matching the fixture model.
    check_model_dataset_compatible(model, normalized_loader)


def test_wrong_input_dim_raises(normalized_loader):
    wrong = TabularMLP(input_dim=99, output_dim=2)  # expects 99 features, gets 6
    with pytest.raises(DimensionMismatchError):
        check_model_dataset_compatible(wrong, normalized_loader)


def test_too_few_output_classes_raises(normalized_loader):
    # Model with a single output logit cannot represent label "1".
    wrong = TabularMLP(input_dim=6, output_dim=1)
    with pytest.raises(DimensionMismatchError):
        check_model_dataset_compatible(wrong, normalized_loader)
