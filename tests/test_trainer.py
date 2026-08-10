"""Tests for train() and evaluate()."""

import pytest

from susgrad.models import TabularMLP
from susgrad.training import (
    DataNotNormalizedError,
    DimensionMismatchError,
    evaluate,
    train,
)


def _param_shapes(model):
    return {name: tuple(p.shape) for name, p in model.state_dict().items()}


def test_training_preserves_parameter_shapes(model, normalized_loader):
    before = _param_shapes(model)
    train(model, normalized_loader, epochs=1, progress=False)
    after = _param_shapes(model)
    assert before == after, "Training must not change the model's dimensions."


def test_training_changes_values(model, normalized_loader):
    # Snapshot on CPU; training may move the model to MPS/CUDA, so compare on CPU.
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train(model, normalized_loader, epochs=2, progress=False)
    after = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    changed = any(not after[k].equal(before[k]) for k in before)
    assert changed, "Training should actually update some weights."


def test_train_raises_on_dim_mismatch(normalized_loader):
    wrong = TabularMLP(input_dim=99, output_dim=2)
    with pytest.raises(DimensionMismatchError):
        train(wrong, normalized_loader, epochs=1, progress=False)


def test_evaluate_returns_sensible_result(model, normalized_loader):
    train(model, normalized_loader, epochs=3, progress=False)
    result = evaluate(model, normalized_loader)
    assert 0.0 <= result.accuracy <= 100.0
    assert result.n_samples == 64


def test_evaluate_rejects_unnormalized_data(model, raw_loader):
    with pytest.raises(DataNotNormalizedError):
        evaluate(model, raw_loader)
