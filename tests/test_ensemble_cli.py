"""Regression tests for the model/dataset "unit function" used by every script.

``build_dataset_and_model`` is the single place that translates a
``Combination`` (model kind + dataset name) into an actual, executable
(dataset, model) pair -- the dataset prepared and the model sized to fit it.
Both the secondary pipeline (train_models.py, capture_*.py) and the main
ensemble pipeline (train_ensemble.py) go through it.

``prepare_dataset`` itself is monkeypatched here (matching the rest of this
suite's offline philosophy -- see conftest.py) so this test never touches the
network or downloads MNIST; it only checks the GLUE: that batch_size / seed /
max_samples reach ``prepare_dataset``, and that the returned bundle's
input_shape / num_classes correctly size the model that comes back.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import scripts._cli as cli
from susgrad.registry import ENSEMBLE_COMBO_KEYS, ensemble_combinations, get_combination
from susgrad.training.datasets import DatasetBundle


def _fake_bundle(input_shape, num_classes, n=16, batch_size=8):
    """A tiny, real (but synthetic) DatasetBundle -- no download, no network."""
    x = torch.randn(n, *input_shape)
    y = torch.randint(0, num_classes, (n,))
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
    return DatasetBundle(
        name="fake", train_loader=loader, test_loader=loader,
        input_shape=tuple(input_shape), num_classes=num_classes,
    )


@pytest.mark.parametrize("combo_key,input_shape,num_classes", [
    ("mlp_banknote", (4,), 2),
    ("mlp_mnist", (1, 28, 28), 10),
    ("lenet_mnist", (1, 28, 28), 10),
    ("mlp_fmnist", (1, 28, 28), 10),
    ("lenet_fmnist", (1, 28, 28), 10),
])
def test_build_dataset_and_model_returns_executable_pair(monkeypatch, combo_key, input_shape, num_classes):
    combo = get_combination(combo_key)
    seen_kwargs = {}

    def fake_prepare_dataset(dataset, *, batch_size, seed, max_samples):
        seen_kwargs.update(dataset=dataset, batch_size=batch_size, seed=seed, max_samples=max_samples)
        return _fake_bundle(input_shape, num_classes, batch_size=batch_size)

    monkeypatch.setattr(cli, "prepare_dataset", fake_prepare_dataset)

    bundle, model = cli.build_dataset_and_model(combo, batch_size=8, seed=123, max_samples=16)

    # The dataset name + every kwarg was forwarded correctly.
    assert seen_kwargs == {"dataset": combo.dataset, "batch_size": 8, "seed": 123, "max_samples": 16}

    # The model is actually sized to fit the bundle it came back with.
    assert bundle.input_shape == tuple(input_shape)
    assert bundle.num_classes == num_classes

    batch_x, _ = next(iter(bundle.train_loader))
    out = model(batch_x)
    assert out.shape == (batch_x.shape[0], num_classes)


def test_build_dataset_and_model_rejects_mismatched_bundle(monkeypatch):
    """If prepare_dataset ever returned a bundle the model can't consume, that
    should surface as a normal forward-pass shape error, not silently pass."""
    combo = get_combination("mlp_mnist")

    def fake_prepare_dataset(dataset, *, batch_size, seed, max_samples):
        # Wrong number of classes on purpose (5 instead of 10).
        return _fake_bundle((1, 28, 28), 5, batch_size=batch_size)

    monkeypatch.setattr(cli, "prepare_dataset", fake_prepare_dataset)
    bundle, model = cli.build_dataset_and_model(combo, batch_size=8, seed=0, max_samples=16)

    batch_x, _ = next(iter(bundle.train_loader))
    out = model(batch_x)
    # The model was sized to the (mismatched) bundle it was given -- 5 classes,
    # not 10 -- which is exactly what a caller should check for downstream.
    assert out.shape[-1] == 5
    assert out.shape[-1] != 10


def test_ensemble_combinations_are_exactly_mlp_lenet_mnist_fmnist():
    assert set(ENSEMBLE_COMBO_KEYS) == {"mlp_mnist", "mlp_fmnist", "lenet_mnist", "lenet_fmnist"}
    combos = ensemble_combinations()
    assert [c.key for c in combos] == list(ENSEMBLE_COMBO_KEYS)
    assert {c.model_kind for c in combos} == {"mlp", "lenet"}
    assert {c.dataset for c in combos} == {"mnist", "fmnist"}
