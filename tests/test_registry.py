"""Tests for the model/dataset registry (incl. the new dense MNIST MLP)."""

import torch

from susgrad.registry import COMBINATIONS, COMBINATIONS_BY_KEY, get_combination


def test_combination_keys_are_unique_and_filesystem_safe():
    keys = [c.key for c in COMBINATIONS]
    assert len(keys) == len(set(keys)), "combination keys must be unique"
    for k in keys:
        assert k.replace("_", "").isalnum(), f"key {k!r} not filesystem-safe"


def test_mlp_mnist_combination_builds_and_runs():
    combo = get_combination("mlp_mnist")
    model = combo.build_model(input_shape=(1, 28, 28), num_classes=10)
    # Dense model: must accept a (B, 1, 28, 28) image batch and emit 10 logits.
    out = model(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)
    assert model.input_shape == (1, 28, 28)
    assert model.output_dim == 10


def test_tabular_combination_output_matches_classes():
    combo = get_combination("mlp_banknote")
    model = combo.build_model(input_shape=(4,), num_classes=2)
    out = model(torch.randn(8, 4))
    assert out.shape == (8, 2)


def test_lenet_combination_builds():
    combo = get_combination("lenet_mnist")
    model = combo.build_model(input_shape=(1, 28, 28), num_classes=10)
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)


def test_lenet_is_size_adaptive_for_cifar():
    # 3x32x32 RGB input must work (fc1 inferred from a dummy forward).
    combo = get_combination("lenet_cifar10")
    model = combo.build_model(input_shape=(3, 32, 32), num_classes=10)
    assert model.input_shape == (3, 32, 32)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_dense_mlp_handles_cifar():
    combo = get_combination("mlp_cifar10")
    model = combo.build_model(input_shape=(3, 32, 32), num_classes=10)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_all_image_combos_present():
    keys = set(COMBINATIONS_BY_KEY)
    for model in ("mlp", "lenet"):
        for ds in ("mnist", "fmnist", "cifar10"):
            assert f"{model}_{ds}" in keys
