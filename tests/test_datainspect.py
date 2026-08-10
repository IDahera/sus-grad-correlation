"""Tests for the datasets-overview HTML (static descriptions, no data needed)."""

from susgrad.datainspect import DATASET_META, build_datasets_html


def test_datasets_html_covers_all_and_shows_ground_truth():
    names = ["banknote", "sonar", "mnist", "fmnist", "cifar10"]
    html = build_datasets_html(names, with_samples=False)

    # Titles present.
    assert "Banknote Authentication" in html and "CIFAR-10" in html
    # Tabular features and ground-truth wording.
    assert "variance" in html                      # banknote feature
    assert "Ground truth" in html
    # Image class labels (ground truth) present.
    assert "Ankle boot" in html                    # fmnist class
    assert "airplane" in html                      # cifar10 class


def test_datasets_html_dedupes_repeated_names():
    # mnist appears for both mlp_mnist and lenet_mnist -> only one section.
    html = build_datasets_html(["mnist", "mnist", "mnist"], with_samples=False)
    assert html.count("<h2>MNIST") == 1


def test_long_feature_lists_are_truncated():
    html = build_datasets_html(["sonar"], with_samples=False)  # 60 features
    assert "more" in html  # "+N more" chip rather than 60 chips


def test_every_registry_dataset_has_metadata():
    from susgrad.registry import COMBINATIONS

    for c in COMBINATIONS:
        assert c.dataset in DATASET_META, f"no metadata for {c.dataset}"
