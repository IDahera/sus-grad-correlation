"""The single source of truth for model/dataset combinations.

Every script (training, gradient capture, suspiciousness capture, visualisation)
reads its work items from :data:`COMBINATIONS` here, so there is exactly one
place to add, remove or toggle a model/dataset pair.

Each :class:`Combination` has a stable ``key`` used throughout for:
    * CLI on/off flags (``--<key>`` / ``--no-<key>``),
    * saved model filenames, and
    * the gradient/suspiciousness output subfolders.
"""

import math
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import torch.nn as nn

from susgrad.models import LeNet5, MLPClassifier, TabularMLP


@dataclass(frozen=True)
class Combination:
    """A model architecture paired with a dataset."""

    key: str            # unique, filesystem-safe id, e.g. "mlp_banknote"
    label: str          # human-readable, e.g. "MLP × Banknote"
    dataset: str        # dataset name understood by prepare_dataset
    model_kind: str     # "tabular" | "mlp" | "lenet"
    hidden: Tuple[int, ...] = field(default=(64, 32))

    def build_model(self, input_shape: Sequence[int], num_classes: int) -> nn.Module:
        """Construct a fresh model sized for this combination's dataset.

        ``input_shape`` is the per-sample shape (e.g. ``(4,)`` for tabular,
        ``(1, 28, 28)`` for MNIST).
        """
        input_dim = int(math.prod(input_shape))
        if self.model_kind == "tabular":
            return TabularMLP(
                input_dim=input_dim, output_dim=num_classes, hidden_dims=self.hidden
            )
        if self.model_kind == "mlp":
            return MLPClassifier(
                input_shape=input_shape, num_classes=num_classes, hidden_dims=self.hidden
            )
        if self.model_kind == "lenet":
            return LeNet5(num_classes=num_classes, input_shape=input_shape)
        raise ValueError(f"Unknown model_kind {self.model_kind!r}")


COMBINATIONS: Tuple[Combination, ...] = (
    # Tabular binary-classification baselines.
    Combination("mlp_banknote", "MLP × Banknote", "banknote", "tabular", (64, 32)),
    Combination("mlp_sonar", "MLP × Sonar", "sonar", "tabular", (128, 64)),
    Combination("mlp_ionosphere", "MLP × Ionosphere", "ionosphere", "tabular", (64, 32)),
    Combination("mlp_moons", "MLP × Moons (synthetic)", "moons", "tabular", (32, 16)),
    Combination("mlp_circles", "MLP × Circles (synthetic)", "circles", "tabular", (32, 16)),
    # Image: dense MLP and LeNet-5, each on MNIST / Fashion-MNIST / CIFAR-10.
    Combination("mlp_mnist", "MLP × MNIST (dense)", "mnist", "mlp", (128, 64)),
    Combination("mlp_fmnist", "MLP × Fashion-MNIST (dense)", "fmnist", "mlp", (128, 64)),
    Combination("mlp_cifar10", "MLP × CIFAR-10 (dense)", "cifar10", "mlp", (256, 128)),
    Combination("lenet_mnist", "LeNet-5 × MNIST", "mnist", "lenet"),
    Combination("lenet_fmnist", "LeNet-5 × Fashion-MNIST", "fmnist", "lenet"),
    Combination("lenet_cifar10", "LeNet-5 × CIFAR-10", "cifar10", "lenet"),
)

COMBINATIONS_BY_KEY = {c.key: c for c in COMBINATIONS}


def get_combination(key: str) -> Combination:
    if key not in COMBINATIONS_BY_KEY:
        raise KeyError(f"Unknown combination {key!r}. Known: {list(COMBINATIONS_BY_KEY)}")
    return COMBINATIONS_BY_KEY[key]


IMAGE_DATASETS = {"mnist", "fmnist", "cifar10"}


def domain_of(combo: Combination) -> str:
    """'image' if the dataset is image-based, else 'tabular'."""
    return "image" if combo.dataset in IMAGE_DATASETS else "tabular"


def select_combinations(enabled: dict[str, bool], only: str = "all") -> list[Combination]:
    """Return combinations that are toggled on and match the ``only`` domain filter.

    Args:
        enabled: ``{key: bool}`` from the CLI flags.
        only: "all", "tabular" or "image".
    """
    chosen = []
    for c in COMBINATIONS:
        if not enabled.get(c.key, True):
            continue
        if only != "all" and domain_of(c) != only:
            continue
        chosen.append(c)
    return chosen


# --- main (ensemble) pipeline scope --------------------------------------------
# The main pipeline considers EXACTLY these 4 pairs: MLP and LeNet, separately, on
# MNIST and Fashion-MNIST. Kept centrally here (like COMBINATIONS above) so every
# ensemble script (train/correlate/evaluate/overview) reads the same source of
# truth instead of re-listing the keys.
ENSEMBLE_COMBO_KEYS: Tuple[str, ...] = (
    "mlp_mnist", "mlp_fmnist", "lenet_mnist", "lenet_fmnist",
)


def ensemble_combinations() -> Tuple[Combination, ...]:
    """The 4 Combinations the main (ensemble) pipeline trains: MLP/LeNet x MNIST/FMNIST."""
    return tuple(COMBINATIONS_BY_KEY[k] for k in ENSEMBLE_COMBO_KEYS)
