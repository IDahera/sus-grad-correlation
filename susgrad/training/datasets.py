"""Dataset preparation.

``prepare_dataset`` brings a dataset into the right shape (tensorised) and
normalises it for the model. If a model is supplied it also verifies the two are
compatible and raises :class:`DimensionMismatchError` otherwise.

Supported datasets (all non-medical):
    "banknote"   : UCI Banknote Authentication (4 features; genuine vs forged).
    "sonar"      : Sonar returns (60 features; rock vs mine).
    "ionosphere" : Radar returns (34 features; good vs bad).
    "moons"      : Synthetic two-moons (2 features) -- always available offline.
    "circles"    : Synthetic concentric circles (2 features) -- always offline.
    "mnist"      : MNIST handwritten digits (1x28x28, 10 classes).
    "fmnist"     : Fashion-MNIST clothing images (1x28x28, 10 classes).
    "cifar10"    : CIFAR-10 natural images (3x32x32, 10 classes).

Tabular sets are standardised with a train-fitted scaler. Image sets are scaled
to ~zero mean / unit variance with per-dataset statistics. The OpenML-backed sets
fall back to a synthetic generator if they cannot be downloaded.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from susgrad.training._checks import check_model_dataset_compatible
from susgrad.utils.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

_DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class DatasetBundle:
    """Everything a training script needs about a prepared dataset."""

    name: str
    train_loader: DataLoader
    test_loader: DataLoader
    input_shape: tuple[int, ...]
    num_classes: int

    @property
    def input_dim(self) -> int:
        """Flat feature count (useful for sizing a tabular model)."""
        n = 1
        for d in self.input_shape:
            n *= d
        return n


def prepare_dataset(
    dataset: str,
    *,
    model: Optional[nn.Module] = None,
    batch_size: int = 32,
    test_size: float = 0.2,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> DatasetBundle:
    """Load, tensorise and normalise *dataset*; optionally validate against *model*.

    Args:
        dataset: "banknote" or "mnist".
        model: if given, the prepared loaders are checked for compatibility and
            a :class:`DimensionMismatchError` is raised when the model is not
            cut out for the dataset.
        batch_size: loader batch size.
        test_size: tabular train/test split fraction (ignored for MNIST, which
            ships its own split).
        max_samples: optional cap on the number of training samples (speeds up
            smoke runs).
        seed: split/shuffle seed.

    Returns:
        A :class:`DatasetBundle`.
    """
    name = dataset.lower()
    if name in _IMAGE_SPECS:
        bundle = _prepare_image(name, batch_size, max_samples)
    elif name in _TABULAR_LOADERS:
        X, y = _TABULAR_LOADERS[name](seed)
        bundle = _build_tabular_bundle(
            name, X, y, batch_size, test_size, max_samples, seed
        )
    else:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Choose from "
            f"{sorted(list(_TABULAR_LOADERS) + list(_IMAGE_SPECS))}."
        )

    if model is not None:
        check_model_dataset_compatible(model, bundle.train_loader)

    return bundle


def _build_tabular_bundle(
    name: str, X, y, batch_size: int, test_size: float,
    max_samples: Optional[int], seed: int,
) -> DatasetBundle:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if max_samples is not None and max_samples < len(X):
        X, y = X[:max_samples], y[:max_samples]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Standardise using statistics from the TRAIN split only.
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return DatasetBundle(
        name=name,
        train_loader=DataLoader(
            _to_tensor_dataset(X_train, y_train), batch_size=batch_size, shuffle=True
        ),
        test_loader=DataLoader(
            _to_tensor_dataset(X_test, y_test), batch_size=batch_size, shuffle=False
        ),
        input_shape=(X.shape[1],),
        num_classes=int(len(np.unique(y))),
    )


def _load_openml_binary(data_id: int, label: str, n_features: int, seed: int):
    """Fetch an OpenML binary dataset, mapping its target to 0/1.

    Falls back to a synthetic ``make_classification`` set of the same width if the
    download fails, so scripts keep working offline.
    """
    try:
        from sklearn.datasets import fetch_openml

        ds = fetch_openml(data_id=data_id, as_frame=True, data_home=str(_DATA_DIR))
        X = ds.data.to_numpy(dtype=np.float32)
        y_raw = ds.target.to_numpy()
        classes = sorted(set(y_raw))
        mapping = {c: i for i, c in enumerate(classes)}
        y = np.array([mapping[v] for v in y_raw], dtype=np.int64)
        logger.info("Loaded %s from OpenML: %s", label, X.shape)
        return X, y
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not fetch %s from OpenML (%s); using a synthetic fallback.",
            label, exc,
        )
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=1000, n_features=n_features,
            n_informative=max(2, n_features // 2), n_redundant=0,
            n_classes=2, random_state=seed,
        )
        return X.astype(np.float32), y.astype(np.int64)


def _load_banknote(seed: int):
    return _load_openml_binary(1462, "Banknote Authentication", 4, seed)


def _load_sonar(seed: int):
    return _load_openml_binary(40, "Sonar", 60, seed)


def _load_ionosphere(seed: int):
    return _load_openml_binary(59, "Ionosphere", 34, seed)


def _load_moons(seed: int):
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=1000, noise=0.2, random_state=seed)
    return X.astype(np.float32), y.astype(np.int64)


def _load_circles(seed: int):
    from sklearn.datasets import make_circles

    X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=seed)
    return X.astype(np.float32), y.astype(np.int64)


# Name -> loader returning (X, y). Single place to register tabular datasets.
_TABULAR_LOADERS = {
    "banknote": _load_banknote,
    "sonar": _load_sonar,
    "ionosphere": _load_ionosphere,
    "moons": _load_moons,
    "circles": _load_circles,
}


def load_tabular_raw(name: str, seed: int = 42):
    """Return the raw (un-scaled) ``(X, y)`` arrays for a tabular dataset.

    Useful for previews/inspection. Raises ``ValueError`` for unknown names.
    """
    if name not in _TABULAR_LOADERS:
        raise ValueError(f"{name!r} is not a tabular dataset.")
    return _TABULAR_LOADERS[name](seed)


def _to_tensor_dataset(X, y) -> TensorDataset:
    return TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )


# name -> (torchvision class name, mean, std, input_shape). Single place to add
# an image dataset. Stats are the standard per-dataset normalisation constants.
_IMAGE_SPECS = {
    "mnist":   ("MNIST",        (0.1307,), (0.3081,), (1, 28, 28)),
    "fmnist":  ("FashionMNIST", (0.2860,), (0.3530,), (1, 28, 28)),
    "cifar10": ("CIFAR10",
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), (3, 32, 32)),
}


def _prepare_image(name: str, batch_size: int, max_samples: Optional[int]) -> DatasetBundle:
    from torch.utils.data import Subset
    from torchvision import datasets, transforms

    cls_name, mean, std, input_shape = _IMAGE_SPECS[name]
    cls = getattr(datasets, cls_name)
    transform = transforms.Compose([
        transforms.ToTensor(),                 # -> [0, 1]
        transforms.Normalize(mean, std),       # -> ~zero mean / unit var
    ])

    train_ds = cls(root=str(_DATA_DIR), train=True, download=True, transform=transform)
    test_ds = cls(root=str(_DATA_DIR), train=False, download=True, transform=transform)

    if max_samples is not None:
        train_ds = Subset(train_ds, range(min(max_samples, len(train_ds))))
        test_ds = Subset(test_ds, range(min(max_samples, len(test_ds))))

    return DatasetBundle(
        name=name,
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        input_shape=input_shape,
        num_classes=10,
    )
