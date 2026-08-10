"""Store/load per-epoch gradient/suspiciousness and per-variant correlation.

A **variant** is a captured window of epochs starting at 1, e.g. ``1..10`` or
``1..100`` — named ``e010`` / ``e100``. Each variant is stored separately so the
across-epoch correlation can be measured over different horizons and compared.

Layout::

    outputs/gradients/<combo>/<variant>/epoch_<NN>.pt        {layer: tensor}
    outputs/suspiciousness/<combo>/<variant>/epoch_<NN>.pt    {metric: {layer: tensor}}
    outputs/correlation/<combo>/<variant>/correlation.pt
        -> {susp_metric: {corr_method: {layer: tensor}}}

Stored tensors keep the exact per-layer shape produced from the activations, so
loaded dimensions always reflect the model's dimensions.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from susgrad.utils.paths import (
    CORRELATION_DIR,
    GRADIENTS_DIR,
    SUSPICIOUSNESS_DIR,
    ensure_dir,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
LayerTensors = Dict[str, torch.Tensor]
MetricLayerTensors = Dict[str, LayerTensors]


# --- variant naming ------------------------------------------------------------

def variant_name(stop: int) -> str:
    """Name for the window ``1..stop`` (e.g. 10 -> 'e010', 100 -> 'e100')."""
    return f"e{int(stop):03d}"


def variant_stop(name: str) -> int:
    """Inverse of :func:`variant_name` ('e010' -> 10)."""
    m = re.match(r"e(\d+)$", name)
    if not m:
        raise ValueError(f"Not a variant name: {name!r}")
    return int(m.group(1))


# --- main (ensemble) pipeline naming --------------------------------------------
# The ensemble pipeline reuses the exact same save/load functions below, just
# passing a different kind of label in the ``variant`` slot: an *instance* id
# (for gradients/suspiciousness, one randomly-initialised model per id) or an
# *epoch snapshot* id (for correlation, since the ensemble correlates across
# instances separately at each epoch, rather than across epochs for one instance).

def instance_name(i: int) -> str:
    """Name for ensemble instance *i* (e.g. 0 -> 'inst000')."""
    return f"inst{int(i):03d}"


def instance_index(name: str) -> int:
    """Inverse of :func:`instance_name` ('inst000' -> 0)."""
    m = re.match(r"inst(\d+)$", name)
    if not m:
        raise ValueError(f"Not an instance name: {name!r}")
    return int(m.group(1))


def seed_run_name(seed: int) -> str:
    """Name for experiment 2's single seeded run (e.g. 42 -> 'seed042').

    Experiment 2 trains ONE instance per combo over many epochs, so the
    ``variant`` slot holds the seed that produced it: two runs with different
    seeds land in different folders instead of overwriting each other.
    """
    return f"seed{int(seed):03d}"


def seed_run_seed(name: str) -> int:
    """Inverse of :func:`seed_run_name` ('seed042' -> 42)."""
    m = re.match(r"seed(\d+)$", name)
    if not m:
        raise ValueError(f"Not a seed-run name: {name!r}")
    return int(m.group(1))


def epoch_snapshot_name(epoch: int) -> str:
    """Name for the ensemble's per-epoch (across-instances) correlation snapshot."""
    return f"epoch{int(epoch):02d}"


def epoch_snapshot_index(name: str) -> int:
    """Inverse of :func:`epoch_snapshot_name` ('epoch00' -> 0)."""
    m = re.match(r"epoch(\d+)$", name)
    if not m:
        raise ValueError(f"Not an epoch snapshot name: {name!r}")
    return int(m.group(1))


def list_instances(base_dir: PathLike, combo_key: str) -> List[str]:
    """Instance subfolders present for *combo_key* (ensemble pipeline), sorted by index."""
    folder = Path(base_dir) / combo_key
    if not folder.is_dir():
        return []
    out = []
    for d in folder.iterdir():
        if d.is_dir() and re.match(r"inst\d+$", d.name):
            out.append(d.name)
    return sorted(out, key=instance_index)


def _epoch_file(base_dir: Path, combo_key: str, variant: str, epoch: int) -> Path:
    return ensure_dir(Path(base_dir) / combo_key / variant) / f"epoch_{epoch:02d}.pt"


def list_variants(base_dir: PathLike, combo_key: str) -> List[str]:
    """Variant subfolders present for *combo_key*, sorted by their stop."""
    folder = Path(base_dir) / combo_key
    if not folder.is_dir():
        return []
    variants = []
    for d in folder.iterdir():
        if d.is_dir() and re.match(r"e\d+$", d.name):
            variants.append(d.name)
    return sorted(variants, key=variant_stop)


def list_epochs(base_dir: PathLike, combo_key: str, variant: str) -> List[int]:
    """Sorted epoch numbers stored for a (combo, variant)."""
    folder = Path(base_dir) / combo_key / variant
    if not folder.is_dir():
        return []
    epochs = []
    for f in folder.glob("epoch_*.pt"):
        m = re.match(r"epoch_(\d+)\.pt$", f.name)
        if m:
            epochs.append(int(m.group(1)))
    return sorted(epochs)


def variant_complete(base_dir: PathLike, combo_key: str, stop: int) -> bool:
    """True if epochs ``1..stop`` are all present for the variant."""
    have = set(list_epochs(base_dir, combo_key, variant_name(stop)))
    return set(range(1, stop + 1)).issubset(have)


# --- gradients -----------------------------------------------------------------

def gradients_path(combo_key: str, variant: str, epoch: int,
                   base_dir: Optional[PathLike] = None) -> Path:
    base = Path(base_dir) if base_dir is not None else GRADIENTS_DIR
    return base / combo_key / variant / f"epoch_{epoch:02d}.pt"


def save_gradients(gradients: LayerTensors, combo_key: str, variant: str, epoch: int,
                   base_dir: Optional[PathLike] = None) -> Path:
    base = Path(base_dir) if base_dir is not None else GRADIENTS_DIR
    path = _epoch_file(base, combo_key, variant, epoch)
    torch.save({n: t.detach().cpu() for n, t in gradients.items()}, path)
    return path


def load_gradients(combo_key: str, variant: str, epoch: int,
                   base_dir: Optional[PathLike] = None) -> LayerTensors:
    path = gradients_path(combo_key, variant, epoch, base_dir)
    if not path.exists():
        raise FileNotFoundError(f"No gradient file at {path}")
    return torch.load(path, weights_only=True)


# --- suspiciousness ------------------------------------------------------------

def suspiciousness_path(combo_key: str, variant: str, epoch: int,
                        base_dir: Optional[PathLike] = None) -> Path:
    base = Path(base_dir) if base_dir is not None else SUSPICIOUSNESS_DIR
    return base / combo_key / variant / f"epoch_{epoch:02d}.pt"


def save_suspiciousness(suspiciousness: MetricLayerTensors, combo_key: str, variant: str,
                        epoch: int, base_dir: Optional[PathLike] = None) -> Path:
    base = Path(base_dir) if base_dir is not None else SUSPICIOUSNESS_DIR
    path = _epoch_file(base, combo_key, variant, epoch)
    payload = {
        metric: {n: t.detach().cpu() for n, t in layers.items()}
        for metric, layers in suspiciousness.items()
    }
    torch.save(payload, path)
    return path


def load_suspiciousness(combo_key: str, variant: str, epoch: int,
                        base_dir: Optional[PathLike] = None) -> MetricLayerTensors:
    path = suspiciousness_path(combo_key, variant, epoch, base_dir)
    if not path.exists():
        raise FileNotFoundError(f"No suspiciousness file at {path}")
    return torch.load(path, weights_only=True)


# --- correlation (one file per combo+variant) ----------------------------------
# Stored nested: {susp_metric: {corr_method: {layer: tensor}}}.

def correlation_path(combo_key: str, variant: str,
                     base_dir: Optional[PathLike] = None) -> Path:
    base = Path(base_dir) if base_dir is not None else CORRELATION_DIR
    return base / combo_key / variant / "correlation.pt"


def _to_cpu_tree(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_tree(v) for k, v in obj.items()}
    raise TypeError(f"Unexpected node type in correlation tree: {type(obj)}")


def save_correlation(correlation: dict, combo_key: str, variant: str,
                     base_dir: Optional[PathLike] = None) -> Path:
    base = Path(base_dir) if base_dir is not None else CORRELATION_DIR
    path = ensure_dir(base / combo_key / variant) / "correlation.pt"
    torch.save(_to_cpu_tree(correlation), path)
    logger.info("Saved correlation -> %s", path)
    return path


def load_correlation(combo_key: str, variant: str,
                     base_dir: Optional[PathLike] = None) -> dict:
    path = correlation_path(combo_key, variant, base_dir)
    if not path.exists():
        raise FileNotFoundError(f"No correlation file at {path}")
    return torch.load(path, weights_only=True)
