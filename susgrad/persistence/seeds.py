"""Per-instance manifests for the MAIN (ensemble) pipeline.

Every ensemble instance is a *fresh random initialisation*, so "which seed made
which instance" is the only thing that makes a single instance reproducible
after the fact. Alongside the seed, each entry carries that instance's held-out
**accuracy and loss at every captured epoch**, which is what tells you whether a
correlation was measured on an untrained, a half-trained or a converged model.
The training script writes one manifest per combo, separate from the tensor
dumps::

    outputs/ensemble/seeds/<combo>.json    machine-readable (this module)
    outputs/ensemble/seeds/seeds.md        the same tables, human-readable
    outputs/ensemble/seeds/accuracy.csv    one row per (combo, instance, epoch)

The seed rule itself is trivial (instance *i* uses ``set_seed(base_seed + i)``),
but recording it explicitly means a manifest stays valid even if the rule ever
changes, and it survives partial re-runs where only some instances were
retrained.
"""

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from susgrad.utils.paths import ENSEMBLE_SEEDS_DIR, ensure_dir

PathLike = Union[str, Path]

# The manifest keys that describe the run as a whole (everything except the
# per-instance ``entries`` list).
MANIFEST_FIELDS = (
    "combo", "label", "base_seed", "seed_rule", "data_seed",
    "instances", "epochs_trained", "capture_epochs", "written",
)


def seed_manifest_path(combo_key: str, base_dir: Optional[PathLike] = None) -> Path:
    base = Path(base_dir) if base_dir is not None else ENSEMBLE_SEEDS_DIR
    return base / f"{combo_key}.json"


def build_seed_manifest(
    *,
    combo_key: str,
    label: str,
    base_seed: int,
    data_seed: int,
    epochs_trained: int,
    capture_epochs: Sequence[int],
    seeds: Sequence[dict],
    metrics: Sequence[str] = (),
) -> dict:
    """Assemble a manifest dict.

    Args:
        seeds: one dict per instance, each with at least ``index``, ``instance``
            and ``seed``; ``trained_this_run``, ``accuracy`` and ``loss`` (both
            ``{epoch: value}``) are kept when present.
        metrics: the suspiciousness metrics captured for this combo.
    """
    return {
        "combo": combo_key,
        "label": label,
        "base_seed": int(base_seed),
        "seed_rule": "instance i is initialised with set_seed(base_seed + i)",
        "data_seed": int(data_seed),
        "instances": len(seeds),
        "epochs_trained": int(epochs_trained),
        "capture_epochs": [int(e) for e in capture_epochs],
        "metrics": list(metrics),
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entries": [dict(s) for s in seeds],
    }


def accuracy_by_epoch(manifest: dict) -> Dict[int, List[float]]:
    """``{epoch: [accuracy per instance]}`` gathered from a manifest's entries."""
    out: Dict[int, List[float]] = {int(e): [] for e in manifest.get("capture_epochs", [])}
    for entry in manifest.get("entries", []):
        for epoch, value in (entry.get("accuracy") or {}).items():
            if value is not None:
                out.setdefault(int(epoch), []).append(float(value))
    return {e: out[e] for e in sorted(out)}


def accuracy_summary(manifest: dict) -> List[dict]:
    """Per-epoch mean/std/min/max held-out accuracy across a combo's instances."""
    summary = []
    for epoch, values in accuracy_by_epoch(manifest).items():
        if not values:
            continue
        summary.append({
            "epoch": epoch,
            "n": len(values),
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        })
    return summary


def write_accuracy_csv(manifests: List[dict], base_dir: Optional[PathLike] = None,
                       filename: str = "accuracy.csv") -> Path:
    """One row per (combo, instance, epoch): seed, accuracy and loss.

    A flat CSV is the format that goes straight into a stats tool or a LaTeX
    table -- the JSON manifests keep the same numbers nested per instance.
    """
    base = Path(base_dir) if base_dir is not None else ENSEMBLE_SEEDS_DIR
    path = ensure_dir(base) / filename
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["combo", "instance", "seed", "epoch", "accuracy", "loss"])
        for manifest in manifests:
            for entry in manifest.get("entries", []):
                acc = entry.get("accuracy") or {}
                loss = entry.get("loss") or {}
                for epoch in sorted(acc, key=int):
                    writer.writerow([
                        manifest["combo"], entry["instance"], entry["seed"], int(epoch),
                        f"{float(acc[epoch]):.4f}",
                        "" if loss.get(epoch) is None else f"{float(loss[epoch]):.6f}",
                    ])
    return path


def save_seed_manifest(manifest: dict, combo_key: str,
                       base_dir: Optional[PathLike] = None) -> Path:
    base = Path(base_dir) if base_dir is not None else ENSEMBLE_SEEDS_DIR
    path = ensure_dir(base) / f"{combo_key}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def load_seed_manifest(combo_key: str, base_dir: Optional[PathLike] = None) -> dict:
    path = seed_manifest_path(combo_key, base_dir)
    if not path.exists():
        raise FileNotFoundError(f"No seed manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_seed_manifests(base_dir: Optional[PathLike] = None) -> List[dict]:
    """Every manifest stored under *base_dir*, sorted by combo key.

    Lets a partial run (``--no-lenet-mnist``, say) still regenerate a complete
    ``seeds.md`` instead of dropping the combos it didn't touch.
    """
    base = Path(base_dir) if base_dir is not None else ENSEMBLE_SEEDS_DIR
    if not base.is_dir():
        return []
    out = []
    for path in sorted(base.glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(out, key=lambda m: m.get("combo", ""))


def seed_manifest_markdown(manifest: dict) -> str:
    """Render one manifest as a markdown section (header fields + instance table)."""
    head = [
        f"### {manifest.get('label', manifest['combo'])} (`{manifest['combo']}`)",
        "",
        f"- **Base seed:** {manifest['base_seed']}",
        f"- **Seed rule:** {manifest['seed_rule']}",
        f"- **Dataset seed (shared by all instances):** {manifest['data_seed']}",
        f"- **Instances:** {manifest['instances']}",
        f"- **Epochs trained:** {manifest['epochs_trained']}",
        f"- **Captured at epochs:** {', '.join(str(e) for e in manifest['capture_epochs'])}",
        f"- **Metrics:** {', '.join(manifest.get('metrics', [])) or 'n/a'}",
        f"- **Written:** {manifest['written']}",
        "",
    ]

    summary = accuracy_summary(manifest)
    if summary:
        head += [
            "Held-out accuracy across the population (%):",
            "",
            "| epoch | instances | mean | std | min | max |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ] + [
            f"| {s['epoch']} | {s['n']} | {s['mean']:.2f} | {s['std']:.2f} | "
            f"{s['min']:.2f} | {s['max']:.2f} |"
            for s in summary
        ] + [""]

    epochs = [int(e) for e in manifest.get("capture_epochs", [])]
    head += [
        "| instance | index | seed | trained this run |"
        + "".join(f" acc @ e{e} |" for e in epochs),
        "| --- | ---: | ---: | :---: |" + "".join(" ---: |" for _ in epochs),
    ]

    rows = []
    for entry in manifest.get("entries", []):
        # Keys are strings by construction (and always are after a JSON round-trip).
        acc = {str(k): v for k, v in (entry.get("accuracy") or {}).items()}
        cells = "".join(
            f" {float(acc[str(e)]):.2f} |" if acc.get(str(e)) is not None else " — |"
            for e in epochs
        )
        rows.append(
            f"| `{entry['instance']}` | {entry['index']} | {entry['seed']} | "
            f"{'yes' if entry.get('trained_this_run') else 'no (already captured)'} |" + cells
        )
    return "\n".join(head + rows) + "\n"


def write_seeds_markdown(manifests: List[Dict], base_dir: Optional[PathLike] = None,
                         filename: str = "seeds.md") -> Path:
    """Write every manifest into one human-readable markdown file."""
    base = Path(base_dir) if base_dir is not None else ENSEMBLE_SEEDS_DIR
    path = ensure_dir(base) / filename
    body = [
        "# Ensemble instance seeds",
        "",
        "One section per model/dataset combo. Re-run "
        "`scripts/train_ensemble.py --seed <base>` with the same base seed to "
        "reproduce exactly these instances.",
        "",
    ]
    body += [seed_manifest_markdown(m) for m in manifests]
    path.write_text("\n".join(body), encoding="utf-8")
    return path
