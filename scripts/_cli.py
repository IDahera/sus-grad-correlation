"""Shared CLI plumbing for the experiment scripts.

Keeps the per-script files small: logging setup, the registry-driven on/off flags
(``--<combo>`` / ``--no-<combo>``), the ``--only`` kind filter, and a helper that
builds a dataset + correctly-sized model for one combination.
"""

import logging
import sys
import time
from pathlib import Path

# Allow running the scripts directly (without installing the package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from susgrad.persistence import variant_complete, variant_name
from susgrad.registry import COMBINATIONS, ENSEMBLE_COMBO_KEYS, ensemble_combinations
from susgrad.training import prepare_dataset, train_epochs
from susgrad.utils import LOGS_DIR, ensure_dir


def setup_logging(script_name: str, verbose: bool = False):
    """Configure logging for a script run.

    Logs to the console (with timestamps) and, in addition, to a markdown run-log
    at ``outputs/logs/<script>_<timestamp>.md`` that you can keep as a reference.
    The console keeps a timestamp prefix; the markdown file gets the raw message
    so headers/bullets render cleanly.

    Returns ``(logger, logfile_path)``.
    """
    ensure_dir(LOGS_DIR)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    logfile = LOGS_DIR / f"{script_name}_{stamp}.md"

    logger = logging.getLogger("susgrad")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    logger.addHandler(console)

    md = logging.FileHandler(logfile, encoding="utf-8")
    md.setFormatter(logging.Formatter("%(message)s"))  # raw -> valid markdown
    logger.addHandler(md)

    logger.info("# %s\n", script_name)
    logger.info("_Run started %s_\n", time.strftime("%Y-%m-%d %H:%M:%S"))
    return logger, logfile


# --- formatting helpers --------------------------------------------------------
# Keep lines readable: section headers, key/value fields, and long values (paths)
# placed on their own indented line.

def section(logger: logging.Logger, title: str) -> None:
    """A markdown H2 section header (with surrounding blank lines)."""
    logger.info("\n## %s\n", title)


def field(logger: logging.Logger, label: str, value) -> None:
    """A short ``- **label:** value`` line."""
    logger.info("- **%s:** %s", label, value)


def long_field(logger: logging.Logger, label: str, value) -> None:
    """A field whose (long) value goes on its own indented line, e.g. a path."""
    logger.info("- **%s:**\n      `%s`", label, value)


def combo_options(func):
    """Add a ``--<key>/--no-<key>`` flag (default on) for every combination."""
    for combo in reversed(COMBINATIONS):
        flag = combo.key.replace("_", "-")
        func = click.option(
            f"--{flag}/--no-{flag}",
            default=True,
            help=f"Toggle {combo.label} (default on).",
        )(func)
    return func


only_option = click.option(
    "--only",
    type=click.Choice(["all", "tabular", "image"]),
    default="all",
    show_default=True,
    help="Restrict to a model kind.",
)

overwrite_option = click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Recompute and overwrite even if stored files already exist.",
)


def parse_stops(stops: str, epochs: int) -> list:
    """Parse a comma-separated stops string into a sorted, de-duplicated list.

    Each stop is the end of a window ``1..stop`` to store separately. Empty input
    defaults to a single stop at ``epochs``. The training length is ``max(stops)``.
    """
    if stops and stops.strip():
        try:
            values = sorted({int(s) for s in stops.split(",") if s.strip()})
        except ValueError:
            raise click.BadParameter("stops must be comma-separated integers, e.g. 10,50,100")
    else:
        values = [int(epochs)]
    if any(s < 1 for s in values):
        raise click.BadParameter("stops must be >= 1")
    return values


def parse_epoch_list(spec: str, *, max_epoch: int = None, allow_auto: bool = False):
    """Parse ``"0,1,10"`` into ``[0, 1, 10]`` (sorted, de-duplicated).

    Used by the MAIN (ensemble) pipeline to say *which epoch snapshots* to
    capture / correlate / render -- epoch 0 being the untrained model, so 0 is a
    legal value here (unlike the secondary pipeline's 1-based windows).

    Args:
        max_epoch: if given, every entry must be ``<= max_epoch``.
        allow_auto: accept the literal ``"auto"`` (returned as ``None``, meaning
            "work it out from whatever is on disk").
    """
    text = (spec or "").strip()
    if allow_auto and text.lower() == "auto":
        return None
    if not text:
        raise click.BadParameter("epoch list must not be empty, e.g. 0,1,10")
    try:
        values = sorted({int(s) for s in text.split(",") if s.strip()})
    except ValueError:
        raise click.BadParameter(f"epochs must be comma-separated integers, e.g. 0,1,10 (got {spec!r})")
    if not values:
        raise click.BadParameter("epoch list must not be empty, e.g. 0,1,10")
    if any(v < 0 for v in values):
        raise click.BadParameter("epochs must be >= 0 (0 = before any training)")
    if max_epoch is not None and values[-1] > max_epoch:
        raise click.BadParameter(
            f"epoch {values[-1]} exceeds --epochs {max_epoch}; raise --epochs or lower the list."
        )
    return values


def collect_enabled(kwargs: dict) -> dict:
    """Extract the ``{combo_key: bool}`` toggle map from click kwargs."""
    return {c.key: kwargs.get(c.key, True) for c in COMBINATIONS}


def ensemble_combo_options(func):
    """Add a ``--<key>/--no-<key>`` flag (default on) for each of the 4 ensemble combos.

    Same idea as :func:`combo_options`, but scoped to the main (ensemble)
    pipeline's fixed set (MLP/LeNet x MNIST/Fashion-MNIST) instead of every
    registered combination.
    """
    for combo in reversed(ensemble_combinations()):
        flag = combo.key.replace("_", "-")
        func = click.option(
            f"--{flag}/--no-{flag}",
            default=True,
            help=f"Toggle {combo.label} (default on).",
        )(func)
    return func


def collect_enabled_ensemble(kwargs: dict) -> dict:
    """Extract the ``{combo_key: bool}`` toggle map for the 4 ensemble combos."""
    return {k: kwargs.get(k, True) for k in ENSEMBLE_COMBO_KEYS}


def select_ensemble_combinations(enabled: dict) -> list:
    """Ensemble combinations (in registry order) whose flag is still on."""
    return [c for c in ensemble_combinations() if enabled.get(c.key, True)]


def build_dataset_and_model(combo, *, batch_size: int, seed: int, max_samples):
    """Prepare the dataset and a model sized to it for *combo*."""
    bundle = prepare_dataset(
        combo.dataset, batch_size=batch_size, seed=seed, max_samples=max_samples
    )
    model = combo.build_model(bundle.input_shape, bundle.num_classes)
    return bundle, model


def model_summary(model) -> str:
    import torch.nn as nn

    n_params = sum(p.numel() for p in model.parameters())
    n_layers = sum(
        1 for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))
    )
    return f"{type(model).__name__}: {n_layers} weight layers, {n_params:,} params"


def run_capture(
    log,
    *,
    combos,
    stop_list,
    base_dir,
    output_dir,
    batch_size,
    lr,
    seed,
    max_samples,
    overwrite,
    compute_fn,
    save_fn,
    summarize_fn,
    kind_label,
):
    """Shared capture loop for the gradient/suspiciousness scripts.

    Trains each combo ONCE up to ``max(stop_list)`` and, after every epoch, calls
    ``compute_fn(model, test_loader)`` and stores the result into each window whose
    stop has not yet been passed via ``save_fn(data, combo_key, variant, epoch, output_dir)``.
    Combos whose variants are already complete are skipped unless *overwrite*.
    """
    train_to = max(stop_list)
    for combo in combos:
        section(log, combo.label)
        if not overwrite and all(variant_complete(base_dir, combo.key, s) for s in stop_list):
            field(log, "Skipped", "all variants already complete (use --overwrite to redo)")
            continue

        bundle, model = build_dataset_and_model(
            combo, batch_size=batch_size, seed=seed, max_samples=max_samples
        )
        field(log, "Model", model_summary(model))
        log.info("")

        for result in train_epochs(
            model, bundle.train_loader, n_epochs=train_to, lr=lr, progress=False
        ):
            t0 = time.perf_counter()
            data = compute_fn(model, bundle.test_loader)
            compute_s = time.perf_counter() - t0

            stored = []
            for s in stop_list:
                if result.epoch <= s:
                    save_fn(data, combo.key, variant_name(s), result.epoch, output_dir)
                    stored.append(variant_name(s))
            log.info("  - epoch %2d/%d: %s in %.3fs (%s) -> %s",
                     result.epoch, train_to, kind_label, compute_s,
                     summarize_fn(data), ",".join(stored))
