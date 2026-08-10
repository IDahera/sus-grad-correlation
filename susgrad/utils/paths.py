"""Project paths and a small directory helper.

This project has TWO pipelines (see the Makefile for the full breakdown):

    * the SECONDARY (original) pipeline -- one instance per combo, trained once
      over many epochs; correlation is measured across that EPOCH axis.
    * the MAIN (ensemble) pipeline -- many freshly, randomly initialised
      instances per combo, each trained for a few epochs; correlation is
      measured across the INSTANCE axis, separately per epoch snapshot.

All experiment outputs live under ``outputs/`` in clearly named subfolders:

    trained_models/                       final trained models (<combo>_e<NN>.pt)
    outputs/gradients/<combo>/            secondary pipeline: per-epoch gradients
    outputs/suspiciousness/<combo>/       secondary pipeline: per-epoch suspiciousness
    outputs/correlation/<combo>/          secondary pipeline: per-variant correlation
    outputs/ensemble/gradients/<combo>/   main pipeline: per-instance, per-epoch gradients
    outputs/ensemble/suspiciousness/<combo>/  main pipeline: per-instance, per-epoch susp.
    outputs/ensemble/correlation/<combo>/ main pipeline: per-epoch (across instances) correlation
    outputs/ensemble/seeds/                main pipeline: per-instance seed manifests
    outputs/ensemble/figures/              main pipeline: PNG/PDF heatmaps for LaTeX
    outputs/trajectory/                    experiment 2: one seeded instance per combo,
                                           per-epoch dumps + per-neuron trajectory plots
    outputs/visualizations/               generated HTML views (both pipelines)
    outputs/training_logs/                training result CSVs
"""

from pathlib import Path

# .../susgrad/utils/paths.py -> project root is two parents up from the package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Final trained models.
DEFAULT_MODEL_DIR = PROJECT_ROOT / "trained_models"

# --- secondary (original) pipeline: per-epoch experiment artefacts -------------
GRADIENTS_DIR = OUTPUTS_DIR / "gradients"
SUSPICIOUSNESS_DIR = OUTPUTS_DIR / "suspiciousness"
CORRELATION_DIR = OUTPUTS_DIR / "correlation"

# --- main (ensemble) pipeline: per-instance, per-epoch artefacts ---------------
ENSEMBLE_DIR = OUTPUTS_DIR / "ensemble"
ENSEMBLE_GRADIENTS_DIR = ENSEMBLE_DIR / "gradients"
ENSEMBLE_SUSPICIOUSNESS_DIR = ENSEMBLE_DIR / "suspiciousness"
ENSEMBLE_CORRELATION_DIR = ENSEMBLE_DIR / "correlation"
# Which random seed produced which instance (written by train_ensemble.py).
ENSEMBLE_SEEDS_DIR = ENSEMBLE_DIR / "seeds"
# Standalone PNG/PDF heatmaps for LaTeX (written by figures_ensemble.py).
ENSEMBLE_FIGURES_DIR = ENSEMBLE_DIR / "figures"

# --- experiment 2 (trajectory): ONE instance per combo, many epochs -------------
# Per-epoch gradients/suspiciousness for a single seeded instance, plus the
# per-neuron trajectory plots built from them.
TRAJECTORY_DIR = OUTPUTS_DIR / "trajectory"
TRAJECTORY_GRADIENTS_DIR = TRAJECTORY_DIR / "gradients"
TRAJECTORY_SUSPICIOUSNESS_DIR = TRAJECTORY_DIR / "suspiciousness"
TRAJECTORY_FIGURES_DIR = TRAJECTORY_DIR / "figures"

# --- shared -------------------------------------------------------------------
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
TRAINING_LOGS_DIR = OUTPUTS_DIR / "training_logs"

# Human-readable markdown run logs (one per script invocation).
LOGS_DIR = OUTPUTS_DIR / "logs"


def ensure_dir(directory: str | Path) -> Path:
    """Create *directory* (and parents) if needed and return it as a Path."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path
