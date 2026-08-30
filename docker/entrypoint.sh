#!/usr/bin/env bash
# Entry point for the susgrad reproduction image.
#
# Turns the two experiments into single words, so a reviewer never has to know
# the internal step order:
#
#   experiment1   suspiciousness vs. gradient ACROSS random initialisations
#                 (train -> correlate -> HTML report -> LaTeX figures)
#   experiment2   how one model's values evolve ACROSS epochs
#                 (train/capture every epoch -> per-neuron and population plots)
#   paper         curate a small, flat set of paper-ready figures
#   smoke         both experiments at a tiny scale (~2 min) -- proves the
#                 pipeline runs end to end without the multi-hour cost
#   test          the unit-test suite
#   make <t>      any Makefile target
#   <anything>    executed verbatim (e.g. `python scripts/train_ensemble.py --help`)
#
# Anything the Makefile reads from the environment (ENSEMBLE_INSTANCES,
# ENSEMBLE_EPOCHS, ENSEMBLE_CAPTURE_EPOCHS, ENSEMBLE_METRICS, TRAJ_EPOCHS,
# TRAJ_SEED, ...) can be passed with `-e VAR=value`.
set -euo pipefail

# The Makefile defaults to running Python through conda; inside the image the
# interpreter is simply `python`.
export PYTHON="${PYTHON:-python}"

usage() {
    cat <<'EOF'
susgrad -- suspiciousness / gradient correlation experiments

Usage: docker run --rm -v "$PWD/outputs:/app/outputs" -v "$PWD/data:/app/data" \
           susgrad:cpu <command> [args]

Commands:
  experiment1    Experiment 1: across 100 random initialisations, at epochs 0/1/10.
                 Writes correlation tensors + text report + HTML report + PNG/PDF
                 figures to outputs/ensemble/ and outputs/visualizations/.
  experiment2    Experiment 2: one seeded model per combo over 200 epochs, then the
                 per-neuron and whole-population value curves. Writes to
                 outputs/trajectory/.
  paper          Curate a small, flat set of paper-ready figures into outputs/paper/.
  smoke          Both experiments at a tiny scale, to verify the setup quickly.
  test           Run the unit-test suite.
  make <target>  Run any Makefile target (see `make help`).
  help           This message.

Tuning (environment variables, all optional):
  ENSEMBLE_INSTANCES=100  ENSEMBLE_EPOCHS=10  ENSEMBLE_CAPTURE_EPOCHS=0,1,10
  ENSEMBLE_METRICS=ochiai,tarantula,dstar     ENSEMBLE_OVERWRITE=1
  TRAJ_EPOCHS=200 TRAJ_SEED=42  TRAJ_NEURON_SEED=42  TRAJ_OVERWRITE=1

Examples:
  docker run --rm -v "$PWD/outputs:/app/outputs" -v "$PWD/data:/app/data" \
      -e ENSEMBLE_INSTANCES=20 susgrad:cpu experiment1
  docker run --rm --gpus all -v "$PWD/outputs:/app/outputs" -v "$PWD/data:/app/data" \
      susgrad:gpu experiment2
EOF
}

command="${1:-help}"
[ $# -gt 0 ] && shift || true

case "${command}" in
    experiment1|exp1|ensemble)
        echo "==> Experiment 1: suspiciousness vs. gradient across random initialisations"
        make train-ensemble correlate-ensemble evaluate-ensemble figures-ensemble "$@"
        make value-stats
        ;;
    experiment2|exp2|trajectory)
        echo "==> Experiment 2: how the values move across training"
        make train-trajectory plot-trajectory "$@"
        make plot-trajectory-population
        ;;
    all)
        echo "==> Both experiments"
        make train-ensemble correlate-ensemble evaluate-ensemble figures-ensemble
        make value-stats
        make train-trajectory plot-trajectory
        make plot-trajectory-population
        ;;
    smoke)
        # Deliberately tiny: enough instances for a correlation to exist at all,
        # enough epochs for a trajectory to have a shape, and a capped evaluation
        # split. Proves wiring, not results.
        echo "==> Smoke run (tiny scale -- verifies the pipeline, NOT the results)"
        python scripts/train_ensemble.py --instances 4 --epochs 2 --capture-epochs 0,1,2 \
            --max-samples 512
        python scripts/correlate_ensemble.py --epochs 0,1,2
        python scripts/evaluate_ensemble.py --epochs 0,1,2
        python scripts/figures_ensemble.py --epochs 0,1,2 --formats png --no-singles
        python scripts/train_trajectory.py --epochs 3 --max-samples 512
        python scripts/plot_trajectory.py --formats png
        python scripts/value_stats.py --only trajectory
        python scripts/plot_trajectory_population.py --formats png
        echo "==> Smoke run complete -- see outputs/"
        ;;
    paper)
        echo "==> Curating the paper figure set"
        make paper-figures "$@"
        ;;
    test)
        make test "$@"
        ;;
    make)
        make "$@"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        exec "${command}" "$@"
        ;;
esac
