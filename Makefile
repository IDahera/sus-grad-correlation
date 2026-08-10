# susgrad -- task runner.
#
# The Makefile is the canonical "which command do I run?" entry point, so the
# Python files themselves can keep descriptive names (no 01_/02_ prefixes).
# Run `make help` to see everything available.
#
# TWO PIPELINES:
#
#   MAIN (ensemble) PIPELINE -- many freshly, randomly-initialised instances of
#   MLP and LeNet, separately, on MNIST and Fashion-MNIST. Correlation is
#   measured ACROSS INSTANCES, separately per epoch snapshot ("do suspicious
#   neurons also have high gradient, at a given point in training, across many
#   independent random initialisations?").
#   BASE CASE: 100 instances/combo, trained 10 epochs, measured at epochs 0
#   (untrained), 1 and 10, with ochiai/tarantula/dstar -- each instance is
#   trained ONCE, straight through, and susp/grad are captured in between.
#     1. make train-ensemble       train N instances/combo, capture susp+grad at
#                                  epochs 0,1,10; log seeds + accuracy per instance
#     2. make correlate-ensemble   correlate susp vs gradient ACROSS INSTANCES,
#                                  separately per epoch snapshot; write the text
#                                  report (stats + ASCII heatmaps)
#     3. make evaluate-ensemble    render the interactive HTML report
#                                  (epochs 0/1/10 side by side, per row)
#     4. make figures-ensemble     export the heatmaps as PNG/PDF files for LaTeX
#     (`make pipeline-ensemble` runs 1-2; then evaluate-ensemble / figures-ensemble.)
#     `make overview-ensemble` renders this pipeline's data-free overview.
#
#   EXPERIMENT 2 (trajectory) -- the opposite question: ONE seeded instance per
#   combo (seed 42), trained 200 epochs, captured before epoch 1 and after EVERY
#   epoch. Then one random neuron per model is followed across those epochs.
#     1. make train-trajectory     train + capture (skips what already exists)
#     2. make plot-trajectory      pick a neuron, plot its susp/grad curves, log
#                                  the values and which neuron was picked
#     (`make pipeline-trajectory` runs both.)
#
#   SECONDARY (original) PIPELINE -- one instance per combo (all 11: 5 tabular +
#   3 image datasets x {dense MLP, LeNet}), trained once over many epochs.
#   Correlation is measured ACROSS EPOCHS, for that one instance.
#     1. make train-models            train + evaluate the networks
#     2. make capture-gradients       per-neuron gradients, per epoch
#     3. make capture-suspiciousness  per-neuron suspiciousness, per epoch
#     4. make correlate               correlate susp vs gradient across epochs
#     5. make evaluate                FINAL: render the HTML reports
#     (`make pipeline` runs 1-4; then run `make evaluate`.)
#     `make overview` renders this pipeline's data-free overview.

# Python interpreter. By default we run inside the conda env named `py14`
# (no need to `conda activate` first). Override either piece:
#   make test                       # uses conda env py14
#   make test CONDA_ENV=otherenv    # different conda env
#   make test PYTHON=python3.14     # bypass conda entirely
CONDA_ENV ?= py14
PYTHON ?= conda run --no-capture-output -n $(CONDA_ENV) python

# --- secondary (original) pipeline: convenience variables ---------------------
EPOCHS ?=
STOPS ?=
OVERWRITE ?=

# Forward EPOCHS=<n> / STOPS=5,10,20 / OVERWRITE=1 when provided. Leave unset to
# use each script's own default (STOPS defaults to 5,10,20 -> windows 1..5,
# 1..10, 1..20).
EPOCHS_ARG := $(if $(EPOCHS),--epochs $(EPOCHS),)
STOPS_ARG := $(if $(STOPS),--stops $(STOPS),)
OVERWRITE_ARG := $(if $(OVERWRITE),--overwrite,)
CAPTURE_ARGS := $(EPOCHS_ARG) $(STOPS_ARG) $(OVERWRITE_ARG)

# --- main (ensemble) pipeline: convenience variables ---------------------------
ENSEMBLE_INSTANCES ?=
ENSEMBLE_EPOCHS ?=
ENSEMBLE_CAPTURE_EPOCHS ?=
ENSEMBLE_OVERWRITE ?=

# Forward ENSEMBLE_INSTANCES=<n> / ENSEMBLE_EPOCHS=<n> /
# ENSEMBLE_CAPTURE_EPOCHS=0,1,10 / ENSEMBLE_METRICS=ochiai,dstar /
# ENSEMBLE_OVERWRITE=1 when provided. Leave unset to use each script's own
# default -- the base case: 100 instances x 10 epochs, captured at epochs
# 0,1,10, with ochiai/tarantula/dstar.
#
# ENSEMBLE_CAPTURE_EPOCHS is shared by all four steps: it is --capture-epochs
# for train-ensemble and --epochs for correlate-/evaluate-/figures-ensemble
# (which epoch snapshots to correlate, display and export).
ENSEMBLE_METRICS ?=

ENSEMBLE_INSTANCES_ARG := $(if $(ENSEMBLE_INSTANCES),--instances $(ENSEMBLE_INSTANCES),)
ENSEMBLE_EPOCHS_ARG := $(if $(ENSEMBLE_EPOCHS),--epochs $(ENSEMBLE_EPOCHS),)
ENSEMBLE_CAPTURE_ARG := $(if $(ENSEMBLE_CAPTURE_EPOCHS),--capture-epochs $(ENSEMBLE_CAPTURE_EPOCHS),)
ENSEMBLE_SELECT_ARG := $(if $(ENSEMBLE_CAPTURE_EPOCHS),--epochs $(ENSEMBLE_CAPTURE_EPOCHS),)
ENSEMBLE_METRICS_ARG := $(if $(ENSEMBLE_METRICS),--metrics $(ENSEMBLE_METRICS),)
ENSEMBLE_OVERWRITE_ARG := $(if $(ENSEMBLE_OVERWRITE),--overwrite,)
ENSEMBLE_TRAIN_ARGS := $(ENSEMBLE_INSTANCES_ARG) $(ENSEMBLE_EPOCHS_ARG) $(ENSEMBLE_CAPTURE_ARG) \
                       $(ENSEMBLE_METRICS_ARG) $(ENSEMBLE_OVERWRITE_ARG)

# --- experiment 2 (trajectory): convenience variables ---------------------------
# One seeded instance per combo, many epochs, then per-neuron trajectory plots.
TRAJ_EPOCHS ?=
TRAJ_SEED ?=
TRAJ_NEURON_SEED ?=
TRAJ_OVERWRITE ?=

TRAJ_EPOCHS_ARG := $(if $(TRAJ_EPOCHS),--epochs $(TRAJ_EPOCHS),)
TRAJ_SEED_ARG := $(if $(TRAJ_SEED),--seed $(TRAJ_SEED),)
TRAJ_NEURON_SEED_ARG := $(if $(TRAJ_NEURON_SEED),--neuron-seed $(TRAJ_NEURON_SEED),)
TRAJ_OVERWRITE_ARG := $(if $(TRAJ_OVERWRITE),--overwrite,)

.DEFAULT_GOAL := help

.PHONY: help install overview pipeline train-models train-binary-models train-image-models \
        capture-gradients capture-suspiciousness correlate evaluate \
        overview-ensemble pipeline-ensemble train-ensemble correlate-ensemble evaluate-ensemble \
        figures-ensemble value-stats paper-figures train-trajectory plot-trajectory pipeline-trajectory \
        test clean clean-caches clean-data clean-ensemble clean-trajectory

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

install: ## Install the package (editable) and dependencies
	$(PYTHON) -m pip install -e ".[dev]"

# =============================================================================
# MAIN (ensemble) PIPELINE -- MLP & LeNet x MNIST & Fashion-MNIST, many instances
# =============================================================================

train-ensemble: ## E1. Train N instances/combo, capture at 0,1,10 (ENSEMBLE_INSTANCES=, ENSEMBLE_EPOCHS=, ENSEMBLE_CAPTURE_EPOCHS=, ENSEMBLE_OVERWRITE=1)
	$(PYTHON) scripts/train_ensemble.py $(ENSEMBLE_TRAIN_ARGS)

correlate-ensemble: ## E2. Correlate ACROSS INSTANCES per epoch + write the text report (ENSEMBLE_CAPTURE_EPOCHS=, ENSEMBLE_OVERWRITE=1)
	$(PYTHON) scripts/correlate_ensemble.py $(ENSEMBLE_SELECT_ARG) $(ENSEMBLE_OVERWRITE_ARG)

evaluate-ensemble: ## E3. Render the ensemble HTML report (epochs side by side)
	$(PYTHON) scripts/evaluate_ensemble.py $(ENSEMBLE_SELECT_ARG)

figures-ensemble: ## E4. Export the heatmaps as PNG/PDF files for LaTeX (+ figures.tex, manifest.csv)
	$(PYTHON) scripts/figures_ensemble.py $(ENSEMBLE_SELECT_ARG)

value-stats: ## Value ranges (min/median/mean/max) per model x layer x metric x epoch -> CSV
	$(PYTHON) scripts/value_stats.py

# The curated set for the write-up. Everything below is just figures_ensemble.py
# with filters -- the script stays simple, the SELECTION lives here, in one place
# you can read and change. Four blocks, matching the four figures a paper needs:
#   1. one model, all three metrics          (metric comparison)
#   2. all four models, one metric           (does it generalise?)
#   3. one dense + one conv layer            (layer-type comparison)
#   4. susp + gradient of one layer          (what is being correlated)
PAPER_DIR ?= outputs/paper
PAPER_METHOD ?= spearman
PAPER_MODEL ?= mlp_mnist
paper-figures: ## Curate a small, flat set of paper-ready figures into outputs/paper/
	@rm -rf $(PAPER_DIR)
	$(PYTHON) scripts/figures_ensemble.py --flat --out-dir $(PAPER_DIR) \
		--metrics ochiai,tarantula,dstar --methods $(PAPER_METHOD) --layers net.1 \
		--no-samples --no-mlp-fmnist --no-lenet-mnist --no-lenet-fmnist
	$(PYTHON) scripts/figures_ensemble.py --flat --out-dir $(PAPER_DIR) \
		--metrics ochiai --methods $(PAPER_METHOD) --layers net.1,fc1 --no-samples
	$(PYTHON) scripts/figures_ensemble.py --flat --out-dir $(PAPER_DIR) \
		--metrics ochiai --methods $(PAPER_METHOD) --layers conv1,fc1 \
		--no-samples --no-mlp-mnist --no-mlp-fmnist --no-lenet-fmnist
	$(PYTHON) scripts/figures_ensemble.py --flat --out-dir $(PAPER_DIR) \
		--metrics ochiai --methods $(PAPER_METHOD) --layers net.1 --no-rows \
		--instance-seed 42 --no-mlp-fmnist --no-lenet-mnist --no-lenet-fmnist
	@echo
	@echo "Curated set in $(PAPER_DIR) ($$(ls $(PAPER_DIR)/*.pdf 2>/dev/null | wc -l | tr -d ' ') PDFs):"
	@ls $(PAPER_DIR) | grep '\.pdf$$' | sed 's/^/  /'

overview-ensemble: ## Render the data-free MAIN pipeline overview HTML (models, steps, metrics)
	$(PYTHON) scripts/overview_ensemble.py

pipeline-ensemble: train-ensemble correlate-ensemble ## Run ensemble steps E1-E2 (then evaluate-/figures-ensemble)

# =============================================================================
# EXPERIMENT 2 (trajectory) -- ONE seeded instance per combo, 200 epochs
# =============================================================================

train-trajectory: ## T1. Train 1 instance/combo over many epochs, capture every epoch (TRAJ_EPOCHS=, TRAJ_SEED=, TRAJ_OVERWRITE=1)
	$(PYTHON) scripts/train_trajectory.py $(TRAJ_EPOCHS_ARG) $(TRAJ_SEED_ARG) $(TRAJ_OVERWRITE_ARG)

plot-trajectory: ## T2. Pick a random neuron per model, plot susp/grad over epochs (TRAJ_SEED=, TRAJ_NEURON_SEED=)
	$(PYTHON) scripts/plot_trajectory.py $(TRAJ_SEED_ARG) $(TRAJ_NEURON_SEED_ARG)

pipeline-trajectory: train-trajectory plot-trajectory ## Run experiment 2 end to end (T1 + T2)

# =============================================================================
# SECONDARY (original) PIPELINE -- 11 combos, one instance each, many epochs
# =============================================================================

train-models: ## 1. Train + evaluate every enabled model/dataset combination
	$(PYTHON) scripts/train_models.py $(EPOCHS_ARG)

train-binary-models: ## 1a. Train only the tabular (binary-classification) models
	$(PYTHON) scripts/train_models.py --only tabular $(EPOCHS_ARG)

train-image-models: ## 1b. Train only the image models (LeNet + dense MLP on MNIST)
	$(PYTHON) scripts/train_models.py --only image $(EPOCHS_ARG)

capture-gradients: ## 2. Dump gradients per epoch (EPOCHS=, STOPS=5,10,20, OVERWRITE=1)
	$(PYTHON) scripts/capture_gradients.py $(CAPTURE_ARGS)

capture-suspiciousness: ## 3. Dump suspiciousness per epoch (EPOCHS=, STOPS=, OVERWRITE=1)
	$(PYTHON) scripts/capture_suspiciousness.py $(CAPTURE_ARGS)

correlate: ## 4. Correlate susp vs gradient, per variant window (OVERWRITE=1)
	$(PYTHON) scripts/correlate.py $(OVERWRITE_ARG)

# Usage: make evaluate CHANNEL=0   (variant + metric tabs in one report)
CHANNEL ?= 0
evaluate: ## 5. FINAL STEP — render ONE tabbed HTML report (run after all the above)
	$(PYTHON) scripts/evaluate.py --channel $(CHANNEL)

overview: ## Render the data-free SECONDARY pipeline overview HTML (models, loop, outputs)
	$(PYTHON) scripts/overview.py

pipeline: train-models capture-gradients capture-suspiciousness correlate ## Run steps 1-4 (then `make evaluate`)

# =============================================================================

test: ## Run the unit-test suite
	$(PYTHON) -m pytest

clean-caches: ## Remove only Python/pytest caches (non-destructive)
	rm -rf .pytest_cache **/__pycache__

clean: ## Delete pipeline outputs + models, BOTH pipelines (asks for confirmation; keeps data/ dataset cache)
	@echo "This will permanently delete everything both pipelines produced,"
	@echo "EXCEPT the downloaded datasets (data/ is kept so they aren't re-downloaded):"
	@echo "  - trained_models/*.pt   (saved models)"
	@echo "  - outputs/              (secondary pipeline: gradients, suspiciousness,"
	@echo "                           correlation, visualizations, logs, training_logs)"
	@echo "  - outputs/ensemble/     (main pipeline: per-instance gradients,"
	@echo "                           suspiciousness and correlation)"
	@echo "  (data/ is NOT deleted -- run 'make clean-data' if you really want that)"
	@printf "Are you sure? [y/N] "; read ans; \
	  if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
	    rm -rf outputs trained_models/*.pt .pytest_cache **/__pycache__; \
	    echo "Deleted. (kept trained_models/.gitkeep and data/)"; \
	  else \
	    echo "Aborted; nothing deleted."; \
	  fi

clean-ensemble: ## Delete ONLY the main (ensemble) pipeline's outputs (gradients, susp, correlation, seeds)
	@echo "This will permanently delete outputs/ensemble/ (per-instance gradients,"
	@echo "suspiciousness, correlation + text reports, and the seed manifests)."
	@echo "Trained models, the secondary pipeline's outputs and data/ are NOT touched."
	@printf "Are you sure? [y/N] "; read ans; \
	  if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
	    rm -rf outputs/ensemble; \
	    echo "Deleted outputs/ensemble."; \
	  else \
	    echo "Aborted; nothing deleted."; \
	  fi

clean-trajectory: ## Delete ONLY experiment 2's outputs (per-epoch dumps, figures, logs)
	@echo "This will permanently delete outputs/trajectory/ (experiment 2: per-epoch"
	@echo "gradients/suspiciousness for the seeded instances, figures and value logs)."
	@printf "Are you sure? [y/N] "; read ans; \
	  if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
	    rm -rf outputs/trajectory; \
	    echo "Deleted outputs/trajectory."; \
	  else \
	    echo "Aborted; nothing deleted."; \
	  fi

clean-data: ## Delete downloaded datasets too (MNIST, OpenML cache) -- asks for confirmation
	@echo "This will permanently delete data/ (downloaded datasets: MNIST, OpenML cache)."
	@printf "Are you sure? [y/N] "; read ans; \
	  if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
	    rm -rf data; \
	    echo "Deleted data/."; \
	  else \
	    echo "Aborted; nothing deleted."; \
	  fi
