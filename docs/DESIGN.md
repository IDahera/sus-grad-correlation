# Design notes

Answers to the questions raised when scoping this project, plus the rationale
behind the structure.

## File naming — avoiding `01_`, `02_` prefixes

Numeric prefixes encode *run order* into filenames, which is noisy and breaks as
soon as the order changes. Two cleaner mechanisms are used here instead, and
they compose:

1. **Descriptive, verb-first names.** `train_tabular.py` and `train_image.py`
   say what they do, not when to run them. There is no implied global order —
   each script is independently runnable.
2. **A `Makefile` as the canonical entry point.** Ordering and "which command
   do I run?" live in named targets (`make train-binary-models`,
   `make train-image-models`, `make train-all`), not in filenames. `make help`
   is self-documenting.
3. **Console entry points** (in `pyproject.toml`). After `pip install -e .` the
   scripts are runnable as `train-tabular` / `train-image` from anywhere.

So yes — a Makefile makes good sense here, and it's the recommended interface.
It also gives you one place to add future steps (e.g. `compute-suspiciousness`,
`train-classifiers`) without renaming anything.

## Reusing the suspiciousness logic

The SBFL core was **copied** into `susgrad/sbfl/spectrum.py` rather than imported
from `static-sbfl-for-nn`, keeping this repository self-contained. It preserves
the original semantics:

- `HitSpectrum` holds the four per-neuron counts (`a_s`, `a_f`, `n_s`, `n_f`):
  *active* = activation > threshold (statement executed), *success* = sample
  handled correctly (test passes).
- `compute_hit_spectrum` builds those tensors at the layer's shape.
- `get_ochiai` / `get_tarantula` derive suspiciousness from them.
- `get_activations_with_hooks` captures Linear/Conv2d outputs; `unsqueeze_tensors`
  flattens conv maps so each position is treated as a neuron.

These are tested in `tests/test_spectrum.py` and ready for a future
"compute suspiciousness" script — the current scripts only train and evaluate,
as requested.

## Training submodule contract

Each unit function does one thing and validates its inputs:

| Function | Guarantees | Raises |
|---|---|---|
| `prepare_dataset` | tensorises + normalises; optional model-compatibility check | `DimensionMismatchError` |
| `train` | checks model/dataset shapes before training; trains in place | `DimensionMismatchError` |
| `evaluate` | checks shapes + that inputs look normalised | `DimensionMismatchError`, `DataNotNormalizedError` |
| `save_model` | writes `<name>.pt`; default dir or a custom (test) dir | — |
| `load_model` | returns a usable model with identical params & dims | `FileNotFoundError` |

The matching tests assert exactly the invariants you asked for: input/output
dimension checks, that training leaves dimensions unchanged, that a saved model
is loadable and round-trips with identical parameters, and that evaluation
rejects un-normalised data. Persistence tests use pytest's `tmp_path`, which is
auto-cleaned, so nothing leaks into `trained_models/`.

### The "is it normalised?" check

`assert_normalized` is a deliberate heuristic: normalised inputs have small
magnitudes, so a batch whose peak absolute value exceeds ~50 (e.g. raw 0–255
pixels or un-scaled columns) is rejected. It catches the common real mistake
without pretending to be a rigorous statistical test. Tune `max_abs` if your
features legitimately have large scales.

## Apple Silicon / GPU

`susgrad/utils/devices.py` selects `mps → cuda → cpu`. On an M4 Pro this resolves
to `mps`, and training/eval/`torch.load` all use it by default. No code path
hardcodes CPU or CUDA.

## Other things considered / recommended next

- **`DatasetBundle`** bundles loaders with `input_shape` / `num_classes`, so a
  script can size a model from the data instead of hardcoding dimensions.
- **Custom exception types** make failures explicit and testable.
- **Deterministic seeding** (`set_seed`) across random/NumPy/torch.
- **Whole-model `torch.save`** (vs. state-dict) so loading needs no class
  re-declaration; loaded with `weights_only=False`.
- Natural future modules to add (mirroring `static-sbfl-for-nn`):
  a `compute_suspiciousness` script over saved models, a feature-extraction step
  that turns hit spectra into ML feature matrices, and a `train_classifiers`
  step using `susgrad/classifiers.py`.
- Consider pinning exact versions once the 3.14 + torch-MPS toolchain is fixed,
  and adding a CI job that runs `make test` on CPU.

---

# Step 02 decisions

## Pipeline shape

One combined trainer plus two capture scripts and a visualiser, all driven by the
same registry:

- `scripts/train_models.py` — trains/evaluates every enabled combination, logs the
  plan up front, then per-epoch durations and a final accuracy line, and appends
  `outputs/training_logs/training_results.csv`.
- `scripts/capture_gradients.py` and `scripts/capture_suspiciousness.py` — expanded
  variants that train epoch-by-epoch (`train_epochs`) and snapshot **after every
  epoch**. They store only their own tensors and time the *pure* computation
  (excluding training and storing) via `time.perf_counter()`.
- `scripts/visualize.py` — loads one epoch's dumps and renders HTML.

The classifier registry from Step 01 was **removed** — this project is only about
neural networks.

## Single source of truth for combinations

`susgrad/registry.py` defines `COMBINATIONS`. Each `Combination` carries a stable
`key` reused as the CLI flag, the model filename, and the output subfolder. Every
script selects work via `select_combinations(...)`, so adding/removing a pair is a
one-line change in exactly one place.

## Suspiciousness metrics

`susgrad/sbfl/spectrum.py` offers `ochiai`, `tarantula` and `dstar` (D*, default
exponent 3: `a_f^* / (a_s + n_f)`), exposed via the `METRICS` registry.

- Ochiai and Tarantula are bounded to **[0, 1]** (listed in `BOUNDED_METRICS`).
  Note 0 and 1 are legitimately attainable (e.g. a neuron never active on a
  success gives 0), so the tests assert the closed interval rather than strict
  inequality.
- D* is **unbounded** (≥ 0), so only non-negativity/finiteness is checked.
- All metrics pass through `nan_to_num`, and a test feeds a degenerate all-zero
  spectrum to confirm no NaN/inf escape.

A **success** is defined as a correct prediction (`argmax == label`) — the "proper
output" analogy — rather than the original repo's `target == class 0` quirk.

## Per-neuron gradient

`grads/capture.py` records the gradient of the loss w.r.t. each Linear/Conv2d
**output activation** and reduces it to `mean(|·|)` over the eval samples. This
yields one value per neuron with the **same shape** as the suspiciousness tensors,
so the two line up 1:1 for the heatmaps and the sorted comparison.

## Grouping (computation vs store/load)

- Computation: `sbfl/` (suspiciousness) and `grads/` (gradients).
- Store/load: `persistence/` with `models.py` (whole models) and `spectra.py`
  (gradient + suspiciousness tensors, one file per epoch, dimension-preserving).

## Visualisation bounds

`viz/transform.py` raises `VisualizationError` for: empty layers, non-finite
values, bounded-metric values outside [0, 1], more than `MAX_LAYERS` (64), or a
layer above the hard ceiling `MAX_NEURONS_PER_LAYER` (2,000,000, configurable via
`--max-neurons`).

Large but reasonable layers are **not** rejected: a layer wider than
`HEATMAP_MAX_CELLS` (16,384) is block-averaged into a bounded overview grid for
the *heatmap only* (labelled "downsampled ×N"). The full per-neuron vector is kept
in `LayerView.values`, so the sorted susp-vs-gradient comparison and the
alignment checks still use **every** neuron — the "datapoints don't change"
guarantee holds for the analysis. (The original 8192 cap rejected even a LeNet
`conv1` map once channels grew; downsampling fixes that without losing data.)

`align_layers` requires the gradient and suspiciousness mappings to cover the same
layers with the **same neuron counts**, and `sorted_comparison` applies one
permutation to both arrays so a neuron's (susp, grad) pair stays together — the
tests assert the value multiset is unchanged (no data points added/dropped/altered).

---

# Step 03 decisions

## Per-channel heatmaps

`build_heatmap` renders conv layers `(C, H, W)` by selecting a single channel and
drawing it at its true `H×W` resolution (spatial structure preserved), instead of
flattening all `C·H·W` neurons into an arbitrary square. The **same** `--channel`
is used for suspiciousness, gradient and correlation so they stay comparable; the
index is clamped per layer (`channel % C`) and the chosen channel + size are shown
in the heatmap caption. 2-D layers render directly; 1-D dense layers keep the
near-square overview (downsampled if huge). Earlier confusion — "6 channels in
conv1, how does it become a 2-D picture?" — was exactly the old flatten behaviour;
this replaces it.

## Decrement comparison

The comparison plot now sorts suspiciousness **and** gradient each from largest to
smallest *independently* (`sorted_descending`), so the two decrement curves can be
compared shape-to-shape. (The older `sorted_comparison`, which sorts grad by the
susp permutation, is retained for the alignment tests.)

## Across-epoch correlation

`susgrad/correlation/` correlates the per-neuron suspiciousness and gradient *time
series* across epochs. Each coefficient is its own unit function
(`pearson_correlation`, `spearman_correlation`) registered in `CORRELATIONS` and
logged by name in `scripts/correlate.py`. They operate on tensors shaped
`(epochs, *layer)` and return one value per neuron with the layer's shape, so the
result drops straight into the same heatmap/persistence machinery. Zero-variance
series (constant over epochs) yield 0 (undefined correlation). Values are clamped
to `[-1, 1]`.

Correlation is computed **separately for each suspiciousness metric** — a distinct
`corr(ochiai, grad)`, `corr(tarantula, grad)` and `corr(dstar, grad)` — and stored
nested as `{susp_metric: {corr_method: {layer: tensor}}}`. In the report, each
metric sub-tab shows *its own* correlation heatmap (e.g. the tarantula sub-tab
shows `corr(tarantula, grad)`). Persistence uses a recursive CPU-mover so the
nested tree round-trips; tests cover both the flat and nested layouts.

Tests: exact values (perfect ±1, a known Pearson 0.7850, Spearman=1 on
monotonic-nonlinear data, zero-variance→0) and store/load **dimension
preservation** mirroring the grad/susp persistence tests.

## Ochiai fix (carried over)

Ochiai's numerator was corrected from `a_s` to the failure-aligned `a_f`, restoring
the `[0, 1]` bound (see the formula tests in `test_metric_values.py`). The original
reference code's `a_s` numerator produced values > 1.

## New dense model

`MLPClassifier` (flatten + dense stack) is added as `mlp_mnist` — a simple,
all-`Linear` counterpart to LeNet on the same data. Its per-neuron values are 1-D,
so it renders as the square overview. `Combination.build_model` now takes the
per-sample `input_shape` (not just a flat `input_dim`) so image models can flatten
internally while the compatibility check still matches the dataset's sample shape.

## Naming / pipeline

`visualize` was renamed **`evaluate`** — it's the capstone step and now folds in
correlation, so "evaluate the relationship" fits. The Makefile documents the 1→5
order and `make pipeline` runs steps 1–4. All scripts cover every combination and
expose `--no-<combo>` disable flags. `make clean` deletes everything the scripts
produced (`trained_models/*.pt`, `outputs/`, `data/`) **after a confirmation
prompt**; `make clean-caches` is the non-destructive cache cleaner.

## Tests that *have* to be covered

Already implemented here:

- **Metrics**: bounded metrics ∈ [0,1]; all metrics finite (incl. degenerate zero
  spectrum); D* non-negative; hit-spectrum counts partition the samples
  (`a_s+a_f+n_s+n_f == n_samples`).
- **Gradient capture**: per-layer shapes equal the model's activation shapes;
  values finite and non-negative.
- **Persistence**: model save/load round-trips with identical params & dims;
  grad/susp dumps round-trip and keep per-layer dimensions; temp dirs auto-clean.
- **Training**: dimension checks; training preserves parameter *shapes* but
  *changes* values; evaluate rejects un-normalised data.
- **Visualisation**: bounds enforced; neuron counts equal across metrics; sorting
  is a value-preserving permutation.

Worth adding as the project grows:

- **Determinism**: same `--seed` ⇒ identical metrics/gradients (guards the RNG).
- **Activation hooks**: hook count equals the number of Linear/Conv2d layers, and
  removing hooks leaves no lingering handles (no memory leak across epochs).
- **End-to-end smoke**: a 1-epoch run of each script on `--max-samples` tiny data
  produces the expected files (cheap CI integration test).
- **Threshold sensitivity**: activation `threshold` changes the hit spectrum in
  the expected direction.
- **Registry integrity**: combination keys are unique and filesystem-safe; each
  builds a model whose `output_dim` matches its dataset's class count.

# Step 04 decisions — ensemble base case (epochs 0 / 1 / 10)

## Capture is decoupled from training

The main (ensemble) pipeline's base case is **20 instances per combo, trained 10
epochs, measured at epochs 0, 1 and 10**. Training and measuring are separate
concerns: `train_ensemble.py` trains each instance **once, straight through**
(`train_epochs` yields between epochs) and only computes suspiciousness +
gradients at the epochs in `--capture-epochs`. Nothing is trained twice, and no
snapshot is computed that nobody asked for; training also stops at the last
requested epoch, since epochs after it would produce no measurement.

`--capture-epochs` is parsed by the shared `parse_epoch_list` helper, which
accepts 0 (the untrained model — the whole point of the base case) and refuses
epochs beyond `--epochs`. The same helper backs `--epochs` on the correlate and
evaluate steps, so one list (`ENSEMBLE_CAPTURE_EPOCHS` in the Makefile) drives
capture, correlation and display.

## The population must be identical at every epoch

Correlation across instances is only meaningful if the same population is behind
every epoch snapshot. Two guards enforce that:

- `correlate_ensemble.py`/`evaluate_ensemble.py` only use instances that carry
  **all** requested epochs (a stale 5-epoch instance from an earlier run is
  logged and ignored, rather than silently truncating the epoch list);
- each stored correlation gets a `population.json` sidecar naming the exact
  instances it was computed over. A stored file is reused only if that list
  matches; otherwise it is recomputed. Without this, epoch 0 correlated over
  yesterday's 50 models could sit next to epoch 10 over today's 20.

## Seeds are recorded, not just derived

Instance *i* is initialised with `set_seed(base + i)`, but the rule alone is not
a record — it stops being true the moment it changes. `susgrad/persistence/seeds.py`
writes one JSON manifest per combo (plus a combined `seeds.md`) listing every
instance, its seed and whether this run trained it, so any single instance can be
re-created later.

## Text report alongside the HTML

`susgrad/viz/textmap.py` renders the same grids as ASCII, and
`correlate_ensemble.py` writes `correlation_report.md` per combo: pooled and
per-layer `mean |r|` / signed `mean r` / median / `% |r|>0.5` / `% r=0` per epoch,
then one heatmap per layer with the epochs printed one under the other. The
diverging ramp uses **disjoint glyph sets** for negative (`@ % ; ,`) and positive
(`: = * #`) values so the sign can never be misread, and every map carries its
legend, its true cell count and its block-averaging factor.

## HTML: epochs side by side, one shared scale per row

The ensemble report drops the epoch dropdown: each quantity (suspiciousness,
gradient, correlation) is one **row**, with one card per captured epoch. Within a
row the susp/grad colour scale is shared across epochs (correlation is already on
a fixed −1…+1 scale) — per-panel normalisation would make "epoch 0 vs epoch 10"
look identical no matter what happened. Each panel still prints its own value
range, and correlation panels carry the layer's `mean |r|`, median, `% |r|>0.5`
and `% r=0`, computed in Python from the **full** tensor (display grids may be
block-averaged).

## Spearman mid-ranks (bug fix)

`_rank_along_epochs` used ordinal ranks, so a constant series (a neuron that is
never active — very common at epoch 0) was ranked 0, 1, 2, … and came out
*perfectly correlated* with anything, instead of the documented 0. It now uses
mid-ranks (tied values share the average of their positions), which makes a
constant series constant in rank space too, so the existing zero-variance guard
catches it — matching Pearson, the module docstring, and scipy on tied data.

# Step 05 decisions — 100 instances, LaTeX figures, experiment 2

## Three metrics, not seven

The ensemble and trajectory experiments capture **ochiai, tarantula, dstar**
(`CORE_METRIC_NAMES`) — one bounded similarity coefficient, one classic ratio,
one unbounded family. All seven remain implemented and tested for the secondary
pipeline; the subset is a `--metrics` default, not a deletion, so nothing is lost
and the dumps/report/figures shrink by ~60%.

## 100 instances, and accuracy alongside every snapshot

The population went 20 → 100 per combo: a correlation across 20 samples has wide
enough confidence intervals that "0.31 vs 0.38" says nothing. Each instance's
held-out **accuracy and loss** are now recorded at every captured epoch
(`evaluate` on the same split, at the same moment as the tensors), because a
correlation is only interpretable against how well the model was actually doing.
They live in the seed manifest (nested per instance) and in a flat
`accuracy.csv`; a resumed run keeps the accuracies of instances it skips.

## Figures: both shapes, and why

`susgrad/viz/figures.py` writes PNG **and** PDF in a **light** style — the
report's dark theme is unreadable in print. Every figure is self-labelled
(epoch, value range, layer, metric, method) so it still says what it is once it
has been pulled into a document.

Both shapes are generated because they answer different needs, and the
recommendation is explicit: **single panels** for the thesis (LaTeX controls the
layout, each panel gets its own caption and `\ref`, fonts match the document),
the **pre-composed epoch row** for drafts and slides (its labels are baked in at
a fixed size and will always look foreign next to body text). `figures.tex`
ships a `\susgradepochs` subfigure macro so the single-panel route is no more
work than the composed one.

Filenames are ordered `combo → layer → channel → kind → metric → method →
epoch`, so a plain directory listing groups exactly the way you go looking for
"the ochiai/pearson map of conv1", and `manifest.csv` makes the set searchable
without `ls`.

## Experiment 2: one model, many epochs, one neuron

The mirror image of the ensemble question. `train_trajectory.py` trains ONE
seeded instance per combo (seed 42) for 200 epochs and captures at epoch 0 and
after every epoch; the ``variant`` slot holds the seed (`seed042`), so different
seeds cannot overwrite each other, and a combo whose epochs are complete is
skipped unless `--overwrite`. `plot_trajectory.py` then follows a single neuron.

Three decisions worth recording:

- **Uniform over neurons, not over layers** (`susgrad/neurons.py`): picking a
  layer first would over-represent the 10-neuron output layer. The pick is
  seeded, and the chosen neuron is logged with its flat index *and* its
  `(channel, y, x)` coordinates, since the flat index alone is unreadable.
- **One panel per metric.** Ochiai is 0…1, D* runs into the thousands, mean
  |gradient| is ~1e-5. On one axis, two of the three flatten onto the baseline;
  each metric therefore gets its own panel and left axis, with the gradient
  repeated as the dashed right-axis reference in every panel.
- **Dead units are reported, not hidden.** A uniform draw can land on a neuron
  that never fires again after epoch 1, giving flat-zero curves. That is a real
  property of the model, so the default keeps the draw and annotates the log;
  `--require-active` re-draws until the neuron still fires at the last epoch and
  says so, because that pick is biased.
