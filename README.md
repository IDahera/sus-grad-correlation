# sus-grad-correlation

Do the neurons that spectrum-based fault localisation (SBFL) calls **suspicious**
coincide with the neurons that carry a **large gradient**? This repository holds
the two experiments that address that question, the code that produces their
figures and numbers, and a container image that reproduces both from scratch.

The SBFL primitives are ported from `static-sbfl-for-nn`. Device selection
prefers **MPS → CUDA → CPU**, so on Apple Silicon everything trains on the Metal
GPU with no extra flags (native runs only — see [Docker](#reproducing-in-docker)).

## Experiments and their results

Two experiments, asking the same question along two different axes.

| | **Experiment 1** | **Experiment 2** |
| --- | --- | --- |
| **Question** | At a fixed point in training, are suspicious neurons the high-gradient ones? | How do a single model's suspiciousness and gradient values *change* while it trains? |
| **What varies** | the random initialisation (100 independent instances per model) | the epoch (one model, 200 epochs) |
| **Correlation axis** | across instances, separately per epoch snapshot | — (trajectories, not a correlation) |
| **Models** | MLP and LeNet-5, each on MNIST and Fashion-MNIST (4 pairs) | the same 4 pairs, one seeded instance each |
| **Measured at** | epochs 0 (untrained), 1 and 10 | epoch 0 and after **every** epoch, 0…200 |
| **Primary result** | correlation heatmaps + per-layer statistics | per-neuron trajectory plots + value log |
| **Run it** | `make pipeline-ensemble && make evaluate-ensemble && make figures-ensemble` | `make pipeline-trajectory` |
| **Details** | [below](#experiment-1--suspiciousness-vs-gradient-across-random-initialisations) | [below](#experiment-2--how-suspiciousness-evolves-during-training) |

### Which file backs which claim

**Experiment 1 — the correlation results.**

| Artefact | Path | What it is |
| --- | --- | --- |
| **Correlation heatmaps** (primary) | `outputs/ensemble/figures/<combo>/correlation/*.{png,pdf}` | Per-neuron `corr(susp_metric, gradient)` across the 100 instances, one file per epoch plus a composed epoch row. **These are the paper's result figures.** |
| **Per-layer statistics** (primary) | `outputs/ensemble/correlation/<combo>/correlation_report.md` | Pooled and per-layer `mean \|r\|`, signed `mean r`, median, `% \|r\|>0.5`, `% r=0`, per epoch — the numbers to quote in the text, plus ASCII renderings of the same heatmaps. |
| Raw correlation tensors | `outputs/ensemble/correlation/<combo>/epoch<EE>/correlation.pt` | `{metric: {method: {layer: tensor}}}`, for re-analysis. |
| Population provenance | `outputs/ensemble/correlation/<combo>/epoch<EE>/population.json` | Exactly which instances went into that correlation. |
| Interactive report | `outputs/visualizations/ensemble_report.html` | The same heatmaps, epochs side by side, with live layer/metric/method dropdowns. Exploration, not a paper figure. |
| **Value ranges** | `outputs/ensemble/value_stats.csv` | Min / median / mean / max / zero-share per layer, metric and epoch — the table that makes ochiai (0…1), tarantula (~0.5) and D\* (~4·10⁸) comparable. |
| Model quality | `outputs/ensemble/seeds/accuracy.csv`, `seeds.md` | Held-out accuracy and loss of every instance at every measured epoch — the context a correlation has to be read against. |
| Reproducibility | `outputs/ensemble/seeds/<combo>.json` | The seed behind every single instance. |
| Illustrative samples | `outputs/ensemble/figures/<combo>/{suspiciousness,gradient}/` | Suspiciousness and gradient maps of **one randomly chosen, logged instance** — what a single member of the population looks like. Supporting evidence, not the result. |

**Experiment 2 — the evolution results.**

| Artefact | Path | What it is |
| --- | --- | --- |
| **Trajectory plots** (primary) | `outputs/trajectory/figures/<combo>__<layer>-n<idx>__trajectory.{png,pdf}` | One panel per suspiciousness metric, the gradient as a dashed reference, over all 201 snapshots. **These are the paper's result figures.** |
| **Overview figure** (primary) | `outputs/trajectory/figures/all-combos__neuron-trajectories.{png,pdf}` | All four models × all three metrics in one grid. |
| **Value log** (primary) | `outputs/trajectory/neuron_trajectories.md` / `.csv` | Which neuron was picked (layer, flat index, `(channel, y, x)`) and its exact value at every epoch, alongside the model's accuracy. |
| Raw per-epoch tensors | `outputs/trajectory/{gradients,suspiciousness}/<combo>/seed042/epoch_<NN>.pt` | Every neuron's values at every epoch, for re-analysis or a different neuron. |
| Training curve | `outputs/trajectory/<combo>__training.csv` | Accuracy, test loss and train loss per epoch. |
| **Value ranges** | `outputs/trajectory/value_stats.csv` | The same per-layer/metric summary, per epoch, for the seeded run. |

Every run of every script also writes a timestamped markdown log to
`outputs/logs/`, recording the parameters it used and what it produced.

## Install

Python **3.11+** (developed on 3.14; the Docker images and CI use 3.12). Install
PyTorch first — the right build for your machine — then the package:

```bash
pip install -e ".[dev]"      # or: pip install -r requirements.txt
```

Prefer not to install anything? See [Reproducing in Docker](#reproducing-in-docker).

## Experiment 1 — suspiciousness vs. gradient across random initialisations

*Implemented by the ensemble pipeline. Produces the correlation heatmaps.*

Many **freshly, randomly-initialised instances** of the same architecture, and
the correlation is measured **across that population**, separately per epoch
snapshot: *do neurons that are suspicious in one randomly-initialised model also
tend to be the ones with high gradient, at a given point in training?*

Correlating across initialisations (rather than across epochs, as the secondary
pipeline does) is what makes the result a statement about *neurons* rather than
about one particular training run — and why the population is 100 per pair.

**Base case:** 100 instances per combo (MLP and LeNet, on MNIST and
Fashion-MNIST), trained **10 epochs**, with suspiciousness (**ochiai, tarantula,
dstar**) + gradient measured on the held-out split at epochs **0 (before any
training), 1 and 10**. Each instance is trained **once, straight through** — the
snapshots are taken in between, so nothing is trained twice. Every instance's
**held-out accuracy and loss** are recorded at each of those epochs.

```bash
make train-ensemble           # E1. 100 instances/combo, captured at epochs 0,1,10
make correlate-ensemble       # E2. correlate ACROSS INSTANCES per epoch + text report
make evaluate-ensemble        # E3. interactive HTML, epochs side by side
make figures-ensemble         # E4. PNG/PDF heatmap files for LaTeX (+ figures.tex)

make pipeline-ensemble        # runs E1-E2 (then evaluate-/figures-ensemble)
make overview-ensemble        # data-free HTML overview of this pipeline
make clean-ensemble           # delete only outputs/ensemble (asks first)
```

Override the base case with `ENSEMBLE_INSTANCES=`, `ENSEMBLE_EPOCHS=`,
`ENSEMBLE_CAPTURE_EPOCHS=` (shared by all four steps — which epochs to capture,
correlate, display and export), `ENSEMBLE_METRICS=` and `ENSEMBLE_OVERWRITE=1`:

```bash
make pipeline-ensemble ENSEMBLE_INSTANCES=50 ENSEMBLE_EPOCHS=20 ENSEMBLE_CAPTURE_EPOCHS=0,1,5,20
```

Outputs:

```
outputs/ensemble/gradients/<combo>/<inst>/epoch_<NN>.pt       per-instance gradients
outputs/ensemble/suspiciousness/<combo>/<inst>/epoch_<NN>.pt  per-instance suspiciousness
outputs/ensemble/correlation/<combo>/epoch<EE>/correlation.pt correlated across instances
outputs/ensemble/correlation/<combo>/epoch<EE>/population.json which instances went into it
outputs/ensemble/correlation/<combo>/correlation_report.md     TEXT report: stats + ASCII heatmaps
outputs/ensemble/seeds/<combo>.json  /  seeds.md               seed + per-epoch accuracy per instance
outputs/ensemble/seeds/accuracy.csv                            flat: combo, instance, epoch, accuracy, loss
outputs/ensemble/figures/<combo>/{correlation,suspiciousness,gradient}/*.{png,pdf}
outputs/ensemble/figures/figures.tex  /  manifest.csv          LaTeX macros + a searchable index
outputs/ensemble/value_stats.csv                               min/median/mean/max per layer x metric
outputs/paper/                                                 the curated set (make paper-figures)
outputs/visualizations/ensemble_report.html                    interactive report
```

The **text report** carries the same information as the HTML, in a form you can
diff, grep or paste into a document: pooled and per-layer `mean |r|`, signed
`mean r`, median, `% |r|>0.5` and `% r=0` per epoch, then one ASCII heatmap per
layer with the epochs printed one under the other. The **HTML report** puts
epochs 0, 1 and 10 side by side in one row per quantity (suspiciousness,
gradient, correlation), sharing one colour scale per row so the panels are
genuinely comparable; layer / channel / metric / correlation-method stay as live
dropdowns.

Instance *i* is initialised with `set_seed(--seed + i)`, and every seed is
written to `outputs/ensemble/seeds/` — so any single instance can be re-created
later. Correlation files record the exact instance population they were computed
over, so a re-run with a different population recomputes instead of silently
mixing the two.

### Finding the right file (without drowning in them)

A full export is ~640 figures. Three levers, in the order you should reach for
them:

**1. `make paper-figures` — the curated set.** Writes a small, *flat* directory
(`outputs/paper/`, ~34 files) holding exactly the figures a write-up needs: one
model with all three metrics, one metric across all four models, one dense and
one conv layer, and the suspiciousness/gradient pair of a single layer. The
selection lives in the Makefile, not in the script — read it, change the four
lines, re-run.

```bash
make paper-figures
```

**2. Filters, when you want something else.** Every axis is a flag, and
`--flat` drops the subfolder nesting:

```bash
python scripts/figures_ensemble.py --flat --out-dir outputs/paper --metrics ochiai --methods spearman --layers conv1,fc1 --no-samples --no-mlp-mnist --no-mlp-fmnist
```

**3. `manifest.csv` — search the full export.** One row per figure with combo,
layer, channel, kind, metric, method, instance, epochs and path, so
"the tarantula/spearman map of fc1" is a `grep`, not a directory walk.

### Value ranges — `make value-stats`

Ochiai is bounded to [0, 1], tarantula sits near 0.5, D\* reaches ~4·10⁸. Two
CSVs make those scales comparable and quotable, one row per **model × layer ×
metric × epoch** (plus an `epoch = all` row pooling every epoch):

```
outputs/ensemble/value_stats.csv     experiment 1, pooled over all 100 instances
outputs/trajectory/value_stats.csv   experiment 2, the single seeded run
```

Columns: `n, min, p05, median, mean, p95, max, std, frac_zero`. The gradient is
included as its own `kind`, and `frac_zero` (the share of neurons that never
activate) is reported separately — it grows from ~3 % at epoch 0 to ~24 % at
epoch 10 in some layers, which changes how the other columns should be read.
Reads the existing dumps, so it is cheap to re-run and needs no retraining.

### Figures for LaTeX

`make figures-ensemble` writes every heatmap as a standalone **PNG and PDF** in a
light print style, each self-labelled with its epoch, value range, layer, metric
and method. Two shapes of the same data:

* **one panel per epoch** — the route to use in the thesis: LaTeX controls the
  layout, each panel gets its own caption and `\ref`, and the fonts match the
  document;
* **the captured epochs side by side in one file** (`…__epochs-0-1-10`) on a
  shared colour scale — convenient for drafts and slides, but its labels are
  baked in at a fixed size and will not match your document's typography.

Filenames sort into the order you go looking for things:

```
<combo>__<layer>[__ch<NN>]__<kind>[__<metric>][__<method>][__<inst>]__epoch(s)-…
lenet-mnist__conv1__ch00__corr__ochiai__pearson__epochs-0-1-10.pdf
mlp-mnist__net-1__susp__dstar__inst007__epoch-00.pdf
```

`figures.tex` ships two ready macros plus a commented index of every stem:

```latex
\input{figures.tex}
\susgradepochs{STEM-epoch-00}{STEM-epoch-01}{STEM-epoch-10}{caption}{fig:label}
\susgradfig{STEM-epoch-10}{caption}{fig:label}
```

The suspiciousness/gradient figures are for **one randomly chosen (and logged)
instance** — an illustrative sample, not the result; the correlation figures are
the result, aggregated over the whole population.

## Experiment 2 — how suspiciousness evolves during training

*Implemented by the trajectory scripts. Produces the per-neuron value curves.*

The opposite question to experiment 1: not "across many initialisations at a
fixed epoch", but **how do a single model's suspiciousness and gradient values
move over a long training run?** One instance of each of the 4 combos, all from
`--seed 42`, trained **200 epochs**, captured before the first epoch and after
**every** epoch. Then one neuron per model is picked at random and its curves are
plotted.

```bash
make train-trajectory         # T1. 4 models × 200 epochs, capture every epoch
make plot-trajectory          # T2. pick a neuron per model, plot + log its values

make pipeline-trajectory      # both
make clean-trajectory         # delete only outputs/trajectory (asks first)
```

Captured values are **never regenerated**: a combo whose epochs are all on disk
is skipped unless you pass `TRAJ_OVERWRITE=1`. Tune with `TRAJ_EPOCHS=`,
`TRAJ_SEED=` (the model seed, which is also the run folder) and
`TRAJ_NEURON_SEED=` (which neuron gets picked).

```
outputs/trajectory/gradients/<combo>/seed042/epoch_<NN>.pt      per-epoch gradients
outputs/trajectory/suspiciousness/<combo>/seed042/epoch_<NN>.pt per-epoch suspiciousness
outputs/trajectory/<combo>__training.csv                        accuracy + loss per epoch
outputs/trajectory/figures/<combo>__<layer>-n<idx>__trajectory.{png,pdf}
outputs/trajectory/figures/all-combos__neuron-trajectories.{png,pdf}
outputs/trajectory/neuron_trajectories.md / .csv                which neuron, and every value
outputs/trajectory/value_stats.csv                              min/median/mean/max per layer x metric
```

The neuron is drawn **uniformly over all of a model's neurons** (so a big layer
supplies proportionally more picks), reproducibly from `--neuron-seed`. Pin one
instead with `--neuron mlp_mnist=net.1:57` (or by coordinates,
`--neuron lenet_mnist=conv1:1,15,22`). A random draw can land on a dead unit
whose curves are flat at zero — that is recorded as such in the log; pass
`--require-active` to redraw until the neuron still fires at the last epoch
(a biased pick, hence opt-in).

Each metric gets **its own panel and y-axis**, with the gradient repeated as a
dashed line on the right axis: ochiai is 0…1, D\* runs into the thousands and
mean |gradient| is ~1e-5, so a shared axis would flatten two of the three.

## Reproducing in Docker

No local Python, PyTorch or dataset download needed. Two variants, one
Dockerfile — they differ only in which PyTorch wheel index is used:

| Variant | Hardware | Platforms |
| --- | --- | --- |
| **cpu** | CPU only | Linux, Windows (Docker Desktop / WSL2), macOS on Intel **and** Apple Silicon |
| **gpu** | NVIDIA CUDA | Linux and Windows/WSL2 with an NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |

### With docker compose (recommended)

```bash
docker compose run --rm cpu experiment1
docker compose run --rm cpu experiment2
```

```bash
docker compose run --rm gpu experiment1
docker compose run --rm gpu experiment2
```

### With plain docker run

```bash
docker build -f docker/Dockerfile -t susgrad:cpu .
```

```bash
docker run --rm -v "$PWD/outputs:/app/outputs" -v "$PWD/data:/app/data" susgrad:cpu experiment1
```

```bash
docker build -f docker/Dockerfile -t susgrad:gpu --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 .
```

```bash
docker run --rm --gpus all -v "$PWD/outputs:/app/outputs" -v "$PWD/data:/app/data" susgrad:gpu experiment2
```

On Windows PowerShell use `${PWD}` instead of `$PWD`.

The GPU variant pulls CUDA 13.0 wheels, which carry the **same torch version as
the CPU variant** — both compute identically. If the host driver is too old for
CUDA 13, swap the index for `https://download.pytorch.org/whl/cu128` (in the
build arg above, or in the `gpu` service in `docker-compose.yml`). No CUDA base
image is involved: the runtime ships inside the wheels, so the host only needs
the driver and the container toolkit.

### Outputs are kept

`outputs/`, `data/` and `trained_models/` are bind-mounted from the repository,
so **everything the experiments produce stays on the host** after the container
exits — and a second run reuses the downloaded datasets and skips work that is
already captured. Nothing is written inside the image.

The container runs as uid 1000 so those files are not root-owned. If your host
user has a different uid (`id -u`), add `--user "$(id -u):$(id -g)"` to
`docker run`, or `user: "1001:1001"` to the compose service.

### Other commands

| Command | What it does |
| --- | --- |
| `experiment1` / `experiment2` | one experiment, end to end |
| `all` | both |
| `smoke` | both at a tiny scale (~2 min) — proves the setup works without the full run |
| `test` | the unit-test suite |
| `make <target>` | any Makefile target |
| `help` | usage, including every tuning variable |

Tune a run with `-e`, exactly like the `make` variables:

```bash
docker compose run --rm -e ENSEMBLE_INSTANCES=20 -e ENSEMBLE_CAPTURE_EPOCHS=0,1,5 cpu experiment1
```

Two caveats worth knowing before you start a full run:

* **Apple Silicon**: MPS is not reachable from inside a container, so the image
  runs on CPU there. For the GPU on a Mac, run natively with `make` instead.
* **Runtime**: experiment 1 is 4 model/dataset pairs × 100 instances × 10 epochs
  — hours on CPU. It is resumable (already-captured instances are skipped), so
  interrupting and re-running is safe. Use `smoke` first to check the setup.

## Releases

Tagging a version runs the tests (Python 3.11 and 3.12), builds the CPU image to
prove the artefact still builds, then publishes a GitHub release with the sdist,
the wheel and their checksums — see
[`.github/workflows/release.yml`](.github/workflows/release.yml):

```bash
git tag -a v1.0.0 -m "Artefact for <paper>" && git push origin v1.0.0
```

The workflow can also be started manually from the Actions tab against an
existing tag.

## Secondary pipeline (exploratory — not a paper result)

Predates the two experiments and is kept for reference: it trains **one**
instance of each of 11 model/dataset pairs and correlates across the **epoch**
axis, using all seven suspiciousness metrics. Experiment 1 replaced it as the
correlation result (a per-epoch correlation over a single run confounds "this
neuron" with "this initialisation"), and experiment 2 replaced it as the
over-training view. Nothing in the paper comes from here.

Five ordered steps — each consumes the previous step's output, so **`evaluate`
comes last**. The `Makefile` is the entry point.

```bash
make train-models             # 1. train + evaluate every enabled combination
make capture-gradients   STOPS=10,50,100   # 2. per-neuron gradients, per epoch
make capture-suspiciousness STOPS=10,50,100 # 3. per-neuron suspiciousness, per epoch
make correlate                # 4. correlate susp vs gradient, per window
make evaluate CHANNEL=0       # 5. FINAL: render ONE tabbed HTML report

make pipeline                 # runs steps 1-4 (then run `make evaluate`)
make overview                 # data-free HTML overview of models + pipeline + outputs
make test                     # run the unit tests
make help                     # list all targets
```

`make overview` renders `outputs/visualizations/overview.html` — a tabbed page
with a **Pipeline** tab (model/dataset pairs, the per-epoch training loop, and
what the report visualises) and a **Datasets** tab describing each dataset: the
features (with example rows) for tabular sets, a one-per-class image sample grid
for image sets, and the ground-truth classes. Use `--no-samples` to skip the live
example rows / image grids (faster, no data download).

### Epoch windows (variants)

The capture steps train **once** up to the largest stop and store each window
`1..stop` **separately** as a variant (`e010`, `e050`, `e100`), so correlation can
be measured over different horizons and compared:

```bash
make capture-gradients   STOPS=10,25,50      # trains 50 epochs, stores 3 windows
make capture-suspiciousness STOPS=10,25,50
make correlate                                # corr per variant, per metric
```

The default training length is **50 epochs** (omit `STOPS` for a single `1..50`
window).

Re-running is cheap — a combo whose variants are already complete is **skipped**
unless you pass `OVERWRITE=1` (or `--overwrite`). `correlate` correlates every
variant it finds, and the report gets a **tab per variant**. If you omit `STOPS`,
`EPOCHS=N` is the single window `1..N`.

### Selecting combinations

All model/dataset pairs live in `susgrad/registry.py` (the single source of
truth). Every pair is a default-on CLI flag:

```bash
python scripts/train_models.py --only tabular          # just the MLPs
python scripts/train_models.py --no-lenet-mnist        # everything except MNIST
python scripts/capture_gradients.py --only image --epochs 10
```

Current pairs (11): tabular — `mlp_banknote`, `mlp_sonar`, `mlp_ionosphere`,
`mlp_moons`, `mlp_circles`; image — the dense MLP and LeNet-5 each on MNIST,
Fashion-MNIST and CIFAR-10 (`mlp_mnist`, `mlp_fmnist`, `mlp_cifar10`,
`lenet_mnist`, `lenet_fmnist`, `lenet_cifar10`). LeNet is size-adaptive, so it
handles 1×28×28 and 3×32×32 inputs. Every script covers all pairs and accepts the
disable flags.

### Outputs

```
trained_models/<combo>_e<NN>.pt              final trained models
outputs/training_logs/training_results.csv   accuracies, params, epoch timings
outputs/training_logs/training_report_*.html per-epoch loss/accuracy + summary table
outputs/gradients/<combo>/<variant>/epoch_<NN>.pt       {layer: per-neuron gradient}
outputs/suspiciousness/<combo>/<variant>/epoch_<NN>.pt  {metric: {layer: per-neuron susp}}
outputs/correlation/<combo>/<variant>/correlation.pt    {susp_metric: {method: {layer: corr}}}
outputs/visualizations/report.html             ONE tabbed report (model → variant → metric tabs)
outputs/logs/<script>_<timestamp>.md          human-readable markdown run log
```

Every run also writes a markdown run-log to `outputs/logs/` you can keep as a
reference. `make clean` deletes everything above (after a confirmation prompt);
`make clean-caches` only clears Python caches.

### What `evaluate` draws

A single self-contained HTML with a **tab per model**, a **tab per epoch-window
variant**, and a **sub-tab per suspiciousness metric** — ochiai, tarantula,
dstar, jaccard, kulczynski2, op2, gp13 (bounded ones flagged in `BOUNDED_METRICS`).
Each metric sub-tab shows, per layer:

- suspiciousness, gradient and correlation heatmaps side by side (correlation is
  *that metric* vs the gradient, computed per neuron across **all** epochs);
- a decrement-comparison plot (suspiciousness and gradient each sorted high→low);
- a **correlation distribution histogram** over the layer's neurons, annotated
  with `mean|r|`, median, `% |r|>0.5` and `% r=0` (the undefined-→-0 share).

Conv layers `(C,H,W)` show one channel at its true `H×W` resolution — the **same**
channel (`--channel`) across metrics — with the channel index and size labelled.
Dense layers use a near-square overview. The suspiciousness/gradient heatmaps are
the `--epoch` snapshot; the correlation and its histogram span all epochs.

## Layout

```
susgrad/
  registry.py               single source of truth for model/dataset combinations
  models/architectures.py   TabularMLP, MLPClassifier (dense), LeNet5
  sbfl/                      suspiciousness COMPUTATION
    spectrum.py              HitSpectrum (a_s/a_f/n_s/n_f); ochiai, tarantula, d*
    analysis.py              compute_suspiciousness over a dataset
  grads/capture.py           per-neuron gradient COMPUTATION (mean |dloss/dact|)
  correlation/               susp-vs-gradient correlation across epochs
    metrics.py               pearson_correlation, spearman_correlation (per neuron)
    analysis.py              compute_correlations (stack epochs -> per-neuron corr)
  persistence/               grouped STORE/LOAD
    models.py                save_model / load_model
    spectra.py               save/load gradients, suspiciousness, correlation
    seeds.py                 ensemble seed manifests (which seed made which instance)
  training/
    datasets.py              prepare_dataset (tensorise + normalise + validate)
    trainer.py               train, train_epochs (per-epoch), evaluate
  neurons.py                 address ONE neuron (random pick, flat index <-> coords)
  viz/
    transform.py             bounded transforms + per-channel heatmap builder
    html.py                  HTML utilities — the home for all HTML/figure builders
    render.py                susp/grad/correlation report (delegates to html.py)
    textmap.py               ASCII heatmaps + correlation statistics (text report)
    ensemble_html.py         ensemble report: epochs side by side, per row
    figures.py               standalone PNG/PDF figures for LaTeX (light print style)
  utils/                     device (MPS-first), paths, seeding, sizes
scripts/   experiment 1: train_ensemble, correlate_ensemble, evaluate_ensemble,
                         figures_ensemble, overview_ensemble
           experiment 2: train_trajectory, plot_trajectory
           secondary:    train_models, capture_gradients, capture_suspiciousness,
                         correlate, evaluate
tests/                       pytest suite
docker/
  Dockerfile                 one image, CPU or CUDA via TORCH_INDEX_URL
  entrypoint.sh              `experiment1` / `experiment2` / `smoke` / `test` / `make`
docker-compose.yml           the cpu and gpu services, with outputs/ bind-mounted
.github/workflows/
  release.yml                tests + image build, then publishes a GitHub release
Makefile                     the canonical "which command do I run?" entry point
```

See `docs/DESIGN.md` for design decisions and the recommended test coverage.
