# sus-grad-correlation

Are the neurons that spectrum-based fault localisation (SBFL) calls
**suspicious** the same neurons that carry a **large gradient**? This repository
holds the two experiments that answer that question, the code that produces
their figures and numbers, and a container image that reproduces both from
scratch.

**In a hurry?** Go straight to [How to run it](#how-to-run-it) — everything can
be run either through Docker or through Python, and both do the same work.

The SBFL parts are ported from `static-sbfl-for-nn`. The code picks a device in
the order **MPS → CUDA → CPU**, so on Apple Silicon everything trains on the
Metal GPU with no extra flags (that applies to the Python path only — a
container cannot reach Metal).

## Experiments and their results

Two experiments, asking the same question along two different axes.

| | **Experiment 1** | **Experiment 2** |
| --- | --- | --- |
| **Question** | At a fixed point in training, are suspicious neurons the high-gradient ones? | How do a single model's suspiciousness and gradient values *change* while it trains? |
| **What varies** | the random initialisation (100 independent instances per model) | the epoch (one model, 200 epochs) |
| **Correlation axis** | across instances, separately per epoch snapshot | — (trajectories, not a correlation) |
| **Models** | MLP and LeNet-5, each on MNIST and Fashion-MNIST (4 pairs) | the same 4 pairs, one seeded instance each |
| **Measured at** | epochs 0 (untrained), 1 and 10 | epoch 0 and after **every** epoch, 0…200 |
| **Primary result** | correlation heatmaps + per-layer statistics | per-neuron and whole-population trajectory plots + value log |
| **Run it, Docker** | `docker compose run --rm cpu experiment1` | `docker compose run --rm cpu experiment2` |
| **Run it, Python** | [4 `make` steps](#run-experiment-1--are-suspicious-neurons-the-high-gradient-ones) | [3 `make` steps](#run-experiment-2--how-do-the-values-change-while-a-model-trains) |
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
| **Population overview** (primary) | `outputs/trajectory/figures/all-combos__population-trajectories.{png,pdf}` | The same 4 × 3 grid, but every curve is the **mean over all of that model's neurons** (5th–95th percentile band around it), with the fitted trend drawn on top. |
| **Population plots** | `outputs/trajectory/figures/<combo>__population__trajectory.{png,pdf}` | One model's three metrics, population mean vs population mean \|gradient\|. |
| **Fitted trends** | `outputs/trajectory/population_fits.csv`, `population_trajectories.md` | Per model × metric: asymptote `a`, time constant `τ`, R², the epoch from which the fit stays within 5 % of `a`, and the drift over the last quarter of the run — the numeric answer to "does it converge?". |
| Population values | `outputs/trajectory/population_trajectories.csv` | Every plotted point: mean, median, p05, p95, std, zero-share and the fitted value, per epoch. |
| **Value ranges** | `outputs/trajectory/value_stats.csv` | The same per-layer/metric summary, per epoch, for the seeded run — plus a `layer = all` row per epoch, which is what the population figures are drawn from. |

Every run of every script also writes a timestamped markdown log to
`outputs/logs/`, recording the parameters it used and what it produced.

## How to run it

There are **two ways** to run the experiments, and they do exactly the same
work. Pick whichever suits you:

| | **Docker** | **Python on your machine** |
| --- | --- | --- |
| You need | Docker | Python 3.11+ and PyTorch |
| Runs on | CPU, or an NVIDIA GPU | CPU, an NVIDIA GPU, **or** Apple Silicon (MPS) |
| Good for | reproducing the results without installing anything | changing the code and re-running |

Both write their results to `outputs/` in this folder. Both can be stopped and
started again: anything already on disk is skipped, so you never redo finished
work.

### Set up: Docker

```bash
docker compose build cpu
```

For an NVIDIA GPU, build `gpu` instead of `cpu` and add `--gpus all` when you
run it. Details and the plain `docker run` form are in
[Docker, in more detail](#docker-in-more-detail).

### Set up: Python

Install PyTorch first — pick the build for your machine at
[pytorch.org](https://pytorch.org/get-started/locally/) — then this package:

```bash
pip install -e ".[dev]"
```

### Check it works first (2 minutes)

A full run takes hours. This runs both experiments at a tiny scale, so you find
out that everything is wired up before you commit to the real thing:

```bash
docker compose run --rm cpu smoke
```

The Python equivalent is any command below with the numbers turned down, for
example `make train-ensemble ENSEMBLE_INSTANCES=4 ENSEMBLE_EPOCHS=2`.

### Run experiment 1 — are suspicious neurons the high-gradient ones?

**Docker** — one word runs all five steps:

```bash
docker compose run --rm cpu experiment1
```

**Python** — the same five steps, one at a time:

```bash
make train-ensemble        # 1. train 100 models per pair, measure at epochs 0, 1 and 10
make correlate-ensemble    # 2. correlate across those 100 models, one result per epoch
make evaluate-ensemble     # 3. build the interactive HTML report
make figures-ensemble      # 4. write the PNG/PDF figures for LaTeX
make value-stats           # 5. write the value-range table
```

Results land in `outputs/ensemble/` and `outputs/visualizations/`. What each
file is: [Experiment 1](#experiment-1--suspiciousness-vs-gradient-across-random-initialisations).

### Run experiment 2 — how do the values change while a model trains?

**Docker** — one word runs all three steps:

```bash
docker compose run --rm cpu experiment2
```

**Python** — the same three steps, one at a time:

```bash
make train-trajectory            # 1. train 4 models for 200 epochs, measure after every epoch
make plot-trajectory             # 2. plot one random neuron per model
make plot-trajectory-population  # 3. plot the mean over ALL neurons, with a fitted trend
```

Or `make pipeline-trajectory` to run all three in order. Results land in
`outputs/trajectory/`. What each file is:
[Experiment 2](#experiment-2--how-suspiciousness-evolves-during-training).

### Change the settings

The same knobs work in both paths — as `make` variables, or as `-e` flags to
Docker:

```bash
make pipeline-ensemble ENSEMBLE_INSTANCES=50 ENSEMBLE_EPOCHS=20
docker compose run --rm -e ENSEMBLE_INSTANCES=50 -e ENSEMBLE_EPOCHS=20 cpu experiment1
```

`make help` lists every target and every variable.

## Experiment 1 — suspiciousness vs. gradient across random initialisations

*Run it: [Docker or Python](#run-experiment-1--are-suspicious-neurons-the-high-gradient-ones).
Produces the correlation heatmaps.*

We build many **fresh copies of the same model**, each starting from a different
random initialisation, and then ask: *do the neurons that look suspicious in one
copy tend to be the high-gradient ones, at the same point in training?* The
correlation is measured **across those copies**, one result per epoch snapshot.

Correlating across copies — rather than across epochs, as the secondary pipeline
does — is what makes the answer a statement about *neurons* instead of about one
particular training run. That is also why there are 100 copies per pair.

**The default run:** 100 copies of each pair (MLP and LeNet, on MNIST and
Fashion-MNIST), trained for **10 epochs**. Suspiciousness (**ochiai, tarantula,
dstar**) and the gradient are measured on the held-out split at epochs **0
(before any training), 1 and 10**. Each copy is trained **once, straight
through** — the snapshots are taken along the way, so nothing is trained twice.
Every copy's **held-out accuracy and loss** are recorded at each of those epochs.

Two more targets are useful here:

```bash
make overview-ensemble        # an HTML overview of this pipeline (no data needed)
make clean-ensemble           # delete only outputs/ensemble (asks first)
```

Change the default run with `ENSEMBLE_INSTANCES=`, `ENSEMBLE_EPOCHS=`,
`ENSEMBLE_CAPTURE_EPOCHS=` (shared by all the steps — which epochs to measure,
correlate, show and export), `ENSEMBLE_METRICS=` and `ENSEMBLE_OVERWRITE=1`:

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

**1. `make paper-figures` — the curated set.** Five numbered folders under
`outputs/paper/`, each holding a handful of PNGs **and a `README.md`** that says
what they show, carries a ready-to-paste `\caption{}` and lists the caveats
worth naming in the text:

| Folder | Goes in | Shows |
| --- | --- | --- |
| `01_method_what-is-correlated` | Method | suspiciousness, gradient and their correlation for one dense layer, epochs 0 vs 1 |
| `02_dense_first-layer` | Results | the main result: correlation at epochs 0/1/10, first dense layer, both MLPs |
| `03_conv_layers` | Results | the same for LeNet's two convolution stages (dense vs. conv contrast) |
| `04_value_ranges` | Results | the condensed value-range table (markdown + CSV) |
| `05_trajectories` | Results | experiment 2's two overview figures — one random neuron, and the population mean with its fitted trend — plus both value logs and the fit table |

```bash
make paper-figures
```

It *selects* from the full export via `manifest.csv` — it draws nothing itself,
so the figures in the paper are byte-identical to the ones in the artefact. The
selection is `scripts/paper_set.py`, one declarative block per folder. The make
target refreshes the two inputs that a curated folder needs and that the export
does not already hold: the shared-scale row for folder 01, and the trajectory
value-stats plus population figure for folder 05.

There are **no pooling layers to plot**: suspiciousness and gradients are
captured by forward hooks on `Linear`/`Conv2d` modules only, so LeNet-5's
`MaxPool2d` stages produce no values. Folder 03 shows the two convolution stages
instead.

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

*Run it: [Docker or Python](#run-experiment-2--how-do-the-values-change-while-a-model-trains).
Produces the value curves over training.*

This asks the opposite question to experiment 1. Not "across many copies at one
fixed epoch", but **how do one model's suspiciousness and gradient values move
over a long training run?** One copy of each of the 4 pairs, all from
`--seed 42`, trained for **200 epochs**, measured before the first epoch and
after **every** epoch. One neuron per model is then picked at random and its
curves are drawn; step 3 does the same for every neuron at once.

```bash
make clean-trajectory         # delete only outputs/trajectory (asks first)
```

Measured values are **never recomputed**: a pair whose epochs are all on disk is
skipped unless you pass `TRAJ_OVERWRITE=1`. Change the run with `TRAJ_EPOCHS=`,
`TRAJ_SEED=` (the model seed, which is also the name of the run folder) and
`TRAJ_NEURON_SEED=` (which neuron gets picked).

```
outputs/trajectory/gradients/<combo>/seed042/epoch_<NN>.pt      per-epoch gradients
outputs/trajectory/suspiciousness/<combo>/seed042/epoch_<NN>.pt per-epoch suspiciousness
outputs/trajectory/<combo>__training.csv                        accuracy + loss per epoch
outputs/trajectory/figures/<combo>__<layer>-n<idx>__trajectory.{png,pdf}
outputs/trajectory/figures/all-combos__neuron-trajectories.{png,pdf}
outputs/trajectory/neuron_trajectories.md / .csv                which neuron, and every value
outputs/trajectory/value_stats.csv                              min/median/mean/max per layer x metric
outputs/trajectory/figures/<combo>__population__trajectory.{png,pdf}
outputs/trajectory/figures/all-combos__population-trajectories.{png,pdf}
outputs/trajectory/population_trajectories.md / .csv            the whole-model curves
outputs/trajectory/population_fits.csv                          fitted trend per model x metric
```

### T3 — the same overview for **all** neurons at once

One neuron is an illustration; `make plot-trajectory-population` answers the
same question for the whole model. Every neuron of a model is pooled into one
population and summarised per epoch (mean by default, `--stat median` for the
outlier-driven D\*), drawn in the same **4 models × 3 metrics** grid, with the
5th–95th percentile band showing how much of the population follows the mean.

Each panel also carries a fitted trend — `y = a + b·exp(-(e - e₀)/τ)` — so
"does it converge?" gets a number instead of a verdict: `a` is the level the
curve is heading for, `τ` how fast it gets there, and `population_fits.csv`
records both alongside R², the epoch from which the fit stays within 5 % of `a`,
and the drift over the last quarter of the run. Epoch 0 (the untrained model) is
excluded from the fit by default — its values are orders of magnitude off and
would otherwise dominate the least squares. Use `--fit linear` for a plain trend
line, `--fit none` to skip fitting, `--layer conv1` for a single layer instead of
the whole model, and `--from-epoch 1` to drop the untrained spike from the plot.

This step reads `outputs/trajectory/value_stats.csv` — the make target refreshes
it first — so the plotted curve and the reported table are the same numbers.

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

## Docker, in more detail

The commands to run each experiment are [further up](#how-to-run-it); this
section is the background. You need no local Python, no PyTorch and no dataset
download. There are two variants built from one Dockerfile, and they differ only
in which PyTorch wheel index they use:

| Variant | Hardware | Works on |
| --- | --- | --- |
| **cpu** | CPU only | Linux, Windows (Docker Desktop / WSL2), macOS on Intel **and** Apple Silicon |
| **gpu** | NVIDIA CUDA | Linux and Windows/WSL2 with an NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |

With an NVIDIA GPU, swap `cpu` for `gpu` in any command:

```bash
docker compose run --rm gpu experiment1
docker compose run --rm gpu experiment2
```

### Without docker compose

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

### Your results stay on your machine

`outputs/`, `data/` and `trained_models/` are mounted from this folder, so
**everything the experiments produce stays on your machine** after the container
exits — and a second run reuses the datasets it already downloaded and skips the
work it already did. Nothing of value is written inside the image.

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

Change settings with `-e`, exactly like the `make` variables:

```bash
docker compose run --rm -e ENSEMBLE_INSTANCES=20 -e ENSEMBLE_CAPTURE_EPOCHS=0,1,5 cpu experiment1
```

Two things worth knowing before you start a full run:

* **Apple Silicon**: a container cannot reach Metal, so the image runs on CPU
  there. To use the GPU on a Mac, take the Python path with `make` instead.
* **How long it takes**: experiment 1 is 4 model/dataset pairs × 100 copies × 10
  epochs — hours on a CPU. You can interrupt it safely and start again, because
  finished copies are skipped. Run `smoke` first to check your setup.

## Releases

Pushing a version tag runs the tests (on Python 3.11, 3.12 and 3.14), builds the
CPU image to prove the artefact still builds from a clean checkout, and then
publishes a GitHub release with the sdist, the wheel and their checksums — see
[`.github/workflows/release.yml`](.github/workflows/release.yml):

```bash
git tag -a v1.0.0 -m "Artefact for <paper>" && git push origin v1.0.0
```

The workflow can also be started by hand from the Actions tab against a tag that
already exists.

Each published release is archived by [Zenodo](https://zenodo.org/), which mints
a DOI for it.

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
  stats.py                   value ranges of a set of per-neuron values
  trends.py                  fit a + b*exp(-e/tau) (or a line) to a per-epoch curve,
                             and report whether it settles
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
           experiment 2: train_trajectory, plot_trajectory,
                         plot_trajectory_population
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
LICENSE                      MIT
CITATION.cff                 how to cite this work
```

See `docs/DESIGN.md` for design decisions and the recommended test coverage.

## Citation

If you use this code, please cite it. `CITATION.cff` in this repository holds
the machine-readable version, and GitHub turns it into a ready-to-paste entry
under **Cite this repository**.

## License

[MIT](LICENSE). The SBFL parts are ported from `static-sbfl-for-nn`.
