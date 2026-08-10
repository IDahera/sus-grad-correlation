"""Human-readable dataset descriptions + best-effort live previews for the overview.

Static metadata (descriptions, feature names, class meanings) renders with no
dependencies. Live previews — a few example rows for tabular sets, a per-class
image montage for image sets — are added when the data can be loaded (torch /
torchvision / scikit-learn), and skipped gracefully otherwise.
"""

import logging

from susgrad.viz.html import chips, html_table, image_montage_base64

logger = logging.getLogger(__name__)

DATASET_META = {
    "banknote": {
        "title": "Banknote Authentication", "kind": "tabular", "task": "binary",
        "desc": "Statistical features from wavelet transforms of banknote photographs.",
        "features": ["variance", "skewness", "curtosis", "entropy"],
        "target": "two classes (genuine vs forged banknote)",
    },
    "sonar": {
        "title": "Sonar (Connectionist Bench)", "kind": "tabular", "task": "binary",
        "desc": "Energy returned in 60 frequency bands from sonar signals bounced off a surface.",
        "features": [f"band {i+1}" for i in range(60)],
        "target": "two classes (rock vs metal cylinder / 'mine')",
    },
    "ionosphere": {
        "title": "Ionosphere", "kind": "tabular", "task": "binary",
        "desc": "17 radar pulses × (real, imaginary) = 34 features of returns from the ionosphere.",
        "features": [f"pulse {i//2+1} {'re' if i % 2 == 0 else 'im'}" for i in range(34)],
        "target": "two classes (good = structured return vs bad = no structure)",
    },
    "moons": {
        "title": "Two Moons (synthetic)", "kind": "tabular", "task": "binary",
        "desc": "Two interleaving half-circles with Gaussian noise (sklearn make_moons).",
        "features": ["x", "y"], "target": "0 / 1 (which moon)",
    },
    "circles": {
        "title": "Concentric Circles (synthetic)", "kind": "tabular", "task": "binary",
        "desc": "A small circle nested inside a larger one, with noise (sklearn make_circles).",
        "features": ["x", "y"], "target": "0 = outer ring, 1 = inner ring",
    },
    "mnist": {
        "title": "MNIST", "kind": "image", "task": "10-class", "shape": "1×28×28 grayscale",
        "desc": "Handwritten digits.",
        "classes": [str(i) for i in range(10)],
    },
    "fmnist": {
        "title": "Fashion-MNIST", "kind": "image", "task": "10-class", "shape": "1×28×28 grayscale",
        "desc": "Zalando clothing thumbnails.",
        "classes": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    },
    "cifar10": {
        "title": "CIFAR-10", "kind": "image", "task": "10-class", "shape": "3×32×32 RGB",
        "desc": "Small natural images.",
        "classes": ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"],
    },
}


def _feature_chips(features, limit=12) -> str:
    if len(features) <= limit:
        return chips(features)
    return chips(features[:limit]) + f'<span class="pill">+{len(features) - limit} more</span>'


def _tabular_examples_html(name, meta, n_examples, seed) -> str:
    from susgrad.training.datasets import load_tabular_raw  # lazy (needs torch)

    X, y = load_tabular_raw(name, seed)
    feats = meta["features"]
    ncol = min(8, X.shape[1])
    more = X.shape[1] > ncol
    headers = feats[:ncol] + (["…"] if more else []) + ["target"]
    rows = []
    for i in range(min(n_examples, len(X))):
        vals = [f"{float(X[i, j]):.3g}" for j in range(ncol)]
        if more:
            vals.append("…")
        vals.append(str(int(y[i])))
        rows.append(vals)
    return f'<p class="meta">Example rows ({X.shape[1]} features, {len(X)} samples):</p>' + html_table(headers, rows)


def _image_montage_html(name) -> str:
    from torchvision import datasets as tvd, transforms  # lazy

    from susgrad.training.datasets import _DATA_DIR, _IMAGE_SPECS

    cls_name, _mean, _std, _shape = _IMAGE_SPECS[name]
    classes = DATASET_META[name]["classes"]
    ds = getattr(tvd, cls_name)(
        root=str(_DATA_DIR), train=True, download=True, transform=transforms.ToTensor()
    )
    picked = {}
    for img, label in ds:
        label = int(label)
        if label not in picked:
            arr = img.numpy()
            picked[label] = arr[0] if arr.shape[0] == 1 else arr.transpose(1, 2, 0)
        if len(picked) == len(classes):
            break
    order = sorted(picked)
    b64 = image_montage_base64(
        [picked[i] for i in order], [classes[i] for i in order],
        title="one sample per class",
    )
    return f'<div class="tile"><img src="data:image/png;base64,{b64}" alt="samples"></div>'


def build_datasets_html(names, *, with_samples=True, n_examples=5, seed=42) -> str:
    """Build the 'Datasets' tab HTML for the given dataset names (in order)."""
    seen = []
    for n in names:  # de-duplicate, preserve order
        if n not in seen:
            seen.append(n)

    sections = []
    for name in seen:
        meta = DATASET_META.get(name)
        if not meta:
            continue
        badges = f'<span class="badge">{meta["task"]}</span>'
        if meta["kind"] == "image":
            badges += f'<span class="badge">{meta["shape"]}</span>'
        else:
            badges += f'<span class="badge">{len(meta["features"])} features</span>'

        body = [f'<div class="section"><h2>{meta["title"]}{badges}</h2>'
                f'<p class="meta">{meta["desc"]}</p>']

        if meta["kind"] == "tabular":
            body.append(f'<p class="meta">Features:</p>{_feature_chips(meta["features"])}')
            body.append(f'<p class="meta">Ground truth: {meta["target"]}</p>')
            if with_samples:
                try:
                    body.append(_tabular_examples_html(name, meta, n_examples, seed))
                except Exception as exc:  # noqa: BLE001
                    logger.info("No live preview for %s (%s).", name, exc)
                    body.append('<p class="meta"><em>Run on a machine with the data for example rows.</em></p>')
        else:
            body.append(f'<p class="meta">Classes (ground truth):</p>{chips(meta["classes"])}')
            if with_samples:
                try:
                    body.append(f'<div class="row">{_image_montage_html(name)}</div>')
                except Exception as exc:  # noqa: BLE001
                    logger.info("No image sample for %s (%s).", name, exc)
                    body.append('<p class="meta"><em>Run on a machine with the data for a sample grid.</em></p>')

        body.append("</div>")
        sections.append("".join(body))

    return "".join(sections)
