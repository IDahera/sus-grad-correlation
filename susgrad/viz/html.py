"""HTML visualisation utilities (the single home for HTML-producing functions).

Generic building blocks (page template, embedded matplotlib figures, tables, line
charts and heatmaps) plus the higher-level reports that use them:

    * :func:`build_training_report`  -- per-epoch loss/accuracy + summary table
    * heatmap / comparison helpers used by :mod:`susgrad.viz.render`

Keeping every HTML/figure helper here means the other modules just orchestrate.
"""

import base64
import io
import math
from html import escape
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# --- generic helpers -----------------------------------------------------------

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
 :root {{
   --bg: #0f1117; --panel: #181b24; --panel-2: #1f2330; --ink: #e7e9ee;
   --muted: #9aa1b1; --line: #2a2f3d; --accent: #6ea8fe; --accent-2: #f08a8a;
 }}
 * {{ box-sizing: border-box; }}
 body {{
   font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
   margin: 0; padding: 2.2rem clamp(1rem, 4vw, 3rem); color: var(--ink);
   background: radial-gradient(1200px 600px at 80% -10%, #1b2030 0%, var(--bg) 60%);
   line-height: 1.5;
 }}
 header.hero {{ margin-bottom: 1.5rem; }}
 h1 {{ font-size: 1.5rem; margin: 0 0 .35rem; letter-spacing: -0.01em; }}
 h2 {{ font-size: 1.05rem; margin: 0 0 .75rem; }}
 .meta {{ color: var(--muted); font-size: .88rem; margin: .15rem 0; }}
 .section {{
   background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
   padding: 1.1rem 1.25rem; margin: 1.1rem 0;
   box-shadow: 0 1px 0 rgba(255,255,255,.02), 0 12px 30px rgba(0,0,0,.25);
 }}
 .row {{ display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; }}
 .tile {{ background: var(--panel-2); border: 1px solid var(--line); border-radius: 10px; padding: .5rem; }}
 img {{ max-width: 100%; display: block; border-radius: 6px; }}
 .pill {{
   display: inline-block; font-size: .72rem; color: var(--muted);
   border: 1px solid var(--line); border-radius: 999px; padding: .1rem .55rem; margin-left: .4rem;
 }}
 table {{ border-collapse: collapse; margin: .35rem 0 .25rem; font-size: .9rem; width: 100%; }}
 th, td {{ border-bottom: 1px solid var(--line); padding: .5rem .7rem; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 thead th {{ color: var(--muted); font-weight: 600; border-bottom: 1px solid var(--line); }}
 tbody tr:hover {{ background: rgba(110,168,254,.06); }}
 a {{ color: var(--accent); }}
 .tabbar {{ display: flex; flex-wrap: wrap; gap: .3rem; border-bottom: 1px solid var(--line); margin-bottom: 1rem; }}
 .tabbar .tab {{ background: transparent; border: 1px solid var(--line); border-bottom: none;
   color: var(--muted); padding: .4rem .85rem; border-radius: 9px 9px 0 0; cursor: pointer; font-size: .85rem; }}
 .tabbar .tab:hover {{ color: var(--ink); }}
 .tabbar .tab.active {{ background: var(--panel-2); color: var(--ink); }}
 .subtabbar {{ margin-top: .25rem; }}
 .tabpanel {{ display: none; }}
 .tabpanel.active {{ display: block; }}
</style></head><body>
<header class="hero">
<h1>{title}</h1>
<p class="meta">{subtitle}</p>
</header>
{body}
</body></html>
"""

# A dark, readable matplotlib style applied to every figure.
_PLOT_STYLE = {
    "figure.facecolor": "#181b24",
    "axes.facecolor": "#181b24",
    "savefig.facecolor": "#181b24",
    "text.color": "#e7e9ee",
    "axes.labelcolor": "#9aa1b1",
    "axes.edgecolor": "#2a2f3d",
    "xtick.color": "#9aa1b1",
    "ytick.color": "#9aa1b1",
    "axes.titlecolor": "#e7e9ee",
    "grid.color": "#2a2f3d",
    "font.size": 9,
}


def fig_to_base64(fig) -> str:
    """Render a matplotlib figure to a base64 PNG string and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def img_tag(b64_png: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64_png}" alt="{escape(alt)}">'


def render_page(title: str, subtitle: str, sections: Sequence[str]) -> str:
    """Assemble a full HTML document from pre-rendered section strings."""
    return _PAGE.format(
        title=escape(title), subtitle=escape(subtitle), body="\n".join(sections)
    )


def html_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_page(out_path, title: str, subtitle: str, sections: Sequence[str]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_page(title, subtitle, sections), encoding="utf-8")
    return out_path


# Vanilla JS tab switcher (no libraries). Works for nested tab bars because each
# bar only toggles the panels that are its *sibling* direct children.
_TAB_JS = """
<script>
document.querySelectorAll('.tabbar .tab').forEach(function(btn){
  btn.addEventListener('click', function(){
    var bar = btn.closest('.tabbar');
    var wrap = bar.parentElement;
    bar.querySelectorAll('.tab').forEach(function(b){ b.classList.remove('active'); });
    btn.classList.add('active');
    wrap.querySelectorAll(':scope > .tabpanel').forEach(function(p){ p.classList.remove('active'); });
    var target = document.getElementById(btn.dataset.target);
    if (target) target.classList.add('active');
  });
});
</script>
"""


def _render_tab_level(nodes, prefix: str, depth: int = 0) -> str:
    """Recursively render one (possibly nested) level of tabs.

    Each node is ``{"label": str, "children": [...]}`` (branch) or
    ``{"label": str, "html": str}`` (leaf). Branches nest another tab bar.
    """
    buttons, panels = [], []
    barclass = "tabbar" + (" subtabbar" if depth > 0 else "")
    for i, node in enumerate(nodes):
        tid = f"{prefix}-{i}"
        active = " active" if i == 0 else ""
        buttons.append(
            f'<button class="tab{active}" data-target="{tid}">{escape(str(node["label"]))}</button>'
        )
        if "children" in node:
            inner = _render_tab_level(node["children"], tid, depth + 1)
        else:
            inner = node.get("html", "")
        panels.append(f'<div class="tabpanel{active}" id="{tid}">{inner}</div>')
    return f'<div class="{barclass}">{"".join(buttons)}</div>{"".join(panels)}'


_OVERVIEW_CSS = """
<style>
 .grid { display: grid; gap: .8rem; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); }
 .card { background: var(--panel-2); border: 1px solid var(--line); border-radius: 12px; padding: .9rem 1rem; }
 .card h3 { margin: 0 0 .35rem; font-size: .98rem; }
 .card .kv { color: var(--muted); font-size: .82rem; margin: .12rem 0; }
 .badge { display: inline-block; font-size: .68rem; padding: .08rem .5rem; border-radius: 999px;
   border: 1px solid var(--line); color: var(--ink); margin-left: .35rem; }
 .badge.conv { background: rgba(110,168,254,.16); }
 .badge.dense { background: rgba(126,224,162,.14); }
 .rail { display: flex; flex-wrap: wrap; gap: .5rem; align-items: stretch; }
 .step { flex: 1 1 150px; background: var(--panel-2); border: 1px solid var(--line);
   border-radius: 12px; padding: .8rem .9rem; position: relative; }
 .step .n { font-size: .72rem; color: var(--accent); font-weight: 700; letter-spacing: .04em; }
 .step h4 { margin: .15rem 0 .3rem; font-size: .95rem; }
 .step p { margin: 0; color: var(--muted); font-size: .82rem; }
 .final { border-color: var(--accent); }
</style>
"""


# --- formulas tab (SBFL suspiciousness + correlation metrics, with sources) -----

# The neuron-level hit-spectrum variables shared by every suspiciousness metric.
_VARIABLES = [
    ("a_s", "active & success — neuron activation > threshold on a correctly classified sample"),
    ("a_f", "active & failure — neuron activation > threshold on a misclassified sample"),
    ("n_s", "inactive & success — neuron activation ≤ threshold on a correctly classified sample"),
    ("n_f", "inactive & failure — neuron activation ≤ threshold on a misclassified sample"),
    ("threshold", "activation cutoff used to call a neuron \"active\" (0.0 in this project)"),
    ("grad[neuron]", "mean over evaluation samples of |∂loss / ∂activation[neuron]|"),
    ("r", "per-neuron correlation between the suspiciousness series and the gradient series, across an epoch window"),
]

# Where each suspiciousness metric comes from, adapted here to per-neuron hit spectra.
_METRIC_INFO = {
    "ochiai": dict(
        formula="a_f / sqrt[(a_f + n_f) · (a_f + a_s)]", bounded="[0, 1]",
        note="Similarity coefficient from biology (Ochiai, 1957), shown to outperform earlier SBFL coefficients.",
        authors="Abreu, R., Zoeteweij, P., & van Gemund, A. J. C.",
        title="On the Accuracy of Spectrum-based Fault Localization",
        venue="TAICPART-MUTATION 2007", year="2007",
        url="https://doi.org/10.1109/TAIC.PART.2007.13",
    ),
    "tarantula": dict(
        formula="[a_f/(a_f+n_f)] / [a_f/(a_f+n_f) + a_s/(a_s+n_s)]", bounded="[0, 1]",
        note="One of the original, best-known SBFL techniques.",
        authors="Jones, J. A., Harrold, M. J., & Stasko, J.",
        title="Visualization of Test Information to Assist Fault Localization",
        venue="ICSE 2002", year="2002",
        url="https://dl.acm.org/doi/10.1145/581339.581397",
    ),
    "dstar": dict(
        formula="a_f^star / (a_s + n_f)   (star = 3 here)", bounded="unbounded",
        note="Outperformed 16+ prior SBFL techniques across 21 evaluated programs.",
        authors="Wong, W. E., Debroy, V., Gao, R., & Li, Y.",
        title="The DStar Method for Effective Software Fault Localization",
        venue="IEEE Transactions on Reliability", year="2014",
        url="https://doi.org/10.1109/TR.2013.2285319",
    ),
    "jaccard": dict(
        formula="a_f / (a_f + n_f + a_s)", bounded="[0, 1]",
        note="Classic similarity coefficient (Jaccard, 1901); applied to fault localization by Chen et al.",
        authors="Chen, M. Y., Kiciman, E., Fratkin, E., Fox, A., & Brewer, E.",
        title="Pinpoint: Problem Determination in Large, Dynamic Internet Services",
        venue="DSN 2002", year="2002",
        url="https://doi.org/10.1109/DSN.2002.1028897",
    ),
    "kulczynski2": dict(
        formula="0.5 · [a_f/(a_f+n_f) + a_f/(a_f+a_s)]", bounded="[0, 1]",
        note="Mean of failure-recall and failure-precision.",
        authors="Naish, L., Lee, H. J., & Ramamohanarao, K.",
        title="A Model for Spectra-based Software Diagnosis",
        venue="ACM TOSEM 20(3)", year="2011",
        url="https://doi.org/10.1145/2000791.2000795",
    ),
    "op2": dict(
        formula="a_f - a_s / (a_s + n_s + 1)", bounded="unbounded",
        note="Proven ranking-optimal for single-fault programs under the authors' model.",
        authors="Naish, L., Lee, H. J., & Ramamohanarao, K.",
        title="A Model for Spectra-based Software Diagnosis",
        venue="ACM TOSEM 20(3)", year="2011",
        url="https://doi.org/10.1145/2000791.2000795",
    ),
    "gp13": dict(
        formula="a_f · [1 + 1 / (2·a_s + a_f)]", bounded="unbounded",
        note="Discovered by genetic-programming search over formula space; shown human-competitive.",
        authors="Yoo, S.",
        title="Evolving Human Competitive Spectra-Based Fault Localisation Techniques",
        venue="SSBSE 2012", year="2012",
        url="https://doi.org/10.1007/978-3-642-33119-0_18",
    ),
}

# The two epoch-wise correlation coefficients used in the final report.
_CORRELATION_INFO = {
    "pearson": dict(
        formula="r = Σ(xᵢ-x̄)(yᵢ-ȳ) / sqrt[Σ(xᵢ-x̄)² · Σ(yᵢ-ȳ)²]",
        note="Linear correlation of the raw per-epoch suspiciousness/gradient values.",
        authors="Pearson, K.",
        title="Note on Regression and Inheritance in the Case of Two Parents",
        venue="Proceedings of the Royal Society of London, 58", year="1895",
        url="https://doi.org/10.1098/rspl.1895.0041",
    ),
    "spearman": dict(
        formula="ρ = pearson(rank(x), rank(y))",
        note="Pearson correlation of the rank-transformed values -- captures monotonic, not just linear, relationships.",
        authors="Spearman, C.",
        title="The Proof and Measurement of Association Between Two Things",
        venue="American Journal of Psychology, 15", year="1904",
        url="https://www.jstor.org/stable/1412159",
    ),
}

# The paper that adapted SBFL suspiciousness scoring from programs to DNN neurons.
_DEEPFAULT = dict(
    authors="Eniser, H. F., Gerasimou, S., & Sanchez, A.",
    title="DeepFault: Fault Localization for Deep Neural Networks",
    venue="FASE 2019", year="2019",
    url="https://arxiv.org/abs/1902.05974",
)


def _paper_line(info: dict) -> str:
    return (
        f'<div class="kv">{escape(info["authors"])} ({escape(info["year"])}). '
        f'<a href="{escape(info["url"])}" target="_blank" rel="noopener">{escape(info["title"])}</a>. '
        f'{escape(info["venue"])}.</div>'
    )


def _variables_table_html() -> str:
    rows = "".join(
        f"<tr><td><code>{escape(sym)}</code></td><td>{escape(desc)}</td></tr>"
        for sym, desc in _VARIABLES
    )
    return f"<table><thead><tr><th>Symbol</th><th>Meaning</th></tr></thead><tbody>{rows}</tbody></table>"


def _metric_cards_html(metric_names, registry: dict, *, with_citation: bool = True) -> str:
    cells = []
    for name in metric_names:
        info = registry.get(name)
        if not info:
            continue
        cells.append(
            f'<div class="card"><h3>{escape(name)}'
            + (f' <span class="badge">{escape(info["bounded"])}</span>' if "bounded" in info else "")
            + f'</h3><div class="kv"><code>{escape(info["formula"])}</code></div>'
            f'<div class="kv">{escape(info["note"])}</div>'
            + (_paper_line(info) if with_citation else "")
            + "</div>"
        )
    return f'<div class="grid">{"".join(cells)}</div>'


def metric_formula_cards(metric_names, *, with_citation: bool = True) -> str:
    """Public wrapper: pretty formula cards for suspiciousness metrics.

    Used by both overview scripts -- the secondary pipeline's overview shows
    citations (see :func:`_formulas_html`); the main (ensemble) pipeline's
    overview shows the same cards with ``with_citation=False`` (metrics only,
    no paper references).
    """
    return _metric_cards_html(metric_names, _METRIC_INFO, with_citation=with_citation)


def _formulas_html(metric_names, corr_methods) -> str:
    intro = (
        '<div class="section"><h2>Variables</h2>'
        '<p class="meta">Every suspiciousness metric is computed from the same per-neuron hit spectrum: '
        'a neuron is <em>active</em> when its activation exceeds a threshold, and a sample is a '
        '<em>success</em> when the model classifies it correctly (the SBFL idea of "statement executed" / '
        '"test passed", transferred to neurons). This adaptation of SBFL to neural networks follows '
        + _paper_line(_DEEPFAULT) + "</p>" + _variables_table_html() + "</div>"
    )
    metrics_section = (
        '<div class="section"><h2>Suspiciousness metrics (SBFL)</h2>'
        + _metric_cards_html(metric_names, _METRIC_INFO)
        + "</div>"
    )
    corr_section = (
        '<div class="section"><h2>Correlation metrics</h2>'
        + _metric_cards_html(corr_methods, _CORRELATION_INFO)
        + "</div>"
    )
    return intro + metrics_section + corr_section


def _overview_cards(combos) -> str:
    cells = []
    for c in combos:
        klass = "conv" if "conv" in c["kind"].lower() else "dense"
        cells.append(
            f'<div class="card"><h3>{escape(c["label"])}'
            f'<span class="badge {klass}">{escape(c["kind"])}</span></h3>'
            f'<div class="kv">dataset: {escape(c["dataset"])}</div>'
            f'<div class="kv">domain: {escape(c["domain"])}</div>'
            f'<div class="kv">layers: {escape(str(c["hidden"]))}</div></div>'
        )
    return f'<div class="grid">{"".join(cells)}</div>'


def _loop_svg(stops, metrics) -> str:
    stops_txt = ", ".join(f"1..{s}" for s in stops)
    metrics_txt = " / ".join(metrics)
    return f"""
<svg viewBox="0 0 920 250" xmlns="http://www.w3.org/2000/svg" style="max-width:100%">
  <defs>
    <marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 Z" fill="#6ea8fe"/>
    </marker>
  </defs>
  <style>
    .bx {{ fill:#1f2330; stroke:#2a2f3d; }}
    .tt {{ fill:#e7e9ee; font:13px system-ui; }}
    .mut {{ fill:#9aa1b1; font:11px system-ui; }}
    .ar {{ stroke:#6ea8fe; stroke-width:2; fill:none; marker-end:url(#arr); }}
    .lab {{ fill:#9aa1b1; font:11px system-ui; }}
  </style>
  <rect class="bx" x="20" y="60" width="180" height="70" rx="12"/>
  <text class="tt" x="110" y="90" text-anchor="middle">Train 1 epoch (t)</text>
  <text class="mut" x="110" y="110" text-anchor="middle">on the train split</text>

  <rect class="bx" x="280" y="60" width="200" height="70" rx="12"/>
  <text class="tt" x="380" y="88" text-anchor="middle">Per neuron, on test:</text>
  <text class="mut" x="380" y="106" text-anchor="middle">gradient + suspiciousness</text>
  <text class="mut" x="380" y="121" text-anchor="middle">({metrics_txt})</text>

  <rect class="bx" x="560" y="60" width="200" height="70" rx="12"/>
  <text class="tt" x="660" y="88" text-anchor="middle">Snapshot if t ≤ stop</text>
  <text class="mut" x="660" y="106" text-anchor="middle">store window 1..stop</text>
  <text class="mut" x="660" y="121" text-anchor="middle">{stops_txt}</text>

  <line class="ar" x1="200" y1="95" x2="278" y2="95"/>
  <line class="ar" x1="480" y1="95" x2="558" y2="95"/>
  <path class="ar" d="M660,130 C660,200 110,200 110,132"/>
  <text class="lab" x="385" y="195" text-anchor="middle">next epoch (t+1) — repeat to max(stop)</text>

  <rect class="bx final" x="790" y="60" width="110" height="70" rx="12" stroke="#6ea8fe"/>
  <text class="tt" x="845" y="90" text-anchor="middle">Correlate</text>
  <text class="mut" x="845" y="108" text-anchor="middle">per window</text>
  <line class="ar" x1="760" y1="95" x2="788" y2="95"/>
</svg>
"""


def build_pipeline_overview(out_path, *, combos, metrics, corr_methods, stops,
                            datasets_html: str = "",
                            title="susgrad — pipeline overview", subtitle="") -> Path:
    """A standalone overview of the pipeline, models, datasets and outputs.

    If *datasets_html* is provided it becomes a second top-level tab ("Datasets").
    """
    steps = [
        ("01", "Train models", "Train + evaluate each model/dataset pair; per-epoch loss & accuracy."),
        ("02", "Capture gradients", "Mean |∂loss/∂activation| per neuron, after every epoch."),
        ("03", "Capture suspiciousness", f"SBFL per neuron ({', '.join(metrics)}), after every epoch."),
        ("04", "Correlate", f"corr(metric, gradient) across each window, per neuron ({', '.join(corr_methods)})."),
        ("05", "Evaluate", "Render the tabbed HTML report (model → variant → metric)."),
    ]
    rail = '<div class="rail">' + "".join(
        f'<div class="step{" final" if s[0]=="05" else ""}"><div class="n">{s[0]}</div>'
        f'<h4>{escape(s[1])}</h4><p>{escape(s[2])}</p></div>'
        for s in steps
    ) + "</div>"

    evaluated = [
        ("Heatmaps", "Suspiciousness, gradient and correlation per layer — conv layers show one channel at true H×W; dense layers a square overview."),
        ("Decrement comparison", "Suspiciousness and gradient each sorted high→low to compare how the values fall off."),
        ("Correlation distribution", "Per-layer histogram of correlation values with mean|r|, median, %|r|>0.5 and %r=0."),
        ("Variant tabs", f"Compare correlation over different epoch windows ({', '.join(f'1..{s}' for s in stops)})."),
    ]
    eval_cards = '<div class="grid">' + "".join(
        f'<div class="card"><h3>{escape(t)}</h3><div class="kv">{escape(d)}</div></div>'
        for t, d in evaluated
    ) + "</div>"

    pipeline_html = (
        f'<div class="section"><h2>Models × datasets ({len(combos)} pairs)</h2>{_overview_cards(combos)}</div>'
        f'<div class="section"><h2>Pipeline</h2>{rail}</div>'
        f'<div class="section"><h2>The training loop</h2>{_loop_svg(stops, metrics)}</div>'
        f'<div class="section"><h2>What the report shows</h2>{eval_cards}</div>'
    )

    tabs = [{"label": "Pipeline", "html": pipeline_html}]
    if datasets_html:
        tabs.append({"label": "Datasets", "html": datasets_html})
    tabs.append({"label": "Formulas", "html": _formulas_html(metrics, corr_methods)})

    body = _OVERVIEW_CSS + f'<div class="tabs">{_render_tab_level(tabs, "ov")}</div>' + _TAB_JS
    return write_page(out_path, title, subtitle, [body])


def build_tabbed_report(out_path, *, title: str, subtitle: str, tabs,
                        intro_html: str = "") -> Path:
    """One self-contained HTML with arbitrarily nested tabs.

    Args:
        tabs: a tree of nodes — ``{"label": str, "children": [...]}`` for a tab
            that contains more tabs, or ``{"label": str, "html": str}`` for a leaf.
        intro_html: optional pre-rendered HTML shown once, above the tab bar
            (e.g. :func:`colormap_legend_html`).
    """
    if not tabs:
        return write_page(out_path, title, subtitle,
                          ['<div class="section"><p class="meta">Nothing to show.</p></div>'])
    body = f'<div class="tabs">{_render_tab_level(tabs, "t")}</div>'
    return write_page(out_path, title, subtitle, [intro_html, body, _TAB_JS])


# --- charts --------------------------------------------------------------------

def heatmap_base64(
    grid: np.ndarray, cmap: str, title: str, *, vmin=None, vmax=None,
    xlabel: str = "", ylabel: str = "", cbar_label: str = "",
) -> str:
    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(3.9, 3.2))
        im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8.5)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=7.5)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_edgecolor("#2a2f3d")
        cbar.ax.tick_params(labelsize=6.5)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=7.5, color="#9aa1b1")
        return fig_to_base64(fig)


def _cmap_css_gradient(cmap_name: str, n: int = 9) -> str:
    """CSS linear-gradient sampling a matplotlib colormap (for HTML legends)."""
    cmap = plt.get_cmap(cmap_name)
    stops = ", ".join(
        "rgb({},{},{})".format(*(int(round(c * 255)) for c in cmap(i / (n - 1))[:3]))
        for i in range(n)
    )
    return f"linear-gradient(to right, {stops})"


# Self-contained styling so the legend works on any page (the tabbed report does
# not load _OVERVIEW_CSS, so it cannot rely on .card/.kv/.grid).
_CMAP_LEGEND_CSS = """
<style>
 .cmap-grid { display: grid; gap: .8rem; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }
 .cmap-card { background: var(--panel-2); border: 1px solid var(--line); border-radius: 12px; padding: .9rem 1rem; }
 .cmap-card h3 { margin: 0 0 .35rem; font-size: .98rem; }
 .cmap-card .desc { color: var(--muted); font-size: .82rem; margin: .12rem 0; }
 .cmap-bar-row { display: flex; align-items: center; gap: .5rem; margin: .4rem 0; color: var(--muted); font-size: .78rem; }
 .cmap-bar { height: 12px; border-radius: 6px; border: 1px solid var(--line); flex: 1 1 auto; }
</style>
"""


def colormap_legend_html() -> str:
    """A section explaining what the heatmap colors mean, shown once per report."""
    entries = [
        ("magma", "Suspiciousness", "low", "high",
         "Dark → bright = low → high suspiciousness. The numeric scale is per heatmap "
         "(read its colorbar); ochiai / tarantula / jaccard / kulczynski2 are bounded to [0, 1], "
         "dstar / op2 / gp13 are unbounded."),
        ("viridis", "Gradient", "low", "high",
         "Dark → bright = low → high mean |∂loss/∂activation|. The numeric scale is per heatmap "
         "(read its colorbar)."),
        ("coolwarm", "Correlation", "−1", "+1",
         "Blue = −1 (anti-correlated), white = 0 (no correlation), red = +1 (correlated). "
         "Fixed scale across all correlation heatmaps."),
    ]
    cards = "".join(
        '<div class="cmap-card"><h3>{name}</h3>'
        '<div class="cmap-bar-row"><span>{lo}</span>'
        '<div class="cmap-bar" style="background:{bar};"></div><span>{hi}</span></div>'
        '<div class="desc">{desc}</div></div>'.format(
            name=escape(name), lo=lo, hi=hi, bar=_cmap_css_gradient(cmap), desc=escape(desc),
        )
        for cmap, name, lo, hi, desc in entries
    )
    note = ('<p class="meta">Cells without data (the padding when a dense layer\'s neurons are '
            "wrapped into a near-square grid) are drawn blank/background-colored — they are not "
            "values.</p>")
    return (
        _CMAP_LEGEND_CSS
        + f'<div class="section"><h2>Color legend</h2><div class="cmap-grid">{cards}</div>{note}</div>'
    )


def comparison_decrements_base64(
    layer: str, susp_desc: np.ndarray, grad_desc: np.ndarray
) -> str:
    """Both metrics sorted largest→smallest independently, to compare decrements."""
    # Shared x = normalised rank so the two (equal-length) curves line up.
    x = np.linspace(0.0, 1.0, susp_desc.size)
    with plt.rc_context(_PLOT_STYLE):
        fig, ax_left = plt.subplots(figsize=(7.2, 3.0))
        ax_left.plot(x, susp_desc, color="#f08a8a", lw=1.8, label="suspiciousness")
        ax_left.set_xlabel("neuron rank (each metric sorted high → low)")
        ax_left.set_ylabel("suspiciousness", color="#f08a8a")
        ax_left.tick_params(axis="y", labelcolor="#f08a8a")
        ax_left.grid(True, alpha=0.15)

        ax_right = ax_left.twinx()
        ax_right.plot(x, grad_desc, color="#6ea8fe", lw=1.8, label="mean |gradient|")
        ax_right.set_ylabel("mean |gradient|", color="#6ea8fe")
        ax_right.tick_params(axis="y", labelcolor="#6ea8fe")

        ax_left.set_title(f"{layer}: decrement curves (sorted independently)", fontsize=9)
        fig.tight_layout()
        return fig_to_base64(fig)


def correlation_histogram_base64(values, title: str, subtitle: str = "") -> str:
    """Histogram of a layer's correlation values over the fixed range [-1, 1]."""
    v = np.asarray(values, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        if v.size:
            ax.hist(v, bins=30, range=(-1.0, 1.0), color="#6ea8fe", edgecolor="#0f1117")
        ax.axvline(0.0, color="#9aa1b1", lw=1, ls="--")
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel("correlation r")
        ax.set_ylabel("neurons")
        full_title = title if not subtitle else f"{title}\n{subtitle}"
        ax.set_title(full_title, fontsize=8)
        return fig_to_base64(fig)


def image_montage_base64(images, labels, *, ncols=None, title: str = "") -> str:
    """Render a grid of sample images (each [0,1], HxW or HxWx3) with labels."""
    n = len(images)
    ncols = ncols or min(n, 10)
    nrows = max(1, -(-n // ncols))  # ceil
    with plt.rc_context(_PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 0.92, nrows * 1.08))
        axes = list(axes.reshape(-1)) if hasattr(axes, "reshape") else [axes]
        for i, ax in enumerate(axes):
            ax.axis("off")
            if i < n:
                img = images[i]
                ax.imshow(img, cmap="gray" if getattr(img, "ndim", 2) == 2 else None)
                ax.set_title(str(labels[i]), fontsize=7)
        if title:
            fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        return fig_to_base64(fig)


def chips(items) -> str:
    """A row of small pill-style chips."""
    return "".join(f'<span class="pill">{escape(str(x))}</span>' for x in items)


def _loss_accuracy_chart(history: List[dict]) -> str:
    """Side-by-side loss and accuracy curves over epochs."""
    epochs = [h["epoch"] for h in history]
    with plt.rc_context(_PLOT_STYLE):
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(9, 3.0))

        train_loss = [h.get("train_loss") for h in history]
        test_loss = [h.get("test_loss") for h in history]
        if any(v is not None for v in train_loss):
            ax_loss.plot(epochs, train_loss, marker="o", color="#6ea8fe", label="train loss")
        if any(v is not None for v in test_loss):
            ax_loss.plot(epochs, test_loss, marker="o", color="#f08a8a", label="test loss")
        ax_loss.set_title("Loss", fontsize=9)
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss")
        ax_loss.grid(True, alpha=0.15)
        ax_loss.legend(fontsize=8)

        test_acc = [h.get("test_acc") for h in history]
        if any(v is not None for v in test_acc):
            ax_acc.plot(epochs, test_acc, marker="o", color="#7ee0a2", label="test accuracy")
        ax_acc.set_title("Accuracy", fontsize=9)
        ax_acc.set_xlabel("epoch")
        ax_acc.set_ylabel("accuracy (%)")
        ax_acc.grid(True, alpha=0.15)
        ax_acc.legend(fontsize=8)

        fig.tight_layout()
        return fig_to_base64(fig)


# --- training report -----------------------------------------------------------

_SUMMARY_HEADERS = (
    "Combination", "Dataset", "Model", "Weight layers", "Params",
    "Epochs", "Final test acc", "Final test loss",
)


def build_training_report(
    out_path,
    *,
    runs: List[dict],
    title: str = "Training report",
    subtitle: str = "",
) -> Path:
    """Write an HTML training report and return its path.

    Args:
        runs: one dict per model/dataset variant with keys ``label``, ``dataset``,
            ``model_type``, ``weight_layers``, ``params``, ``epochs``,
            ``final_accuracy``, ``final_loss`` and ``history`` (a list of per-epoch
            dicts with ``epoch`` and any of ``train_loss``/``test_loss``/``test_acc``).
    """
    # Summary table across all variants.
    rows = [
        (
            r["label"], r["dataset"], r["model_type"], r["weight_layers"],
            f'{r["params"]:,}', r["epochs"],
            f'{r["final_accuracy"]:.2f}%', f'{r["final_loss"]:.4f}',
        )
        for r in runs
    ]
    sections = [
        '<div class="section"><h2>Summary</h2>'
        + html_table(_SUMMARY_HEADERS, rows)
        + "</div>"
    ]

    # Per-variant loss/accuracy charts (if epoch history is available).
    for r in runs:
        history = r.get("history") or []
        block = [f'<div class="section"><h2>{escape(r["label"])}</h2>']
        block.append(
            f'<p class="meta">{escape(r["model_type"])} · '
            f'{r["weight_layers"]} weight layers · {r["params"]:,} params · '
            f'dataset {escape(r["dataset"])}</p>'
        )
        if history:
            block.append(
                '<div class="row">'
                + img_tag(_loss_accuracy_chart(history), "loss and accuracy")
                + "</div>"
            )
        else:
            block.append('<p class="meta">No per-epoch history available.</p>')
        block.append("</div>")
        sections.append("\n".join(block))

    return write_page(out_path, title, subtitle, sections)
