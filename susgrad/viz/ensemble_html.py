"""HTML report for the MAIN (ensemble) pipeline.

Unlike the secondary pipeline's ``build_tabbed_report`` (everything pre-rendered
server-side as static matplotlib PNGs, one image per tab combination), this
report embeds compact numeric grids as JSON and draws heatmaps LIVE in the
browser with a small vanilla-JS canvas renderer. That's necessary here because
the dropdowns (layer / channel / epoch / metric / correlation method) combine
in far too many ways to pre-render as separate images.

Layout (see the script that calls this, ``scripts/evaluate_ensemble.py``):
    * ONE ROW PER QUANTITY (suspiciousness, gradient, correlation) and ONE
      COLUMN PER CAPTURED EPOCH (base case: 0, 1, 10), so "before training vs.
      after 1 vs. after 10" is a left-to-right comparison instead of a dropdown
      you have to flip between. Within a row the colour scale is SHARED across
      the epochs -- otherwise each panel would be normalised to its own min/max
      and the comparison would be a lie. Each panel still prints its own value
      range underneath.
    * the correlation heatmaps are already aggregated across ALL instances (the
      correlation axis IS the instance population -- see correlate_ensemble.py),
      so there is nothing instance-specific to pick for those; each correlation
      panel is annotated with that layer's mean |r|, median and % strong.
    * the suspiciousness/gradient heatmaps are shown for a SINGLE,
      randomly-chosen instance per combo (logged, so it's
      traceable) -- embedding all instances' raw activations would make the
      file enormous. The instance dropdown reflects this (one, labelled option)
      and is easy to extend later.

Reuses the existing page shell/tab machinery from :mod:`susgrad.viz.html`
(``_PAGE``, tab CSS, ``_render_tab_level``) so both reports share one look.
"""

import json
from html import escape
from pathlib import Path
from typing import Sequence

from susgrad.viz.html import (
    _OVERVIEW_CSS,
    _TAB_JS,
    _overview_cards,
    _render_tab_level,
    _variables_table_html,
    metric_formula_cards,
    write_page,
)

# --- client-side runtime: colormaps + canvas heatmap --------------------------

_ENSEMBLE_CSS = """
<style>
 .ctrl-row { display: flex; flex-wrap: wrap; gap: .6rem 1rem; align-items: center; margin: .6rem 0 1rem; }
 .ctrl-row label { font-size: .78rem; color: var(--muted); display: flex; flex-direction: column; gap: .2rem; }
 .ctrl-row select { background: var(--panel-2); color: var(--ink); border: 1px solid var(--line);
   border-radius: 6px; padding: .3rem .5rem; font-size: .82rem; }
 /* One block per quantity (susp / grad / corr); inside it, one card per epoch. */
 .heat-row { background: var(--panel-2); border: 1px solid var(--line); border-radius: 10px;
   padding: .75rem .8rem; margin: 0 0 1rem; }
 .heat-row > h4 { margin: 0 0 .1rem; font-size: .88rem; font-weight: 600; }
 .heat-row .row-sub { color: var(--muted); font-size: .72rem; margin: 0 0 .6rem; }
 .row-cards { display: grid; gap: .9rem; grid-template-columns: repeat(var(--epochs, 3), minmax(0, 1fr)); }
 .epoch-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: .55rem; }
 .epoch-card h5 { margin: 0 0 .35rem; font-size: .78rem; font-weight: 600; letter-spacing: .02em; }
 /* Cap heatmaps to a fixed width instead of stretching to the full card --
    a small (e.g. 11x12 or 28x28) grid blown up over the whole page width just
    looks blurry/blocky and is harder to read. */
 .epoch-card canvas { display: block; width: 100%; max-width: 320px; height: auto;
   margin: 0 auto; image-rendering: pixelated; border-radius: 6px; background: #10131b; }
 .heat-legend { color: var(--muted); font-size: .7rem; margin-top: .35rem; text-align: center;
   line-height: 1.35; }
 /* One colorbar per ROW (not per card): the scale is shared across the epochs,
    which is exactly what makes the side-by-side comparison meaningful. */
 .cbar { display: flex; align-items: center; gap: .45rem; margin: .6rem 0 0;
   font-size: .68rem; color: var(--muted); }
 .cbar .cb-bar { flex: 1 1 auto; height: 10px; border-radius: 5px; border: 1px solid var(--line); }
 .cbar .cb-bar.seq { background: linear-gradient(to right,
   rgb(20,20,80), rgb(79,58,70), rgb(138,95,60), rgb(196,133,50), rgb(255,170,40)); }
 .cbar .cb-bar.div { background: linear-gradient(to right,
   rgb(20,20,255), rgb(50,50,138), rgb(80,80,80), rgb(138,50,50), rgb(255,20,20)); }
 .axis-note { color: var(--muted); font-size: .74rem; margin: .6rem 0 0; }
 @media (max-width: 900px) { .row-cards { grid-template-columns: 1fr; } }
</style>
"""

# One shared runtime script: colormaps, canvas heatmap drawer, and the dropdown
# wiring that reads each combo's embedded JSON payload.
_ENSEMBLE_JS = r"""
<script>
(function () {
  function seqColor(t) {           // 0..1 -> sequential dark->amber (susp/grad)
    t = Math.max(0, Math.min(1, t));
    var r = Math.round(20 + t * 235), g = Math.round(20 + t * 150), b = Math.round(40 + (1 - t) * 40);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }
  function divColor(t) {           // -1..1 -> diverging blue->white->red (correlation)
    t = Math.max(-1, Math.min(1, t));
    if (t >= 0) {
      var r = Math.round(255 * t + 20 * (1 - t)), g = Math.round(20 + (1 - t) * 60), b = Math.round(20 + (1 - t) * 60);
      return 'rgb(' + r + ',' + g + ',' + b + ')';
    }
    var m = -t, r2 = Math.round(20 + (1 - m) * 60), g2 = Math.round(20 + (1 - m) * 60), b2 = Math.round(255 * m + 20 * (1 - m));
    return 'rgb(' + r2 + ',' + g2 + ',' + b2 + ')';
  }

  // Finite min/max of a grid (nulls/NaNs are padding, not values).
  function gridRange(grid) {
    var lo = Infinity, hi = -Infinity;
    for (var y = 0; y < grid.length; y++) {
      var row = grid[y];
      for (var x = 0; x < row.length; x++) {
        var v = row[x];
        if (v === null || isNaN(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    if (lo === Infinity) { lo = 0; hi = 0; }
    return { min: lo, max: hi };
  }

  // Draw with an EXPLICIT scale (lo/hi) so every panel in a row shares one
  // scale; the returned range is the panel's own, for its caption.
  function drawHeatmap(canvas, grid, diverging, lo, hi) {
    var rows = grid.length, cols = rows ? grid[0].length : 0;
    var own = gridRange(grid);
    if (!rows || !cols) return own;
    canvas.width = cols; canvas.height = rows;
    var ctx = canvas.getContext('2d');
    var span = (hi - lo) || 1;
    for (var y = 0; y < rows; y++) {
      for (var x = 0; x < cols; x++) {
        var v = grid[y][x];
        if (v === null || isNaN(v)) { ctx.fillStyle = '#2a2f3d'; }
        else {
          ctx.fillStyle = diverging ? divColor(v) : seqColor((v - lo) / span);
        }
        ctx.fillRect(x, y, 1, 1);
      }
    }
    return own;
  }

  // Compact value formatting: tiny magnitudes (e.g. gradients ~1e-5) would all
  // collapse to "0.000" under toFixed(3), making the colorbar labels useless.
  function fmtVal(v) {
    if (v !== 0 && Math.abs(v) < 0.001) return v.toExponential(2);
    return v.toFixed(3);
  }

  function fmtPct(v) { return (100 * v).toFixed(1) + '%'; }
  function escapeHtml(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }

  function setup(root) {
    var data = JSON.parse(root.querySelector('script.payload').textContent);
    var elLayer = root.querySelector('[data-role=layer]');
    var elChannel = root.querySelector('[data-role=channel]');
    var elMetric = root.querySelector('[data-role=metric]');
    var elMethod = root.querySelector('[data-role=method]');
    var elRows = root.querySelector('[data-role=rows]');

    function fillSelect(el, items, labelFn) {
      el.innerHTML = items.map(function (v, i) {
        return '<option value="' + i + '">' + escapeHtml(labelFn ? labelFn(v) : v) + '</option>';
      }).join('');
    }

    fillSelect(elLayer, data.layers);
    fillSelect(elMetric, data.metrics);
    fillSelect(elMethod, data.corr_methods);

    // The three quantities, each rendered as one row of per-epoch panels.
    var ROWS = [
      { key: 'susp', title: 'Suspiciousness — chosen instance', diverging: false },
      { key: 'grad', title: 'Gradient — chosen instance', diverging: false },
      { key: 'corr', title: 'Correlation — across all ' + data.n_instances + ' instances', diverging: true }
    ];

    // Build the row/card skeleton once: the epoch set is fixed for a combo, so
    // only the pixels and captions change when a dropdown moves.
    elRows.innerHTML = ROWS.map(function (row) {
      var cards = data.epochs.map(function (e) {
        return '<div class="epoch-card" data-epoch="' + e + '">' +
               '<h5>epoch ' + e + (e === 0 ? ' — before training' : '') + '</h5>' +
               '<canvas data-role="' + row.key + '-canvas-' + e + '"></canvas>' +
               '<div class="heat-legend" data-role="' + row.key + '-legend-' + e + '"></div></div>';
      }).join('');
      var cbar = row.diverging
        ? '<div class="cbar"><span>−1 (anti-correlated)</span><div class="cb-bar div"></div>' +
          '<span>+1 (correlated)</span></div>'
        : '<div class="cbar"><span data-role="' + row.key + '-cb-min">low</span>' +
          '<div class="cb-bar seq"></div><span data-role="' + row.key + '-cb-max">high</span></div>';
      return '<div class="heat-row"><h4>' + escapeHtml(row.title) + '</h4>' +
             '<p class="row-sub" data-role="' + row.key + '-sub"></p>' +
             '<div class="row-cards" style="--epochs:' + data.epochs.length + '">' + cards + '</div>' +
             cbar + '</div>';
    }).join('');

    function currentLayer() { return data.layers[elLayer.value]; }
    function refreshChannels() {
      var layer = currentLayer();
      var n = data.layer_channels[layer] || 1;
      var items = []; for (var i = 0; i < n; i++) items.push(i);
      fillSelect(elChannel, items, function (i) { return 'channel ' + i; });
      elChannel.parentElement.style.display = n > 1 ? '' : 'none';
    }

    function gridFor(rowKey, layer, metric, method, epoch, channel) {
      if (rowKey === 'susp') return data.susp[layer][metric][epoch][channel];
      if (rowKey === 'grad') return data.grad[layer][epoch][channel];
      return data.corr[layer][metric][method][epoch][channel];
    }

    function render() {
      var layer = currentLayer();
      var channel = parseInt(elChannel.value || '0', 10);
      var metric = data.metrics[parseInt(elMetric.value, 10)];
      var method = data.corr_methods[parseInt(elMethod.value, 10)];

      ROWS.forEach(function (row) {
        var grids = data.epochs.map(function (e) {
          return gridFor(row.key, layer, metric, method, e, channel);
        });

        // Shared scale across the epochs of this row -- without it each panel
        // would be normalised to itself and "epoch 0 vs 10" would be unreadable.
        var lo = row.diverging ? -1 : Infinity, hi = row.diverging ? 1 : -Infinity;
        if (!row.diverging) {
          grids.forEach(function (g) {
            var r = gridRange(g);
            if (r.min < lo) lo = r.min;
            if (r.max > hi) hi = r.max;
          });
          if (lo === Infinity) { lo = 0; hi = 0; }
        }

        data.epochs.forEach(function (e, i) {
          var canvas = root.querySelector('[data-role=' + row.key + '-canvas-' + e + ']');
          var own = drawHeatmap(canvas, grids[i], row.diverging, lo, hi);
          var caption = 'range [' + fmtVal(own.min) + ', ' + fmtVal(own.max) + ']';
          if (row.key === 'corr') {
            var st = ((((data.corr_stats || {})[layer] || {})[metric] || {})[method] || {})[e];
            if (st) {
              caption = 'mean |r| ' + st.mean_abs.toFixed(3) + ' · median ' + st.median.toFixed(3) +
                        '<br>|r|>0.5: ' + fmtPct(st.frac_strong) + ' · r=0: ' + fmtPct(st.frac_zero);
            }
          }
          root.querySelector('[data-role=' + row.key + '-legend-' + e + ']').innerHTML = caption;
        });

        var sub = root.querySelector('[data-role=' + row.key + '-sub]');
        if (row.diverging) {
          sub.textContent = method + '(' + metric + ', gradient) per neuron, correlated across ' +
            data.n_instances + ' independently initialised instances · fixed −1…+1 scale, ' +
            'so every panel is directly comparable.';
        } else {
          var what = row.key === 'susp' ? metric + ' suspiciousness' : 'mean |gradient|';
          sub.textContent = what + ' · one shared colour scale across the epochs: [' +
            fmtVal(lo) + ', ' + fmtVal(hi) + '].';
          root.querySelector('[data-role=' + row.key + '-cb-min]').textContent = fmtVal(lo) + ' (low)';
          root.querySelector('[data-role=' + row.key + '-cb-max]').textContent = fmtVal(hi) + ' (high)';
        }
      });

      // Explain what the heatmap axes depict for the selected layer.
      var kind = (data.layer_kinds || {})[layer] || 'dense';
      root.querySelector('[data-role=axis-note]').textContent = kind === 'conv'
        ? 'Heatmap axes: each pixel is one neuron of the selected channel at its true spatial ' +
          'position (x = width, y = height of the feature map).'
        : 'Heatmap axes: this dense layer has no spatial layout — its neurons are wrapped ' +
          'row-major into a near-square grid (cell = one neuron; index = row × columns + column). ' +
          'Dark gray cells are padding, not values.';
    }

    elLayer.addEventListener('change', function () { refreshChannels(); render(); });
    [elChannel, elMetric, elMethod].forEach(function (el) { el.addEventListener('change', render); });

    refreshChannels();
    render();
  }

  document.querySelectorAll('.ensemble-combo').forEach(setup);
})();
</script>
"""


def _controls_html(instance_label: str) -> str:
    # No epoch dropdown: every captured epoch is on screen at once, side by side.
    return (
        '<div class="ctrl-row">'
        f'<label>Instance<select data-role="instance" disabled><option>{escape(instance_label)}</option></select></label>'
        '<label>Layer<select data-role="layer"></select></label>'
        '<label>Channel<select data-role="channel"></select></label>'
        '<label>Suspiciousness metric<select data-role="metric"></select></label>'
        '<label>Correlation method<select data-role="method"></select></label>'
        "</div>"
    )


def _combo_html(payload: dict) -> str:
    epochs = ", ".join(str(e) for e in payload["epochs"])
    intro = (
        f'<p class="axis-note">Epochs <strong>{escape(epochs)}</strong> side by side '
        "(epoch 0 = the untrained random initialisation). Rows: suspiciousness, gradient, "
        "then their across-instance correlation.</p>"
    )
    # The rows themselves are built client-side from the payload (one card per
    # epoch), so the same shell works for any number of captured epochs.
    body = (
        f"{intro}"
        '<div class="epoch-rows" data-role="rows"></div>'
        '<div class="axis-note" data-role="axis-note"></div>'
    )
    payload_json = json.dumps(payload)
    return (
        f'<div class="ensemble-combo" data-combo="{escape(payload["key"])}">'
        f"{_controls_html(payload['instance_label'])}{body}"
        f'<script class="payload" type="application/json">{payload_json}</script>'
        "</div>"
    )


def build_ensemble_report(
    out_path,
    *,
    combo_payloads: Sequence[dict],
    title: str = "susgrad — ensemble report",
    subtitle: str = "",
) -> Path:
    """Write the main (ensemble) pipeline's interactive HTML report.

    Args:
        combo_payloads: one dict per combo (see module docstring for the exact
            shape); built by ``scripts/evaluate_ensemble.py`` from the captured
            gradient/suspiciousness/correlation tensors via
            ``susgrad.viz.transform.build_heatmap``.
    """
    if not combo_payloads:
        return write_page(out_path, title, subtitle,
                          ['<div class="section"><p class="meta">Nothing to show -- run train_ensemble.py + correlate_ensemble.py first.</p></div>'])

    tabs = [{"label": p["label"], "html": _combo_html(p)} for p in combo_payloads]
    # _TAB_JS wires up the tab BUTTON clicks (switches which panel is visible);
    # _ENSEMBLE_JS wires up the dropdowns/heatmaps INSIDE each panel. Both are
    # needed -- without _TAB_JS every tab except the default-active first one is
    # dead: its button has no click handler at all, so nothing opens.
    body = (
        _OVERVIEW_CSS + _ENSEMBLE_CSS
        + f'<div class="tabs">{_render_tab_level(tabs, "ens")}</div>'
        + _TAB_JS + _ENSEMBLE_JS
    )
    return write_page(out_path, title, subtitle, [body])


# --- main (ensemble) pipeline overview ------------------------------------------

def _rail_html(steps) -> str:
    """A row of pipeline-step cards (reuses the '.rail'/'.step' CSS from html.py)."""
    return '<div class="rail">' + "".join(
        f'<div class="step{" final" if i == len(steps) - 1 else ""}"><div class="n">{i + 1:02d}</div>'
        f'<h4>{escape(t)}</h4><p>{escape(d)}</p></div>'
        for i, (t, d) in enumerate(steps)
    ) + "</div>"


def build_ensemble_overview(
    out_path,
    *,
    combos,
    instances: int,
    epochs: int,
    metrics,
    capture_epochs=None,
    datasets_html: str = "",
    title: str = "susgrad — main (ensemble) pipeline overview",
    subtitle: str = "",
) -> Path:
    """Data-free overview of the main (ensemble) pipeline's steps, datasets and metrics.

    Unlike the secondary pipeline's overview (see ``susgrad.viz.html``'s
    "Formulas" tab), the metrics here are listed in the same pretty
    name/formula/description cards but WITHOUT paper citations -- this overview
    is about what the pipeline computes, not where the ideas come from.
    """
    captured = list(capture_epochs) if capture_epochs else [0, epochs]
    captured_text = ", ".join(str(e) for e in captured)
    steps = [
        ("Train ensemble", f"For each of the {len(combos)} combos, train {instances} freshly, "
         f"randomly-initialised instances for {epochs} epochs each — each instance trained ONCE, "
         f"straight through. Capture gradients + all suspiciousness metrics on the held-out split "
         f"in between, at epochs {captured_text} (0 = before any training). The seed behind every "
         "instance is written to outputs/ensemble/seeds/."),
        ("Correlate across instances", f"For each captured epoch ({captured_text}), correlate "
         "each neuron's suspiciousness against its gradient ACROSS THE POPULATION of instances -- "
         "not across epochs like the secondary pipeline. Also writes a text report: per-layer "
         "statistics (mean |r|, median, % strong) plus ASCII heatmaps."),
        ("Evaluate", "Render one interactive HTML report per combo: one row for suspiciousness, "
         "one for gradient, one for the across-instance correlation — each with the captured "
         f"epochs ({captured_text}) side by side on a shared colour scale, plus live "
         "metric/method/layer/channel dropdowns."),
    ]

    pipeline_html = (
        f'<div class="section"><h2>Models × datasets ({len(combos)} pairs)</h2>{_overview_cards(combos)}</div>'
        f'<div class="section"><h2>Pipeline</h2>{_rail_html(steps)}</div>'
    )

    tabs = [{"label": "Pipeline", "html": pipeline_html}]
    if datasets_html:
        tabs.append({"label": "Datasets", "html": datasets_html})
    tabs.append({
        "label": "Metrics",
        "html": (
            '<div class="section"><h2>Variables</h2>'
            '<p class="meta">Every suspiciousness metric is computed from the same per-neuron hit '
            'spectrum: a neuron is <em>active</em> when its activation exceeds a threshold, and a '
            'sample is a <em>success</em> when the model classifies it correctly.</p>'
            + _variables_table_html() + "</div>"
            + '<div class="section"><h2>Suspiciousness metrics (SBFL)</h2>'
            + metric_formula_cards(metrics, with_citation=False)
            + "</div>"
        ),
    })

    body = _OVERVIEW_CSS + f'<div class="tabs">{_render_tab_level(tabs, "ensov")}</div>' + _TAB_JS
    return write_page(out_path, title, subtitle, [body])
