"""Render suspiciousness / gradient / correlation views to one HTML file.

Thin orchestration on top of :mod:`susgrad.viz.html`. Per layer it shows three
structure-preserving heatmaps side by side — suspiciousness, gradient and (if
available) their across-epoch correlation — followed by a decrement-comparison
plot where suspiciousness and gradient are each sorted high→low independently.

For convolutional layers ``(C, H, W)`` a single channel is shown at its true
``H×W`` resolution; the **same** channel index is used for all metrics so they
stay comparable. Dense ``(N,)`` layers use a near-square overview.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from susgrad.viz.html import (
    colormap_legend_html,
    comparison_decrements_base64,
    correlation_histogram_base64,
    heatmap_base64,
    img_tag,
    write_page,
)
from susgrad.viz.transform import (
    MAX_NEURONS_PER_LAYER,
    VisualizationError,
    align_layers,
    build_heatmap,
    correlation_stats,
    sorted_descending,
)


def _tile(b64: str, caption: str) -> str:
    return f'<div class="tile">{img_tag(b64, caption)}</div>'


def render_combo_sections(
    *,
    susp_all: Dict[str, Dict[str, torch.Tensor]],
    grad_by_layer: Dict[str, torch.Tensor],
    corr_by_metric: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    metrics,
    bounded_metrics=(),
    channel: int = 0,
    corr_method: str = "",
    max_neurons: int = MAX_NEURONS_PER_LAYER,
) -> Dict[str, str]:
    """Build the per-layer section HTML for each suspiciousness *metric*.

    The correlation shown in each metric's sub-tab is **that metric's own**
    correlation against the gradient: ``corr_by_metric[metric]`` is a
    ``{layer: tensor}`` mapping (already resolved to the chosen correlation
    method). The gradient heatmap and its decrement curve do not depend on the
    metric, so they are rendered once per layer and reused. Returns
    ``{metric: sections_html}``.
    """
    metrics = [m for m in metrics if m in susp_all]
    if not metrics:
        raise VisualizationError("No suspiciousness metrics available to render.")

    layers = align_layers(susp_all[metrics[0]], grad_by_layer)

    # --- metric-independent pieces, computed once per layer ---
    grad_tile, grad_desc, n_neurons = {}, {}, {}
    for layer in layers:
        g_hm = build_heatmap(layer, grad_by_layer[layer], channel=channel, max_neurons=max_neurons)
        grad_tile[layer] = _tile(
            heatmap_base64(g_hm.grid, "viridis", f"gradient\n{g_hm.info}",
                           xlabel=g_hm.xlabel, ylabel=g_hm.ylabel,
                           cbar_label="mean |gradient|"),
            "gradient",
        )
        grad_desc[layer] = sorted_descending(
            grad_by_layer[layer].detach().cpu().reshape(-1).numpy()
        )
        n_neurons[layer] = int(susp_all[metrics[0]][layer].numel())

    # --- one set of sections per metric, with that metric's own correlation ---
    out: Dict[str, str] = {}
    for metric in metrics:
        vr = (0.0, 1.0) if metric in bounded_metrics else None
        corr_layers = corr_by_metric.get(metric) if corr_by_metric else None
        if corr_layers is not None:
            align_layers(susp_all[metric], corr_layers)

        sections = []
        for layer in layers:
            s_hm = build_heatmap(
                layer, susp_all[metric][layer], channel=channel,
                value_range=vr, max_neurons=max_neurons,
            )
            susp_tile = _tile(
                heatmap_base64(s_hm.grid, "magma", f"suspiciousness · {metric}\n{s_hm.info}",
                               xlabel=s_hm.xlabel, ylabel=s_hm.ylabel,
                               cbar_label=f"suspiciousness ({metric})"),
                "suspiciousness",
            )

            corr_tile = ""
            hist_tile = ""
            if corr_layers is not None:
                c_hm = build_heatmap(layer, corr_layers[layer], channel=channel, max_neurons=max_neurons)
                label = f"corr({metric}, grad) · all epochs"
                label += f" · {corr_method}" if corr_method else ""
                corr_tile = _tile(
                    heatmap_base64(c_hm.grid, "coolwarm", f"{label}\n{c_hm.info}",
                                   vmin=-1.0, vmax=1.0,
                                   xlabel=c_hm.xlabel, ylabel=c_hm.ylabel,
                                   cbar_label="correlation r"),
                    "correlation",
                )
                # Distribution of correlation over ALL neurons of this layer
                # (not channel-sliced) for this metric/gradient pair.
                corr_vals = corr_layers[layer].detach().cpu().reshape(-1).numpy()
                st = correlation_stats(corr_vals)
                subtitle = (f"mean|r|={st['mean_abs']:.2f} · med={st['median']:.2f} · "
                            f"|r|>0.5: {st['frac_strong']:.0%} · r=0: {st['frac_zero']:.0%}")
                hist_tile = _tile(
                    correlation_histogram_base64(
                        corr_vals, f"corr distribution · {metric}", subtitle
                    ),
                    "correlation distribution",
                )

            susp_desc = sorted_descending(
                susp_all[metric][layer].detach().cpu().reshape(-1).numpy()
            )
            cmp_tile = _tile(
                comparison_decrements_base64(layer, susp_desc, grad_desc[layer]),
                "decrement comparison",
            )
            tiles = susp_tile + grad_tile[layer] + corr_tile
            sections.append(
                f'<div class="section"><h2>{layer}'
                f'<span class="pill">{n_neurons[layer]} neurons</span></h2>'
                f'<div class="row">{tiles}</div>'
                f'<div class="row">{cmp_tile}{hist_tile}</div></div>'
            )
        out[metric] = "".join(sections)
    return out


def render_html(
    out_path,
    *,
    title: str,
    subtitle: str,
    susp_by_layer: Dict[str, torch.Tensor],
    grad_by_layer: Dict[str, torch.Tensor],
    corr_by_layer: Optional[Dict[str, torch.Tensor]] = None,
    channel: int = 0,
    susp_value_range: Optional[Tuple[float, float]] = None,
    corr_method: str = "",
    max_neurons: int = MAX_NEURONS_PER_LAYER,
) -> Path:
    """Write the HTML report and return its path.

    Raises :class:`susgrad.viz.transform.VisualizationError` if the data cannot be
    visualised or the layers/neuron counts of the metrics do not line up.
    """
    layers = align_layers(susp_by_layer, grad_by_layer)
    if corr_by_layer is not None:
        align_layers(susp_by_layer, corr_by_layer)  # same layers & neuron counts

    sections = []
    for layer in layers:
        susp_hm = build_heatmap(
            layer, susp_by_layer[layer], channel=channel,
            value_range=susp_value_range, max_neurons=max_neurons,
        )
        grad_hm = build_heatmap(
            layer, grad_by_layer[layer], channel=channel, max_neurons=max_neurons
        )

        tiles = [
            _tile(heatmap_base64(susp_hm.grid, "magma", f"suspiciousness\n{susp_hm.info}",
                                 xlabel=susp_hm.xlabel, ylabel=susp_hm.ylabel,
                                 cbar_label="suspiciousness"),
                  "suspiciousness"),
            _tile(heatmap_base64(grad_hm.grid, "viridis", f"gradient\n{grad_hm.info}",
                                 xlabel=grad_hm.xlabel, ylabel=grad_hm.ylabel,
                                 cbar_label="mean |gradient|"),
                  "gradient"),
        ]
        if corr_by_layer is not None:
            corr_hm = build_heatmap(
                layer, corr_by_layer[layer], channel=channel, max_neurons=max_neurons
            )
            label = f"correlation ({corr_method})" if corr_method else "correlation"
            tiles.append(_tile(
                heatmap_base64(corr_hm.grid, "coolwarm", f"{label}\n{corr_hm.info}",
                               vmin=-1.0, vmax=1.0,
                               xlabel=corr_hm.xlabel, ylabel=corr_hm.ylabel,
                               cbar_label="correlation r"),
                "correlation",
            ))

        susp_desc = sorted_descending(susp_by_layer[layer].detach().cpu().reshape(-1).numpy())
        grad_desc = sorted_descending(grad_by_layer[layer].detach().cpu().reshape(-1).numpy())
        cmp_png = comparison_decrements_base64(layer, susp_desc, grad_desc)

        n_neurons = susp_by_layer[layer].numel()
        sections.append(
            f'<div class="section"><h2>{layer}<span class="pill">{n_neurons} neurons</span></h2>'
            f'<div class="row">{"".join(tiles)}</div>'
            f'<div class="row">{_tile(cmp_png, "decrement comparison")}</div></div>'
        )

    return write_page(out_path, title, subtitle, [colormap_legend_html()] + sections)
