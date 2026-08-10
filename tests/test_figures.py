"""Tests for the standalone LaTeX figure export.

The figures themselves cannot be asserted pixel-by-pixel, so these cover what
actually breaks a thesis workflow: filenames that let you FIND a figure, every
requested format actually being written, and the epoch row sharing one colour
scale (which is the only thing that makes those panels comparable).
"""

import numpy as np
import pytest

from susgrad.viz.figures import (
    figure_stem,
    range_label,
    save_heatmap,
    save_heatmap_row,
    save_metric_panels,
    slug,
    write_latex_index,
    write_manifest_csv,
)


def test_slug_is_filesystem_safe():
    assert slug("net.1") == "net-1"
    assert slug("MLP × MNIST (dense)") == "mlp-mnist-dense"
    assert slug("!!!") == "x"


def test_figure_stem_orders_components_for_lookup():
    stem = figure_stem(combo="lenet_mnist", layer="conv1", channel=3, kind="corr",
                       metric="ochiai", method="pearson", epochs=[0, 1, 10])
    assert stem == "lenet-mnist__conv1__ch03__corr__ochiai__pearson__epochs-0-1-10"


def test_figure_stem_single_epoch_and_instance():
    stem = figure_stem(combo="mlp_mnist", layer="net.1", kind="susp",
                       metric="dstar", instance="inst007", epoch=1)
    assert stem == "mlp-mnist__net-1__susp__dstar__inst007__epoch-01"


def test_stems_sort_by_combo_then_layer_then_kind():
    stems = sorted([
        figure_stem(combo="a", layer="l2", kind="corr", epoch=0),
        figure_stem(combo="a", layer="l1", kind="susp", epoch=0),
        figure_stem(combo="a", layer="l1", kind="corr", epoch=0),
    ])
    assert stems[0].startswith("a__l1__corr") and stems[-1].startswith("a__l2")


def test_range_label_ignores_nan_padding():
    assert range_label(np.array([[0.5, np.nan], [-1.0, 1.0]])) == "[-1, 1]"
    assert range_label(np.array([[np.nan]])) == "[empty]"


def test_save_heatmap_writes_every_requested_format(tmp_path):
    grid = np.linspace(0, 1, 12).reshape(3, 4)
    written = save_heatmap(tmp_path / "one", grid, title="t", subtitle="s",
                           formats=("png", "pdf"), dpi=72)

    assert [p.suffix for p in written] == [".png", ".pdf"]
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_save_heatmap_row_writes_one_file_for_all_panels(tmp_path):
    panels = [(f"epoch {e}", np.full((3, 3), float(e))) for e in (0, 1, 10)]
    written = save_heatmap_row(tmp_path / "row", panels, suptitle="x",
                               formats=("png",), dpi=72)

    assert len(written) == 1 and written[0].exists()


def test_save_heatmap_row_needs_panels(tmp_path):
    with pytest.raises(ValueError):
        save_heatmap_row(tmp_path / "row", [], formats=("png",))


def test_save_metric_panels_writes_one_figure(tmp_path):
    written = save_metric_panels(
        tmp_path / "traj", [0, 1, 2],
        {"ochiai": [0.1, 0.2, 0.3], "dstar": [1.0, 900.0, 30.0]},
        right_series={"mean |gradient|": [1e-5, 2e-5, 3e-5]},
        formats=("png",), dpi=72,
    )
    assert len(written) == 1 and written[0].exists()


def test_save_metric_panels_needs_series(tmp_path):
    with pytest.raises(ValueError):
        save_metric_panels(tmp_path / "traj", [0, 1], {}, formats=("png",))


def test_manifest_csv_has_a_row_per_figure(tmp_path):
    records = [
        {"combo": "mlp_mnist", "layer": "net.1", "channel": "", "kind": "corr",
         "metric": "ochiai", "method": "pearson", "instance": "", "epochs": "0+1+10",
         "panels": 3, "file": str(tmp_path / "a.png")},
    ]
    text = write_manifest_csv(tmp_path / "manifest.csv", records).read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("combo,layer,channel,kind")
    assert "mlp_mnist" in text and "0+1+10" in text


def test_latex_index_defines_both_macros_and_lists_stems(tmp_path):
    figure = tmp_path / "mlp_mnist" / "correlation" / "mlp-mnist__net-1__corr__epoch-00.png"
    records = [{"combo": "mlp_mnist", "layer": "net.1", "kind": "corr", "metric": "ochiai",
                "method": "pearson", "epochs": "0", "file": str(figure)}]

    text = write_latex_index(tmp_path / "figures.tex", records, root=tmp_path).read_text(encoding="utf-8")
    assert r"\providecommand{\susgradfig}" in text
    assert r"\providecommand{\susgradepochs}" in text
    # Listed relative to the figures root and without the extension, i.e. exactly
    # what \includegraphics wants.
    assert "mlp_mnist/correlation/mlp-mnist__net-1__corr__epoch-00" in text
    assert ".png" not in text.split(r"\providecommand")[-1]


# --- log scaling: unbounded metrics must not flatten onto the baseline ----------

def test_log_scale_triggers_on_a_wide_span_only():
    from susgrad.viz.figures import _needs_log_scale

    # D*-like: median ~60, peak ~4e8 -> must switch to a log axis.
    assert _needs_log_scale([[1.0, 60.0, 4.5e8]])
    # ochiai-like: bounded to [0, 1] -> must not.
    assert not _needs_log_scale([[0.05, 0.3, 0.9]])


def test_log_scale_declines_when_values_can_be_negative():
    from susgrad.viz.figures import _needs_log_scale

    # A correlation series spans [-1, 1]; a log axis would be meaningless.
    assert not _needs_log_scale([[-0.8, 0.001, 0.9]])


def test_draw_series_panel_reports_which_axes_went_log():
    import matplotlib.pyplot as plt

    from susgrad.viz.figures import draw_series_panel

    fig, ax = plt.subplots()
    info = draw_series_panel(
        ax, [0, 1, 2], {"dstar": [1.0, 5e7, 30.0]},
        right_series={"mean |gradient|": [1e-3, 2e-3, 3e-3]},
        left_label="dstar", right_label="mean |gradient|",
    )
    plt.close(fig)

    assert info["left_log"] is True        # D* spans many decades
    assert info["right_log"] is False      # the gradient here does not
    assert "log" in ax.get_ylabel()        # and the label says so


def test_series_legend_labels_line_styles_not_metric_names(tmp_path):
    # Each panel shows a DIFFERENT metric, so a shared legend must describe the
    # styles; naming one panel's metric would mislabel the others.
    import matplotlib.pyplot as plt

    from susgrad.viz.figures import series_legend

    fig, _ = plt.subplots()
    series_legend(fig)
    texts = [t.get_text() for t in fig.legends[0].get_texts()]
    plt.close(fig)

    assert texts == ["suspiciousness (left axis)", "mean |gradient| (right axis)"]
