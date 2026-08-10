"""Tests for the HTML utility submodule (training report)."""

from susgrad.viz.html import build_pipeline_overview, build_training_report, html_table


def _run(label, history):
    return {
        "label": label, "key": label, "dataset": "banknote",
        "model_type": "TabularMLP", "weight_layers": 3, "params": 4610,
        "epochs": len(history), "final_accuracy": 98.5, "final_loss": 0.04,
        "history": history,
    }


def test_html_table_escapes_and_renders():
    table = html_table(["A", "B"], [[1, "x"], [2, "y<z"]])
    assert "<table>" in table and "<th>A</th>" in table
    assert "y&lt;z" in table  # html-escaped


def test_training_report_written_with_charts(tmp_path):
    history = [
        {"epoch": 1, "train_loss": 0.6, "test_loss": 0.5, "test_acc": 80.0},
        {"epoch": 2, "train_loss": 0.3, "test_loss": 0.25, "test_acc": 95.0},
    ]
    out = build_training_report(
        tmp_path / "report.html", runs=[_run("mlp_banknote", history)]
    )
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Summary" in text
    assert "Weight layers" in text and "4,610" in text  # params formatted
    assert "data:image/png;base64," in text             # chart embedded


def test_training_report_handles_empty_history(tmp_path):
    out = build_training_report(
        tmp_path / "report.html", runs=[_run("empty", [])]
    )
    text = out.read_text(encoding="utf-8")
    assert "No per-epoch history available." in text


def test_overview_formulas_tab_lists_metrics_and_paper_links(tmp_path):
    combos = [{
        "label": "MLP × Banknote", "dataset": "banknote", "domain": "tabular",
        "kind": "dense · tabular", "hidden": "64, 32",
    }]
    metrics = ("ochiai", "tarantula", "dstar", "jaccard", "kulczynski2", "op2", "gp13")
    corr_methods = ("pearson", "spearman")
    out = build_pipeline_overview(
        tmp_path / "overview.html", combos=combos, metrics=metrics,
        corr_methods=corr_methods, stops=[10, 25, 50], subtitle="test",
    )
    text = out.read_text(encoding="utf-8")

    assert "Formulas" in text
    assert "Variables" in text and "a_f" in text and "a_s" in text
    for name in metrics + corr_methods:
        assert name in text
    # Every metric/correlation card should carry a live source link.
    assert text.count('target="_blank" rel="noopener"') == len(metrics) + len(corr_methods) + 1
