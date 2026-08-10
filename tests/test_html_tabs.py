"""Tests for the nested tabbed HTML report builder."""

from susgrad.viz.html import build_tabbed_report


def test_nested_tabs_model_variant_metric(tmp_path):
    tabs = [
        {"label": "LeNet × MNIST", "children": [
            {"label": "1..10", "children": [
                {"label": "ochiai", "html": "<p>O10</p>"},
                {"label": "dstar", "html": "<p>D10</p>"},
            ]},
            {"label": "1..100", "children": [
                {"label": "ochiai", "html": "<p>O100</p>"},
            ]},
        ]},
        {"label": "MLP × Banknote", "children": [
            {"label": "1..10", "children": [
                {"label": "ochiai", "html": "<p>bank</p>"},
            ]},
        ]},
    ]
    out = build_tabbed_report(tmp_path / "report.html", title="t", subtitle="s", tabs=tabs)
    html = out.read_text(encoding="utf-8")

    assert "LeNet × MNIST" in html and "MLP × Banknote" in html
    assert "1..10" in html and "1..100" in html
    # Leaf content carried through all three levels.
    assert "O100" in html and "bank" in html
    # The tab-switching script is embedded once.
    assert html.count("addEventListener('click'") == 1
    # Each tab level marks its first child active (one per branch node + root).
    assert 'class="tab active"' in html


def test_tabbed_report_handles_no_tabs(tmp_path):
    out = build_tabbed_report(tmp_path / "empty.html", title="t", subtitle="s", tabs=[])
    assert "Nothing to show" in out.read_text(encoding="utf-8")
