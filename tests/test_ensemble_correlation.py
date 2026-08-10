"""Regression test: the main (ensemble) pipeline's correlation axis.

``compute_correlations`` (reused unchanged from the secondary pipeline, see
``susgrad/correlation/analysis.py``) just stacks a list of ``{layer: tensor}``
dicts along a new axis and correlates along it. The secondary pipeline calls it
with one dict per EPOCH (one instance, many epochs); ``correlate_ensemble.py``
calls it with one dict per INSTANCE (one epoch snapshot, many instances). This
test proves that reuse is correct by feeding it a per-instance list and
checking the result against a hand-computed Pearson r -- exactly the axis swap
``correlate_ensemble.py`` relies on.
"""

import pytest
import torch

from susgrad.correlation import compute_correlations


def test_compute_correlations_across_instances_matches_hand_computed_pearson():
    # 5 "instances" (not epochs!), one neuron. Suspiciousness rises with the
    # instance's gradient except for one outlier -- a known, hand-checkable r.
    susp_values = [1.0, 2.0, 3.0, 4.0, 100.0]
    grad_values = [1.0, 2.0, 3.0, 4.0, 5.0]

    susp_per_instance = [{"fc": torch.tensor([v])} for v in susp_values]
    grad_per_instance = [{"fc": torch.tensor([v])} for v in grad_values]

    result = compute_correlations(susp_per_instance, grad_per_instance, methods=["pearson"])

    x = torch.tensor(susp_values)
    y = torch.tensor(grad_values)
    expected = float(torch.corrcoef(torch.stack([x, y]))[0, 1])

    assert float(result["pearson"]["fc"][0]) == pytest.approx(expected, abs=1e-6)


def test_compute_correlations_across_instances_preserves_per_neuron_shape():
    # 4 instances, a (2, 3) "layer" (e.g. a small conv channel x width slice) --
    # exactly what correlate_ensemble.py stacks per epoch snapshot.
    n_instances, shape = 4, (2, 3)
    susp_per_instance = [{"conv1": torch.rand(*shape)} for _ in range(n_instances)]
    grad_per_instance = [{"conv1": torch.rand(*shape)} for _ in range(n_instances)]

    result = compute_correlations(susp_per_instance, grad_per_instance, methods=["pearson", "spearman"])

    assert set(result) == {"pearson", "spearman"}
    for method in result:
        out = result[method]["conv1"]
        assert tuple(out.shape) == shape
        assert torch.isfinite(out).all()
        assert float(out.min()) >= -1.0 and float(out.max()) <= 1.0


def test_compute_correlations_needs_at_least_two_instances():
    susp_per_instance = [{"fc": torch.tensor([1.0])}]
    grad_per_instance = [{"fc": torch.tensor([1.0])}]
    with pytest.raises(ValueError):
        compute_correlations(susp_per_instance, grad_per_instance)
