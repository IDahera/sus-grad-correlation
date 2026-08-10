"""Exact-value unit tests for the Tarantula and Ochiai formulas.

    Tarantula(a_s,a_f,n_s,n_f) = (a_f/(a_f+n_f)) /
                                 ((a_f/(a_f+n_f)) + (a_s/(a_s+n_s)))

    Ochiai(a_s,a_f,n_s,n_f)    = a_f / sqrt((a_f+n_f) * (a_f+a_s))

Both lie in [0, 1]; higher means more suspicious.
"""

import math

import pytest
import torch

from susgrad.sbfl import (
    CORE_METRIC_NAMES,
    METRIC_NAMES,
    HitSpectrum,
    get_gp13,
    get_jaccard,
    get_kulczynski2,
    get_ochiai,
    get_op2,
    get_tarantula,
)


def hs(a_s, a_f, n_s, n_f) -> HitSpectrum:
    """Build a single-neuron hit spectrum from the four counts."""
    t = lambda v: torch.tensor([float(v)])
    return HitSpectrum(a_s=t(a_s), a_f=t(a_f), n_s=t(n_s), n_f=t(n_f), layer_shape=(1,))


# (a_s, a_f, n_s, n_f, expected_ochiai, expected_tarantula) -- all hand-computed.
CASES = [
    (1, 1, 1, 1, 0.5, 0.5),
    (2, 8, 8, 2, 0.8, 0.8),
    (3, 1, 1, 3, 0.25, 0.25),
    (4, 2, 4, 8, 0.2581988897, 0.2857142857),  # ochiai != tarantula here
    (5, 0, 5, 5, 0.0, 0.0),                     # no failures -> not suspicious
]


@pytest.mark.parametrize("a_s,a_f,n_s,n_f,exp_ochiai,exp_tarantula", CASES)
def test_metric_exact_values(a_s, a_f, n_s, n_f, exp_ochiai, exp_tarantula):
    spec = hs(a_s, a_f, n_s, n_f)
    assert float(get_ochiai(spec)[0]) == pytest.approx(exp_ochiai, abs=1e-6)
    assert float(get_tarantula(spec)[0]) == pytest.approx(exp_tarantula, abs=1e-6)


def test_metrics_can_differ():
    # The two formulas are distinct: confirm a case where they disagree.
    spec = hs(4, 2, 4, 8)
    assert float(get_ochiai(spec)[0]) != pytest.approx(float(get_tarantula(spec)[0]))


@pytest.mark.parametrize(
    "a_s,a_f,n_s,n_f,jac,kul,op2,gp13",
    [
        (2, 8, 8, 2, 0.6666667, 0.8, 7.8181818, 8.6666667),
        (1, 1, 1, 1, 0.3333333, 0.5, 0.6666667, 1.3333333),
    ],
)
def test_added_metric_exact_values(a_s, a_f, n_s, n_f, jac, kul, op2, gp13):
    spec = hs(a_s, a_f, n_s, n_f)
    assert float(get_jaccard(spec)[0]) == pytest.approx(jac, abs=1e-6)
    assert float(get_kulczynski2(spec)[0]) == pytest.approx(kul, abs=1e-6)
    assert float(get_op2(spec)[0]) == pytest.approx(op2, abs=1e-6)
    assert float(get_gp13(spec)[0]) == pytest.approx(gp13, abs=1e-6)


def test_added_bounded_metrics_stay_in_unit_interval():
    g = torch.Generator().manual_seed(1)
    counts = torch.randint(0, 80, (300, 4), generator=g).float()
    spec = HitSpectrum(a_s=counts[:, 0], a_f=counts[:, 1],
                       n_s=counts[:, 2], n_f=counts[:, 3], layer_shape=(300,))
    for fn in (get_jaccard, get_kulczynski2):
        v = fn(spec)
        assert torch.isfinite(v).all()
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0


def test_purely_failure_active_neuron_is_maximally_suspicious():
    # Active only on failures (a_s=0, n_f=0): both metrics reach 1.0.
    spec = hs(0, 7, 5, 0)
    assert float(get_ochiai(spec)[0]) == pytest.approx(1.0)
    assert float(get_tarantula(spec)[0]) == pytest.approx(1.0)


# --- cross-check against independent closed-form reference --------------------

def _ref_ochiai(a_s, a_f, n_s, n_f):
    den = math.sqrt((a_f + n_f) * (a_f + a_s))
    return a_f / den if den > 0 else 0.0


def _ref_tarantula(a_s, a_f, n_s, n_f):
    left = a_f / (a_f + n_f)
    right = a_s / (a_s + n_s)
    return left / (left + right) if (left + right) > 0 else 0.0


def test_matches_reference_on_many_spectra():
    g = torch.Generator().manual_seed(0)
    # All counts >= 1 so no denominator is zero -> exact agreement expected.
    counts = torch.randint(1, 100, (500, 4), generator=g)
    a_s = counts[:, 0].float()
    a_f = counts[:, 1].float()
    n_s = counts[:, 2].float()
    n_f = counts[:, 3].float()
    spec = HitSpectrum(a_s=a_s, a_f=a_f, n_s=n_s, n_f=n_f, layer_shape=(500,))

    ochiai = get_ochiai(spec)
    tarantula = get_tarantula(spec)
    for i in range(counts.shape[0]):
        vals = tuple(float(counts[i, j]) for j in range(4))  # a_s,a_f,n_s,n_f
        assert float(ochiai[i]) == pytest.approx(_ref_ochiai(*vals), abs=1e-6)
        assert float(tarantula[i]) == pytest.approx(_ref_tarantula(*vals), abs=1e-6)
        assert 0.0 <= float(ochiai[i]) <= 1.0
        assert 0.0 <= float(tarantula[i]) <= 1.0


def test_core_metric_names_are_the_three_the_experiments_report():
    # The ensemble + trajectory experiments deliberately narrow to these three;
    # all seven stay available (and tested above) for the secondary pipeline.
    assert CORE_METRIC_NAMES == ("ochiai", "tarantula", "dstar")
    assert set(CORE_METRIC_NAMES).issubset(METRIC_NAMES)
