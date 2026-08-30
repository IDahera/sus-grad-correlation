"""Tests for the trend fits behind the population trajectory figures.

The fit is what turns "the curve looks flat by epoch 50" into a number in the
write-up, so the properties that matter are: it recovers parameters it was given,
it says "not settled" when a curve is still moving, and it never blows up on the
degenerate inputs these curves actually contain (a dead layer is exactly
constant, a metric can be exactly zero everywhere).
"""

import math

import numpy as np
import pytest

from susgrad.trends import (
    _solve_linear_part,
    compare_models,
    describe_convergence,
    fit_curve_values,
    fit_exponential,
    fit_linear,
    fit_series,
)


def _exp_curve(a, b, tau, epochs=200):
    x = np.arange(epochs + 1, dtype=float)
    return x, a + b * np.exp(-x / tau)


def test_exponential_recovers_the_parameters_it_was_given():
    x, y = _exp_curve(a=0.07, b=0.5, tau=12.0)

    fit = fit_exponential(x, y)

    assert fit.params["a"] == pytest.approx(0.07, rel=1e-3)
    assert fit.params["b"] == pytest.approx(0.5, rel=1e-2)
    assert fit.params["tau"] == pytest.approx(12.0, rel=5e-2)
    assert fit.r2 == pytest.approx(1.0, abs=1e-6)


def test_exponential_survives_noise_and_still_finds_the_asymptote():
    x, y = _exp_curve(a=1.5, b=3.0, tau=25.0)
    rng = np.random.default_rng(0)
    noisy = y + rng.normal(0.0, 0.05, size=y.shape)

    fit = fit_exponential(x, noisy)

    assert fit.params["a"] == pytest.approx(1.5, abs=0.05)
    assert fit.r2 > 0.95


def test_asymptote_is_what_the_curve_tends_to():
    x, y = _exp_curve(a=0.2, b=1.0, tau=8.0)
    fit = fit_exponential(x, y)

    # Far past the last captured epoch the fit must sit on its asymptote.
    assert float(fit.predict([10_000])[0]) == pytest.approx(fit.asymptote, abs=1e-9)
    assert fit.half_life == pytest.approx(fit.tau * math.log(2))


def test_settle_x_answers_within_the_run_or_not_at_all():
    x, y = _exp_curve(a=0.1, b=1.0, tau=5.0, epochs=200)
    fast = fit_exponential(x, y)
    # tau=5 -> within 5% of the asymptote after ~3*tau = 15 epochs.
    assert fast.settle_x(200.0) == pytest.approx(5.0 * math.log(20), rel=1e-3)

    x, y = _exp_curve(a=0.1, b=1.0, tau=400.0, epochs=200)
    slow = fit_exponential(x, y)
    # tau=400 -> nowhere near settled inside a 200-epoch run.
    assert slow.settle_x(200.0) is None


def test_constant_series_is_reported_as_degenerate_not_as_a_lucky_fit():
    x = np.arange(50, dtype=float)
    fit = fit_exponential(x, np.full_like(x, 0.42))

    assert fit.degenerate is True
    assert fit.params["b"] == 0.0
    assert fit.asymptote == pytest.approx(0.42)
    assert fit.r2 == 1.0
    # A flat curve has already settled -- at its first epoch.
    assert fit.settle_x(49.0) == pytest.approx(0.0)


def test_linear_fit_is_offered_as_the_no_trend_baseline():
    x = np.arange(20, dtype=float)
    fit = fit_linear(x, 3.0 + 0.5 * x)

    assert fit.params["a"] == pytest.approx(3.0)
    assert fit.params["b"] == pytest.approx(0.5)
    assert math.isnan(fit.asymptote)          # a line has no limit
    assert fit.settle_x(19.0) is None         # ... so it never settles
    assert fit_linear(x, np.full_like(x, 2.0)).settle_x(19.0) == pytest.approx(0.0)


def test_fits_are_made_in_shifted_coordinates_so_epoch_0_can_be_dropped():
    # Fitting from epoch 1 onwards must not shift the curve by one epoch.
    x, y = _exp_curve(a=0.07, b=0.5, tau=12.0)
    fit = fit_exponential(x[1:], y[1:])

    predicted = fit.predict(x[1:])
    assert np.allclose(predicted, y[1:], atol=1e-5)
    assert fit.x0 == 1.0


def test_convergence_summary_separates_the_fit_from_the_evidence():
    x, y = _exp_curve(a=0.5, b=2.0, tau=10.0)
    summary = describe_convergence(x, y, fit_exponential(x, y))

    assert summary["converged"] is True
    assert summary["tail_drift"] == pytest.approx(0.0, abs=1e-6)
    assert summary["last_value"] == pytest.approx(0.5, abs=1e-6)

    # A curve that is still climbing at the end is NOT converged, however well
    # the model fits it.
    x = np.arange(201, dtype=float)
    rising = 0.1 + 0.002 * x
    summary = describe_convergence(x, rising, fit_linear(x, rising))
    assert summary["converged"] is False
    assert summary["tail_drift"] > 0.05


def test_fit_series_dispatches_by_name_and_rejects_unknown_models():
    x, y = _exp_curve(a=1.0, b=1.0, tau=10.0)

    assert fit_series(x, y, model="exp").model == "exp"
    assert fit_series(x, y, model="linear").model == "linear"
    with pytest.raises(ValueError, match="Unknown fit model"):
        fit_series(x, y, model="quadratic")


def test_fit_curve_values_lines_up_with_the_x_it_was_given():
    x, y = _exp_curve(a=0.3, b=1.0, tau=15.0, epochs=10)
    values = fit_curve_values(x, fit_exponential(x, y))

    assert len(values) == len(x)
    assert all(isinstance(v, float) for v in values)


def test_too_short_or_mismatched_input_is_a_clear_error():
    with pytest.raises(ValueError, match="at least 3 points"):
        fit_exponential([0.0, 1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="same length"):
        fit_exponential([0.0, 1.0, 2.0], [1.0, 2.0])


def test_a_decay_faster_than_a_tenth_of_the_run_is_not_pinned_to_the_search_edge():
    """The regression that motivated the neighbour-bracket scan.

    D*'s population mean loses ~99% of its value within two epochs, i.e. tau ~ 1
    on a 200-epoch run. A search bracket starting at span/100 could not reach
    that, so three unrelated models all came back with the SAME tau -- the
    boundary's value, not their own.
    """
    x, y = _exp_curve(a=13.0, b=200.0, tau=1.05, epochs=200)

    fit = fit_exponential(x, y)

    assert fit.params["tau"] == pytest.approx(1.05, rel=1e-3)
    assert fit.at_bound is False


def test_the_scan_finds_the_same_tau_as_an_exhaustive_search():
    rng = np.random.default_rng(3)
    x, y = _exp_curve(a=0.4, b=1.2, tau=3.7, epochs=200)
    y = y + rng.normal(0.0, 0.01, size=y.shape)

    fit = fit_exponential(x, y)
    grid = np.geomspace(0.05, 2000, 20_000)
    sse = [_solve_linear_part(x - x[0], y, float(tau))[1] for tau in grid]

    assert fit.params["tau"] == pytest.approx(float(grid[int(np.argmin(sse))]), rel=1e-3)


def test_a_decay_finer_than_the_sampling_is_reported_as_a_bound():
    # Gone between two consecutive samples: tau is unresolvable, and the fit says
    # so instead of presenting the floor as a measurement.
    x, y = _exp_curve(a=1.0, b=50.0, tau=0.01, epochs=50)

    fit = fit_exponential(x, y)

    assert fit.at_bound is True
    assert fit.params["tau"] <= 0.11


def test_constant_model_is_the_floor_every_trend_has_to_clear():
    rng = np.random.default_rng(7)
    x = np.arange(200, dtype=float)
    noise_only = 0.5 + rng.normal(0.0, 0.01, size=x.shape)

    fit = fit_series(x, noise_only, model="const")
    assert fit.params["a"] == pytest.approx(0.5, abs=0.01)
    assert fit.asymptote == pytest.approx(fit.params["a"])

    # On pure noise around a level, no trend model may be PREFERRED over the null
    # one -- otherwise the comparison would endorse a trend in data that has
    # none. A tiny spurious slope can still edge ahead by a fraction of an AIC
    # unit, which is exactly what "delta < 2 means indistinguishable" is for.
    scores = compare_models(x, noise_only, models=("const", "linear", "exp"))
    assert scores["const"].delta_aic < 2.0
    assert scores["exp"].delta_aic > scores["const"].delta_aic


def test_exp0_pins_the_asymptote_so_the_pair_tests_decay_to_zero():
    # Noise is deliberate: with a noiseless curve both models fit to ~1e-21 and
    # AIC ends up comparing floating-point dust rather than explanatory power.
    rng = np.random.default_rng(11)
    x = np.arange(201, dtype=float)
    noise = rng.normal(0.0, 0.005, size=x.shape)
    to_zero = 2.0 * np.exp(-x / 20.0) + noise
    to_plateau = 0.4 + 2.0 * np.exp(-x / 20.0) + noise

    # A curve that really does vanish: pinning a=0 costs no fit, so the cheaper
    # model is not beaten by the free asymptote.
    scores = compare_models(x, to_zero, models=("exp0", "exp"))
    assert scores["exp0"].delta_aic < 2.0
    assert fit_series(x, to_zero, model="exp0").asymptote == 0.0

    # A curve that settles above zero: the free asymptote is worth its parameter
    # many times over. This is the comparison that makes "settles at a positive
    # level" a measurement rather than a reading of the plot.
    scores = compare_models(x, to_plateau, models=("exp0", "exp"))
    assert scores["exp"].delta_aic == 0.0
    assert scores["exp0"].delta_aic > 10
    assert fit_series(x, to_plateau, model="exp").asymptote == pytest.approx(0.4, abs=0.01)


def test_asymptote_stability_is_measured_against_the_shorter_run():
    """The limit must be reported with how much it moves as the run grows.

    A curve still drifting at the end gives a systematically different asymptote
    when only its first half is used -- the number that turns "do not
    extrapolate" from a hedge into a measurement.
    """
    x = np.arange(1, 201, dtype=float)
    # Settles to 1.0 quickly, then keeps sliding downwards.
    y = 1.0 + 2.0 * np.exp(-(x - 1) / 10.0) - 0.0005 * x

    scores = compare_models(x, y, models=("exp",), holdout=0.5)
    score = scores["exp"]

    assert score.asymptote == pytest.approx(1.0, abs=0.1)
    # Seeing only the first half, the drift has not accumulated yet, so the
    # estimated limit sits HIGHER than the full-run one.
    assert score.asymptote_train_only > score.asymptote
    assert score.asymptote_shift_pct > 0


def test_models_without_a_limit_report_no_asymptote_rather_than_zero():
    x = np.arange(1, 101, dtype=float)
    y = 3.0 - 0.01 * x

    scores = compare_models(x, y, models=("linear", "exp"))

    assert math.isnan(scores["linear"].asymptote)
    assert math.isnan(scores["linear"].asymptote_shift_pct)
