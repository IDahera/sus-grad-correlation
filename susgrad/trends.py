"""Fitting a trend to a per-epoch curve, and saying where it is heading.

The trajectory figures show a curve per model/metric and invite the eye to
decide "is this still moving?". The eye is a poor judge of that: the epoch-0
value is an extreme outlier, the noise band is wide, and D* spans seven orders
of magnitude. So the question is asked numerically here.

The default is **ordinary linear regression** (``linear``)::

    y(e) = a + b * (e - e0)

fitted by least squares -- the line whose squared vertical distances to the
points are smallest, which has a closed-form solution and needs no search. Its
slope ``b`` is the answer in the units of the question ("the mean falls by
1.1e-4 per epoch"), and it comes with a **95% confidence interval**: the points
scatter around the line, so the slope is an estimate, and the interval says how
much of it is real. An interval that contains zero is the honest way to write
"no trend distinguishable from flat" -- which is the statement a settled metric
deserves.

A line assumes a CONSTANT rate of change, and these curves fall steeply before
flattening. One line over the whole run therefore averages those two phases;
the fitting window (``--fit-from``) is what selects which phase is described,
and it is reported alongside every number.

``exp`` remains available for the same curves::

    y(e) = a + b * exp(-(e - e0) / tau)

whose ``a`` is an explicit limit and ``tau`` a rate. It fits the two-phase shape
far better, at the price of imposing that shape and of parameters that need
explaining; see :func:`fit_exponential`.

**No scipy.** The project computes its own Pearson/Spearman rather than pulling
in a statistics stack (``susgrad/correlation/metrics.py``), and the same applies
here: the t-distribution needed for the confidence interval is implemented from
the regularised incomplete beta function below, and the exponential's ``tau`` is
found by a deterministic scan instead of a non-linear solver -- no seeded starts,
no "fit failed" branch in a figure script.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: How close to the asymptote counts as "settled" (fraction of the fitted swing
#: ``|b|`` that must have been given up). 0.05 -> within 5% of the limit.
DEFAULT_TOLERANCE = 0.05

#: Fraction of the captured epochs treated as "the tail" when measuring drift.
DEFAULT_TAIL = 0.25

#: Confidence level for the slope interval.
DEFAULT_CONFIDENCE = 0.95

_EPS = 1e-300


# --- Student's t, from scratch ------------------------------------------------
# Needed for the slope's confidence interval and p-value. A normal
# approximation would be within a percent at n=200, but the same code has to be
# right for a short run (10 captured epochs -> df=8, where z=1.96 understates
# the interval by 18%), so the exact distribution is computed.

def _betacf(a: float, b: float, x: float, iterations: int = 200) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        # even step
        num = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + num * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + num / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        # odd step
        num = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + num * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + num / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)`` -- the CDF machinery for t."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b   # the mirrored series converges


def student_t_sf(t: float, df: float) -> float:
    """``P(T > t)`` for Student's t with *df* degrees of freedom (upper tail)."""
    if df <= 0:
        return float("nan")
    half = 0.5 * incomplete_beta(0.5 * df, 0.5, df / (df + t * t))
    return half if t > 0 else 1.0 - half


def student_t_ppf(p: float, df: float, *, tol: float = 1e-10) -> float:
    """Quantile of Student's t: the ``t`` with ``P(T <= t) = p``.

    Bisection on the (monotone) CDF. Slower than a closed form and entirely fast
    enough -- it is called twice per fit, not per data point.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}.")
    if p == 0.5:
        return 0.0
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - student_t_sf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class Fit:
    """One fitted trend, in the coordinates ``x - x0``.

    ``params`` holds the model's coefficients (``a``/``b``/``tau`` for the
    exponential, ``a``/``b`` for the line); :meth:`predict` is the only thing
    that needs to know which is which.
    """

    model: str
    params: Dict[str, float]
    x0: float
    r2: float
    rmse: float
    n: int
    #: Set when the series was constant (or too short) and no real fit was made.
    degenerate: bool = field(default=False)
    #: Set when ``tau`` came out at an end of the searched range -- the data want
    #: a decay faster (or slower) than this sampling can resolve, so the number
    #: is a bound, not a measurement.
    at_bound: bool = field(default=False)
    #: Standard error per parameter (linear fit only): how much the estimate
    #: would wobble if the run were repeated.
    stderr: Dict[str, float] = field(default_factory=dict)
    #: ``{parameter: (low, high)}`` confidence interval (linear fit only).
    ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    #: Two-sided p-value for "the slope is zero" (linear fit only).
    p_value: Optional[float] = None
    #: Confidence level behind :attr:`ci`.
    confidence: float = DEFAULT_CONFIDENCE

    def predict(self, x) -> np.ndarray:
        """The fitted curve evaluated at *x* (same units as the fitted x)."""
        t = np.asarray(x, dtype=np.float64) - self.x0
        if self.model in ("exp", "exp0"):
            return self.params["a"] + self.params["b"] * np.exp(-t / self.params["tau"])
        if self.model == "linear":
            return self.params["a"] + self.params["b"] * t
        if self.model == "const":
            return np.full(t.shape, self.params["a"], dtype=np.float64)
        raise ValueError(f"Unknown model {self.model!r}.")

    @property
    def asymptote(self) -> float:
        """The value the fit tends to, or NaN for a model without a limit."""
        if self.model in ("exp", "exp0", "const"):
            return float(self.params["a"])
        return float("nan")

    @property
    def tau(self) -> float:
        """Epochs to close ~63% of the remaining distance (exponential only)."""
        return float(self.params.get("tau", float("nan")))

    @property
    def half_life(self) -> float:
        """Epochs to halve the distance to the asymptote (exponential only)."""
        return self.tau * math.log(2.0) if self.model == "exp" else float("nan")

    def settle_x(self, x_max: float, *, tol: float = DEFAULT_TOLERANCE) -> Optional[float]:
        """First x at which the fit is within *tol* of its asymptote.

        Returns ``None`` if that never happens inside ``x0..x_max`` -- which is
        the useful answer for a curve that is still moving when the run ends.

        A line has no asymptote, so "settled" is read as "it never moved much":
        it counts as settled from ``x0`` if the total change over the run is
        within *tol* of the value it ends at, and never otherwise.
        """
        if self.model == "const":
            return self.x0
        if self.model == "linear":
            end = self.params["a"] + self.params["b"] * (x_max - self.x0)
            swing = abs(self.params["b"]) * (x_max - self.x0)
            scale = abs(end) or 1.0
            return self.x0 if swing <= tol * scale else None
        b, tau = abs(self.params["b"]), self.params["tau"]
        if b <= _EPS:
            return self.x0
        if not math.isfinite(tau) or tau <= 0:
            return None
        # |b| e^{-t/tau} <= tol * |b|  ->  t >= tau * ln(1/tol).
        t = tau * math.log(1.0 / tol)
        x = self.x0 + t
        return x if x <= x_max else None

    def describe(self) -> str:
        """One-line summary for a log or a table cell."""
        if self.model == "exp":
            return (f"y = {self.params['a']:.4g} + {self.params['b']:.4g}·exp(-(e-{self.x0:g})"
                    f"/{self.params['tau']:.3g})   R²={self.r2:.3f}")
        text = (f"y = {self.params['a']:.4g} + {self.params['b']:.3g}·(e-{self.x0:g})"
                f"   R²={self.r2:.3f}")
        if "b" in self.ci:
            lo, hi = self.ci["b"]
            text += (f"   slope {self.confidence:.0%} CI [{lo:.3g}, {hi:.3g}]"
                     f"   p={self.p_value:.3g}")
        return text


def _quality(y: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    resid = y - predicted
    sse = float((resid**2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return {
        "r2": 1.0 - sse / sst if sst > 0 else 1.0,
        "rmse": math.sqrt(sse / len(y)) if len(y) else 0.0,
    }


def _solve_linear_part(t: np.ndarray, y: np.ndarray, tau: float, basis_kind: str = "free"):
    """Least-squares coefficients for a fixed ``tau``, plus the residual sum.

    ``basis_kind='free'`` fits ``a + b·exp(-t/tau)``; ``'zero'`` drops the
    constant column, which nails the asymptote to zero.
    """
    decay = np.exp(-t / tau)
    basis = np.column_stack([np.ones_like(t), decay]) if basis_kind == "free" \
        else decay.reshape(-1, 1)
    coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
    sse = float(((basis @ coeffs - y) ** 2).sum())
    return coeffs, sse


def fit_exponential(x, y, *, coarse: int = 400, refinements: int = 3, _basis: str = "free") -> Fit:
    """Fit ``y = a + b·exp(-(x - x[0])/tau)`` by scanning ``tau``.

    ``tau`` is the only non-linearly entering parameter, so for each candidate
    ``(a, b)`` follow in closed form (ordinary least squares on the design matrix
    ``[1, exp(-t/tau)]``) and the whole fit reduces to picking the ``tau`` with
    the smallest residual sum.

    The scan is geometric -- ``tau`` is a scale, so equal *ratios* deserve equal
    attention -- and spans a tenth of the sampling interval up to ten times the
    run: below that a decay is a step between two samples, above it a straight
    line. Each refinement re-scans the bracket between the winner's two
    NEIGHBOURS on the grid, which both contains the true minimum (the residual is
    smooth in log tau) and shrinks the search by the previous step size, so the
    resolution multiplies per round rather than stalling.

    A minimum sitting on an end of the very first grid is reported through
    ``Fit.at_bound``: the data want a decay this sampling cannot resolve, so the
    returned ``tau`` is a bound rather than a measurement.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"x and y must be the same length ({x.size} vs {y.size}).")
    if x.size < 3:
        raise ValueError("An exponential fit needs at least 3 points.")

    x0 = float(x[0])
    t = x - x0
    span = float(t[-1] - t[0]) or 1.0

    # "Flat" has to be judged relatively: a constant series built by arithmetic
    # (a dead layer's mean, say) carries float noise around its own value, and
    # an absolute ``variance == 0`` test would miss it and report a meaningless
    # tau fitted to that noise.
    model_name = "exp" if _basis == "free" else "exp0"
    scale = float(np.abs(y).max())
    if float(y.max() - y.min()) <= 1e-12 * max(scale, 1e-30) and _basis == "free":
        return Fit(model=model_name, params={"a": float(y.mean()), "b": 0.0, "tau": float("inf")},
                   x0=x0, r2=1.0, rmse=0.0, n=int(y.size), degenerate=True)

    # The resolvable range of decay constants, from the data itself: half a
    # sampling step at the fast end (D* loses 99.9% of its value in the first two
    # captured epochs, so the bracket MUST reach below one epoch -- a floor of
    # span/100 silently pinned those fits to the boundary and returned the same
    # tau for unrelated models), ten runs at the slow end.
    spacing = float(np.min(np.diff(t))) if t.size > 1 else 1.0
    lo, hi = max(spacing / 10.0, span * 1e-9), span * 10.0
    at_bound = False
    for round_index in range(refinements + 1):
        grid = np.geomspace(lo, hi, coarse)
        solved = [_solve_linear_part(t, y, float(tau), _basis) for tau in grid]
        index = int(np.argmin([sse for _, sse in solved]))
        if round_index == 0:
            at_bound = index in (0, coarse - 1)
        coeffs = solved[index][0]
        tau = float(grid[index])
        # Bracket the winner's neighbours: the residual is smooth in log tau, so
        # the true minimum lies between them.
        lo, hi = float(grid[max(index - 1, 0)]), float(grid[min(index + 1, coarse - 1)])
        if lo == hi:
            break

    params = ({"a": float(coeffs[0]), "b": float(coeffs[1]), "tau": tau} if _basis == "free"
              else {"a": 0.0, "b": float(coeffs[0]), "tau": tau})   # 'a' pinned, not fitted
    predicted = params["a"] + params["b"] * np.exp(-t / params["tau"])
    return Fit(model=model_name, params=params, x0=x0, n=int(y.size), at_bound=at_bound,
               **_quality(y, predicted))


def fit_linear(x, y, *, confidence: float = DEFAULT_CONFIDENCE) -> Fit:
    """Fit ``y = a + b·(x - x[0])`` by ordinary least squares, with inference.

    The line minimising the summed squared vertical distances to the points --
    solved in closed form, no search involved. Beyond the two coefficients it
    reports what makes a slope quotable:

    * ``stderr["b"]`` -- how far the points typically miss the line, scaled by
      how spread out the epochs are. The scale on which a slope counts as small.
    * ``ci["b"]`` -- the *confidence* interval for the slope. If it contains
      zero, the data cannot distinguish this curve from a flat one, and that is
      the sentence to write instead of quoting the slope.
    * ``p_value`` -- the same statement as a probability: how easily this slope
      would arise from a genuinely flat curve.

    With 200 epochs even a tiny slope becomes "significant", so the interval is
    reported in the values' own units and read next to the level of the curve --
    statistical detectability is not the same as mattering.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"x and y must be the same length ({x.size} vs {y.size}).")
    if x.size < 2:
        raise ValueError("A linear fit needs at least 2 points.")

    x0 = float(x[0])
    t = x - x0
    slope, intercept = np.polyfit(t, y, 1)
    params = {"a": float(intercept), "b": float(slope)}
    predicted = params["a"] + params["b"] * t
    quality = _quality(y, predicted)

    n = int(y.size)
    df = n - 2
    sum_sq_x = float(((t - t.mean()) ** 2).sum())
    stderr, ci, p_value = {}, {}, None
    if df > 0 and sum_sq_x > 0:
        residual_var = float(((y - predicted) ** 2).sum()) / df
        se_b = math.sqrt(residual_var / sum_sq_x)
        se_a = math.sqrt(residual_var * (1.0 / n + t.mean() ** 2 / sum_sq_x))
        critical = student_t_ppf(0.5 * (1.0 + confidence), df)
        stderr = {"a": se_a, "b": se_b}
        ci = {
            "a": (params["a"] - critical * se_a, params["a"] + critical * se_a),
            "b": (params["b"] - critical * se_b, params["b"] + critical * se_b),
        }
        p_value = (2.0 * student_t_sf(abs(params["b"] / se_b), df)
                   if se_b > 0 else (1.0 if params["b"] == 0 else 0.0))

    return Fit(model="linear", params=params, x0=x0, n=n, stderr=stderr, ci=ci,
               p_value=p_value, confidence=confidence, **quality)


def fit_constant(x, y) -> Fit:
    """Fit ``y = a`` -- the "nothing is happening" null model.

    The floor every other model has to clear. If a curve's variation is pure
    epoch-to-epoch noise around a fixed level, this two-parameter model (the
    level and the noise) wins on any information criterion, and no trend should
    be reported at all.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"x and y must be the same length ({x.size} vs {y.size}).")
    if y.size < 1:
        raise ValueError("A constant fit needs at least 1 point.")

    params = {"a": float(y.mean())}
    predicted = np.full(y.shape, params["a"])
    return Fit(model="const", params=params, x0=float(x[0]), n=int(y.size),
               **_quality(y, predicted))


def fit_exponential_to_zero(x, y, **kwargs) -> Fit:
    """Fit ``y = b·exp(-(x - x[0])/tau)`` -- decay all the way to ZERO.

    The same curve as :func:`fit_exponential` with its asymptote nailed to zero,
    which makes the pair a direct test of a claim that is otherwise only
    asserted: *does this quantity decay to nothing, or to a positive level?*
    Comparing the two by AIC answers it with the free asymptote's extra
    parameter already paid for -- so "it settles above zero" becomes a measured
    statement instead of a reading of the plot.
    """
    fit = fit_exponential(x, y, _basis="zero", **kwargs)
    return fit


#: Registry so a CLI can offer the models by name (and log which was used).
#: Ordered from the simplest upwards, which is the order a comparison table
#: reads best in.
FIT_MODELS = {
    "const": fit_constant,
    "linear": fit_linear,
    "exp0": fit_exponential_to_zero,
    "exp": fit_exponential,
}


def fit_series(x, y, *, model: str = "exp") -> Fit:
    """Fit *y* over *x* with the named model from :data:`FIT_MODELS`."""
    if model not in FIT_MODELS:
        raise ValueError(f"Unknown fit model {model!r}; choose from {sorted(FIT_MODELS)}.")
    return FIT_MODELS[model](x, y)


def describe_convergence(
    x,
    y,
    fit: Fit,
    *,
    tol: float = DEFAULT_TOLERANCE,
    tail: float = DEFAULT_TAIL,
) -> Dict[str, object]:
    """Has this curve settled? -- the fitted answer and the empirical one.

    Two independent checks, because either alone is easy to fool:

    * **fitted** -- ``settle_epoch``, the epoch from which the fitted curve stays
      within *tol* of its asymptote (``None`` if that is beyond the last
      captured epoch), plus ``tau`` and the asymptote itself.
    * **empirical** -- ``tail_drift``: how far the mean of the last *tail* of the
      run moved between its first and second half, as a fraction of the curve's
      full range. A fit can claim an asymptote it never reaches; a drift near
      zero is direct evidence that it did.

    ``converged`` requires both: settled inside the run AND a tail drift within
    *tol*.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    settle = fit.settle_x(float(x[-1]), tol=tol)
    n_tail = max(2, int(round(len(y) * tail)))
    window = y[-n_tail:]
    half = len(window) // 2
    span = float(y.max() - y.min())
    drift = abs(float(window[half:].mean() - window[:half].mean()))
    tail_drift = drift / span if span > 0 else 0.0

    return {
        "model": fit.model,
        "r2": fit.r2,
        "rmse": fit.rmse,
        "asymptote": fit.asymptote,
        "tau": fit.tau,
        "half_life": fit.half_life,
        "settle_epoch": settle,
        "tail_epochs": n_tail,
        "tail_mean": float(window.mean()),
        "tail_std": float(window.std()),
        "tail_drift": tail_drift,
        "first_value": float(y[0]),
        "last_value": float(y[-1]),
        "converged": bool(settle is not None and tail_drift <= tol),
    }


def fit_curve_values(x, fit: Fit) -> Sequence[float]:
    """The fitted curve sampled at *x*, as a plain list (for CSVs and plots)."""
    return [float(v) for v in fit.predict(x)]


# --- justifying the model, rather than asserting it --------------------------------
#
# A curve that "looks exponential" and a fitted exponential drawn over it are a
# circular argument: the reader sees the model, not the evidence for it. The
# three functions below are the evidence -- a criterion that charges the model
# for its extra parameter, a prediction of data the fit never saw, and a check
# that what is left over is noise rather than the shape the model got wrong.

#: Free parameters per model, INCLUDING the residual variance -- the likelihood
#: estimates it too, and an information criterion has to pay for it.
_N_PARAMS = {"const": 2, "linear": 3, "exp0": 3, "exp": 4}


@dataclass(frozen=True)
class ModelScore:
    """How one candidate model did on one curve."""

    model: str
    n: int
    k: int
    sse: float
    r2: float
    rmse: float
    aic: float
    bic: float
    #: AIC minus the best model's AIC (0 for the winner). The usual reading:
    #: <2 means the models are indistinguishable, >10 that the worse one has
    #: essentially no support.
    delta_aic: float = 0.0
    #: RMSE on the held-out tail, from a fit that never saw it.
    holdout_rmse: float = float("nan")
    #: That error relative to the spread of the held-out data. Below 1 means the
    #: extrapolation beats "assume it stays at its average".
    holdout_ratio: float = float("nan")
    #: The asymptote from the full fit, and from the fit that saw only the first
    #: part of the run. NaN for models without a limit.
    asymptote: float = float("nan")
    asymptote_train_only: float = float("nan")
    #: How far the estimate moved when the second half of the run was added, as a
    #: percentage of the full-run value. A limit that keeps sliding as the run
    #: gets longer is not a property of the model but of when you stopped
    #: looking -- which is worth a number rather than a hedge.
    asymptote_shift_pct: float = float("nan")


def _information_criteria(n: int, sse: float, k: int):
    """AIC and BIC for a least-squares fit with Gaussian errors."""
    # For a Gaussian likelihood with the variance profiled out,
    # -2 log L = n * ln(SSE / n) + const, and the constant cancels in every
    # comparison of models on the SAME data -- which is the only comparison
    # these numbers are used for.
    log_likelihood_term = n * math.log(max(sse, _EPS) / n)
    return log_likelihood_term + 2 * k, log_likelihood_term + k * math.log(n)


def compare_models(x, y, models=("exp", "linear"), *, holdout: float = 0.5) -> Dict[str, ModelScore]:
    """Score every candidate model on the same curve, in-sample and out.

    Returns ``{model: ModelScore}``. Two independent verdicts:

    * **AIC / BIC** -- fit quality with an explicit charge per parameter, so the
      three-parameter exponential does not win merely by being more flexible.
      Only differences matter, and only between models fitted to the same data.
    * **Held-out error** -- each model is refitted on the first ``1 - holdout``
      of the epochs and asked to predict the rest. This is the test that matters
      for "where do the values point": extrapolation is exactly what a claim
      about a limit does, so a model that cannot predict the second half of a run
      it has not seen should not be trusted to describe the run's destination.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    split = x[0] + (1.0 - holdout) * (x[-1] - x[0])
    train, test = x <= split, x > split

    scores = {}
    for model in models:
        fit = fit_series(x, y, model=model)
        residual = y - fit.predict(x)
        sse = float((residual**2).sum())
        k = _N_PARAMS[model]
        aic, bic = _information_criteria(len(y), sse, k)

        holdout_rmse = holdout_ratio = float("nan")
        asymptote_train_only = shift = float("nan")
        if test.sum() >= 2 and train.sum() >= 3:
            trained = fit_series(x[train], y[train], model=model)
            error = y[test] - trained.predict(x[test])
            holdout_rmse = float(np.sqrt((error**2).mean()))
            spread = float(y[test].std())
            holdout_ratio = holdout_rmse / spread if spread > 0 else float("nan")

            # The SAME fit, reused: how much did the estimated limit move when
            # the rest of the run was added? Free, and it turns "do not
            # extrapolate" into a measured percentage.
            asymptote_train_only = trained.asymptote
            if math.isfinite(fit.asymptote) and abs(fit.asymptote) > _EPS:
                shift = 100.0 * (asymptote_train_only - fit.asymptote) / abs(fit.asymptote)

        scores[model] = ModelScore(
            model=model, n=len(y), k=k, sse=sse, r2=fit.r2, rmse=fit.rmse,
            aic=aic, bic=bic, holdout_rmse=holdout_rmse, holdout_ratio=holdout_ratio,
            asymptote=fit.asymptote, asymptote_train_only=asymptote_train_only,
            asymptote_shift_pct=shift,
        )

    best = min(s.aic for s in scores.values())
    return {name: ModelScore(**{**score.__dict__, "delta_aic": score.aic - best})
            for name, score in scores.items()}


def residual_diagnostics(x, y, fit: Fit) -> Dict[str, float]:
    """Is what the model failed to explain noise, or the shape it got wrong?

    A line fitted to a curve leaves residuals that are systematically positive,
    then negative, then positive again -- the misfit is visible in the leftovers
    even when R² looks respectable. Two standard summaries of that:

    * **lag-1 autocorrelation** -- how much a residual resembles its neighbour.
      Near 0 for noise; close to 1 when the fit is riding above or below the
      data for long stretches.
    * **runs-test z** -- residuals should change sign about as often as coin
      flips. Far fewer runs than expected (a strongly negative z) means long
      same-sign stretches, i.e. structure.

    ``structured`` is the blunt verdict: |z| > 2, the usual 5% threshold.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    residual = y - fit.predict(x)

    centred = residual - residual.mean()
    denominator = float((centred**2).sum())
    lag1 = float((centred[:-1] * centred[1:]).sum() / denominator) if denominator > 0 else 0.0

    signs = np.sign(residual)
    signs = signs[signs != 0]
    n_pos = int((signs > 0).sum())
    n_neg = int((signs < 0).sum())
    runs = int(1 + (signs[1:] != signs[:-1]).sum()) if signs.size else 0
    total = n_pos + n_neg
    runs_z = 0.0
    if n_pos and n_neg and total > 1:
        expected = 2.0 * n_pos * n_neg / total + 1.0
        variance = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - total)) / (total**2 * (total - 1.0))
        runs_z = (runs - expected) / math.sqrt(variance) if variance > 0 else 0.0

    return {
        "lag1_autocorr": lag1,
        "runs": float(runs),
        "runs_z": runs_z,
        "resid_sd": float(residual.std()),
        "max_abs_resid": float(np.abs(residual).max()) if residual.size else 0.0,
        "structured": bool(abs(runs_z) > 2.0),
    }


def bootstrap_parameter_ci(
    x,
    y,
    fit: Fit,
    parameter: str = "a",
    *,
    resamples: int = 300,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile confidence interval for one fitted parameter, by resampling.

    The exponential's asymptote has no closed-form standard error (it is a
    non-linear fit), and quoting a limit without one invites the reader to take
    four decimals seriously. Residual bootstrap: keep the fitted curve, resample
    the residuals with replacement, refit, and report the middle *confidence* of
    the resulting parameter values.

    Residuals here are correlated between neighbouring epochs, which makes this
    interval somewhat optimistic; it is a scale, not a guarantee.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    baseline = fit.predict(x)
    residual = y - baseline
    rng = np.random.default_rng(seed)

    values = []
    for _ in range(resamples):
        resampled = baseline + rng.choice(residual, size=residual.size, replace=True)
        try:
            refit = fit_series(x, resampled, model=fit.model)
        except ValueError:
            continue
        if parameter in refit.params:
            values.append(refit.params[parameter])
    if not values:
        return float("nan"), float("nan")

    tail = 0.5 * (1.0 - confidence)
    return (float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail)))


def window_levels(x, y, *, windows: int = 4) -> list:
    """The curve's level in successive equal windows -- convergence without a model.

    The strongest evidence for "this has settled" needs no fitted function at
    all: cut the run into consecutive windows and report each one's mean and
    spread. If the last windows agree to within their own noise, the values have
    stopped moving in any sense a reader can dispute; if they keep stepping down,
    no fitted asymptote should be believed over that.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    edges = np.linspace(x[0], x[-1], windows + 1)

    out = []
    for index in range(windows):
        lo, hi = edges[index], edges[index + 1]
        mask = (x >= lo) & (x <= hi) if index == windows - 1 else (x >= lo) & (x < hi)
        if not mask.any():
            continue
        chunk = y[mask]
        out.append({
            "start": float(lo), "end": float(hi), "n": int(mask.sum()),
            "mean": float(chunk.mean()), "std": float(chunk.std()),
            "median": float(np.median(chunk)),
        })
    for previous, current in zip(out, out[1:]):
        # How far this window moved, measured in the PREVIOUS window's own
        # noise: below ~1 the step is indistinguishable from the scatter.
        step = abs(current["mean"] - previous["mean"])
        current["step_in_sd"] = step / previous["std"] if previous["std"] > 0 else float("inf")
        current["step_relative"] = step / abs(previous["mean"]) if previous["mean"] else float("inf")
    return out
