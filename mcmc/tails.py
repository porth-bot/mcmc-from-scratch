"""Heavy tails: what breaks, what does not, and which diagnostic sees it.

``targets.StudentT`` has been in this repo since the start as "the standard
heavy-tail cautionary target" and nothing measured it. This module is the
machinery for doing that, and the reason it is a module rather than a script is
that two of the three quantities below are *predictions*, derived here and
scored against sampling in ``experiments/heavy_tails.py``.

The three facts worth separating, because they are usually run together as
"MCMC mixes badly in heavy tails":

1. **The estimand's own rate.** For a symmetric Student-t with dof ``nu``, the
   tail index is ``nu``: P(|X| > x) ~ x^{-nu}. The ordinary CLT needs a finite
   variance, i.e. ``nu > 2``. For ``1 < nu < 2`` the sample mean still converges
   to ``mu``, but at the generalized-CLT rate ``n^{1/nu - 1}`` rather than
   ``n^{-1/2}`` -- n^{-1/3} at nu = 1.5 -- and at ``nu = 1`` (Cauchy) the sample
   mean of n draws is distributed exactly like a single draw and does not
   converge at all. None of that is a sampler property: it is true of perfect
   i.i.d. draws, which is why the experiment runs an i.i.d. arm as its control.

2. **The plug-in interval survives this, which is not the folklore.** One would
   expect ``mean +/- 1.96 s/sqrt(n)`` to under-cover badly when the variance it
   estimates does not exist. It does not: the statistic is *self-normalized*
   (Student's t-statistic), s is inflated by exactly the draws that inflate the
   numerator, and for symmetric heavy tails the ratio stays tight. The measured
   coverage is at or above nominal at every dof tested, the Cauchy included.
   What fails instead is the interval's *width*: since s grows like
   ``n^{1/nu - 1/2}`` for ``nu < 2``, the half-width ``s/sqrt(n)`` shrinks like
   ``n^{1/nu - 1}`` -- the true error rate of (1), so the interval is honest --
   and at ``nu = 1`` it does not shrink at all. The interval reports the trouble
   in its width, not in its coverage, and only if you watch it across n.

3. **The sampler's own failure, which is real and separate.** A bulk-scaled
   proposal under-visits the tails, so ESS per draw collapses. That is the
   failure ESS was built to see -- and it is blind to (1), which is the point:
   an i.i.d. arm has ESS = n by construction at every dof while its sample mean
   is converging at n^{-1/3} or not at all.

Ground truth for the bounded functional is exact to machine precision by
Gauss-Legendre quadrature of the 1-D density (see ``interval_probability``),
which is what lets the experiment score a statistic whose CLT never fails
against one whose CLT fails at nu <= 2.
"""

from __future__ import annotations

import math

import numpy as np


def mean_error_exponent(dof: float) -> float:
    """Exponent p in ``|mean_n - mu| ~ n^p`` for a symmetric Student-t.

    Three regimes, from the generalized central limit theorem applied to a
    tail index alpha = ``dof``:

    * ``dof > 2``  -- finite variance, ordinary CLT, ``p = -1/2``.
    * ``1 < dof < 2`` -- the sum is in the domain of attraction of an
      alpha-stable law with alpha = dof, so the sum of n draws grows like
      ``n^{1/alpha}`` and the *mean* error like ``n^{1/alpha - 1}``. At
      dof = 1.5 that is ``n^{-1/3}``; the rate degrades continuously to 0 as
      dof approaches 1 from above.
    * ``dof <= 1`` -- ``p = 0``. At dof = 1 the mean of n Cauchy draws is
      itself Cauchy with the same scale, so averaging buys nothing whatsoever;
      below 1 the mean does not exist and the sample mean diverges.

    Returns the exponent, not a rate: it is negative for a converging
    estimator, zero for one that does not converge.
    """
    if dof > 2.0:
        return -0.5
    if dof > 1.0:
        return 1.0 / dof - 1.0
    return 0.0


def sample_sd_exponent(dof: float) -> float:
    """Exponent in ``s_n ~ n^q`` for the plug-in sample standard deviation.

    For ``dof > 2`` the sd converges, so ``q = 0``. For ``dof < 2`` the
    population second moment is infinite and the sample second moment is itself
    in the domain of attraction of a stable law of index ``dof/2``: the sum of
    squares grows like ``n^{2/dof}``, the mean square like ``n^{2/dof - 1}``,
    and s like ``n^{1/dof - 1/2}``.

    The consequence is the one that makes the plug-in interval honest:
    ``s_n / sqrt(n)`` then scales like ``n^{1/dof - 1}``, which is exactly
    :func:`mean_error_exponent`. The interval tracks the true rate for free,
    and at dof = 1 both exponents are 0 -- a width that never shrinks, which is
    the correct report for an estimator that never converges.
    """
    if dof > 2.0:
        return 0.0
    return 1.0 / dof - 0.5


def interval_probability(dof: float, lo: float, hi: float, n_nodes: int = 200) -> float:
    """Exact ``P(lo <= X <= hi)`` for a standard 1-D Student-t, by quadrature.

    Gauss-Legendre on a *bounded* interval of a smooth, bounded integrand, so
    the error is at machine precision by ~50 nodes and this is a closed-form
    ground truth in every sense that matters here. Deliberately not a tail
    probability: the point of this functional in the experiment is that it is a
    bounded statistic whose CLT holds at every dof, which is what isolates the
    heavy-tail failure to the mean.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
    x = half * nodes + mid
    lognorm = (
        math.lgamma(0.5 * (dof + 1.0))
        - math.lgamma(0.5 * dof)
        - 0.5 * math.log(dof * math.pi)
    )
    density = np.exp(lognorm - 0.5 * (dof + 1.0) * np.log1p(x * x / dof))
    return float(half * np.sum(weights * density))


def plug_in_interval(x: np.ndarray, effective_n: np.ndarray | float | None = None,
                     z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """``(centre, half_width)`` of the usual ``mean +/- z s/sqrt(n_eff)``.

    ``x`` is (n_replicates, n_draws); the interval is computed per replicate so
    the experiment can measure coverage as a frequency over replicates rather
    than asserting it.

    ``effective_n`` defaults to the number of draws, which is right for the
    i.i.d. arm. An MCMC arm passes its ESS instead -- the substitution every
    practitioner makes, and the experiment measures whether it survives here.
    """
    x = np.atleast_2d(x)
    n_eff = x.shape[1] if effective_n is None else np.asarray(effective_n, dtype=float)
    centre = x.mean(axis=1)
    half = z * x.std(axis=1, ddof=1) / np.sqrt(n_eff)
    return centre, half


def coverage(centre: np.ndarray, half_width: np.ndarray, truth: float) -> float:
    """Fraction of replicate intervals containing ``truth``."""
    return float(np.mean(np.abs(np.asarray(centre) - truth) <= np.asarray(half_width)))


def log_log_slope(ns, values) -> float:
    """Fitted exponent of ``values ~ n^p``, the estimator every rate here uses.

    A plain least-squares fit of log(value) on log(n). Reported rather than
    rounded to the prediction: the finite-n approach to a stable limit is slow
    (the slowly varying correction is logarithmic), so the measured exponent
    sits a little short of the asymptote at dof close to 2, and saying so is
    more useful than quoting the theory back.
    """
    ns = np.asarray(ns, dtype=float)
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0):
        raise ValueError("log-log slope needs strictly positive values")
    return float(np.polyfit(np.log(ns), np.log(values), 1)[0])
