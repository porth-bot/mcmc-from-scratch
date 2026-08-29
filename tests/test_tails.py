"""Heavy tails: the predictions, and an oracle that shares no code with them.

``mcmc/tails.py`` makes two closed-form claims about a Student-t -- the rate at
which the sample mean converges, and the rate at which the plug-in sd grows --
and computes one ground truth by quadrature. Each gets a check that could fail:

1. **The quadrature against an antiderivative.** At dof = 1 the Student-t is a
   Cauchy, whose CDF is ``1/2 + arctan(x)/pi`` in closed form, so every interval
   probability there has an exact value the quadrature must reproduce. At
   dof = 3 the CDF is elementary too. Neither shares a line with the quadrature.
2. **The exponents against sampling**, on exact draws, at a dof in each of the
   three regimes -- the only way to find out whether the regime boundaries are
   in the right places.
3. **The algebraic identity the honest-interval claim rests on**: the plug-in
   half-width ``s/sqrt(n)`` inherits exponent ``sd - 1/2``, and that must equal
   the mean's own error exponent, or the interval would not track the truth.
"""

import math

import numpy as np
import pytest

from mcmc.tails import (
    coverage,
    interval_probability,
    log_log_slope,
    mean_error_exponent,
    plug_in_interval,
    sample_sd_exponent,
)
from mcmc.targets import StudentT


def cauchy_cdf(x):
    """Exact t_1 CDF, from the antiderivative rather than by quadrature."""
    return 0.5 + math.atan(x) / math.pi


def t3_cdf(x):
    """Exact t_3 CDF (Student's own elementary form for odd dof = 3)."""
    z = x / math.sqrt(3.0)
    return 0.5 + (math.atan(z) + z / (1.0 + z * z)) / math.pi


@pytest.mark.parametrize("lo,hi", [(-1.0, 1.0), (0.0, 2.5), (-4.0, -0.5)])
def test_quadrature_matches_the_closed_form_cdfs(lo, hi):
    assert interval_probability(1.0, lo, hi) == pytest.approx(
        cauchy_cdf(hi) - cauchy_cdf(lo), abs=1e-12
    )
    assert interval_probability(3.0, lo, hi) == pytest.approx(
        t3_cdf(hi) - t3_cdf(lo), abs=1e-12
    )


def test_the_quadrature_is_converged_at_the_default_node_count():
    """If it were not, the "exact ground truth" framing would be wrong."""
    for dof in (1.0, 1.5, 7.0):
        coarse = interval_probability(dof, -1.0, 1.0, n_nodes=40)
        fine = interval_probability(dof, -1.0, 1.0, n_nodes=400)
        assert coarse == pytest.approx(fine, abs=1e-13)


def test_the_exponents_have_the_regimes_in_the_right_places():
    assert mean_error_exponent(30.0) == -0.5
    assert mean_error_exponent(2.5) == -0.5
    assert mean_error_exponent(1.5) == pytest.approx(-1 / 3)
    assert mean_error_exponent(1.25) == pytest.approx(-0.2)
    # At and below the Cauchy the sample mean does not converge at all.
    assert mean_error_exponent(1.0) == 0.0
    assert mean_error_exponent(0.5) == 0.0
    # The boundary at 2 is where the variance appears, and it is approached
    # continuously from below.
    assert mean_error_exponent(1.999) == pytest.approx(-0.4997, abs=1e-3)


def test_the_half_widths_exponent_is_the_means_own_error_exponent():
    """Why the plug-in interval stays honest: s/sqrt(n) tracks the truth.

    This identity is the entire content of the "coverage holds, width does not
    shrink" result -- if it failed the interval would be either anticonservative
    or needlessly wide, rather than correctly rate-matched and uninformative.
    """
    for dof in (1.0, 1.25, 1.5, 1.9, 2.5, 5.0, 30.0):
        assert sample_sd_exponent(dof) - 0.5 == pytest.approx(
            mean_error_exponent(dof)
        )


def test_log_log_slope_recovers_an_exact_power_law_and_rejects_zeros():
    ns = np.array([10.0, 100.0, 1000.0])
    assert log_log_slope(ns, 3.0 * ns**-0.5) == pytest.approx(-0.5)
    assert log_log_slope(ns, 2.0 * ns**0.25) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        log_log_slope(ns, np.array([1.0, 0.0, 1.0]))


@pytest.mark.parametrize("dof,tol", [(1.0, 0.06), (1.5, 0.06), (5.0, 0.05)])
def test_the_measured_rates_match_the_predicted_ones_on_exact_draws(dof, tol):
    """The prediction, scored against sampling in all three regimes."""
    rng = np.random.default_rng(11)
    target = StudentT([0.0], [[1.0]], dof)
    ns = (500, 4000, 32000)
    errors, sds = [], []
    for n in ns:
        x = target.sample(n * 300, rng).reshape(300, n)
        errors.append(np.median(np.abs(x.mean(axis=1))))
        sds.append(np.median(x.std(axis=1, ddof=1)))

    assert log_log_slope(ns, errors) == pytest.approx(
        mean_error_exponent(dof), abs=tol
    )
    assert log_log_slope(ns, sds) == pytest.approx(sample_sd_exponent(dof), abs=tol)


def test_a_bounded_functional_keeps_its_root_n_rate_where_the_mean_loses_it():
    """The contrast the experiment's design rests on, at the worst dof."""
    rng = np.random.default_rng(12)
    target = StudentT([0.0], [[1.0]], 1.0)          # Cauchy
    truth = interval_probability(1.0, -1.0, 1.0)
    ns = (500, 4000, 32000)
    prob_err = []
    for n in ns:
        x = target.sample(n * 300, rng).reshape(300, n)
        p_hat = np.mean(np.abs(x) <= 1.0, axis=1)
        prob_err.append(np.median(np.abs(p_hat - truth)))

    assert log_log_slope(ns, prob_err) == pytest.approx(-0.5, abs=0.06)


def test_the_plug_in_interval_covers_a_gaussian_and_a_cauchy_alike():
    """The negative that moved this section's headline.

    A ``mean +/- 1.96 s/sqrt(n)`` interval on draws with no variance ought to
    be a disaster and is not: the statistic is self-normalized, so it holds
    nominal coverage at dof = 1. Asserted here because it is the surprising
    half and the one a later edit would be tempted to soften.
    """
    rng = np.random.default_rng(13)
    n, replicates = 800, 3000

    gauss = rng.standard_normal((replicates, n))
    centre, half = plug_in_interval(gauss)
    assert coverage(centre, half, 0.0) == pytest.approx(0.95, abs=0.02)

    cauchy = StudentT([0.0], [[1.0]], 1.0).sample(n * replicates, rng)
    centre, half = plug_in_interval(cauchy.reshape(replicates, n))
    assert coverage(centre, half, 0.0) >= 0.95


def test_the_cauchy_interval_does_not_shrink_with_more_draws():
    """And the price of that coverage: 64x the data, no narrower."""
    rng = np.random.default_rng(14)
    widths = []
    for n in (250, 16000):
        x = StudentT([0.0], [[1.0]], 1.0).sample(n * 600, rng).reshape(600, n)
        widths.append(float(np.median(plug_in_interval(x)[1])))
    assert widths[0] / widths[1] == pytest.approx(1.0, abs=0.35)

    # A light-tailed control at the same budget shrinks by the sqrt(64) = 8x
    # it should, so the flat line above is a property of the tail, not of the
    # measurement.
    widths = []
    for n in (250, 16000):
        x = rng.standard_normal((600, n))
        widths.append(float(np.median(plug_in_interval(x)[1])))
    assert widths[0] / widths[1] == pytest.approx(8.0, rel=0.05)
