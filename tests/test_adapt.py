"""Estimating the metric from warmup draws (Sec. 4.11).

Three claims carry this module, and each one fails quietly if it is wrong.

1.  The refactor is not a change. ``hmc.py``'s diagonal adaptation used to be
    four inline lines; it now calls ``WindowMoments`` and ``stan_shrink``. Every
    diagonal-metric number committed to this repo was produced by the inline
    version, so the replacement is asserted *equal*, not close.
2.  The shrinkage does the thing it exists for. A raw window covariance at
    n <= d is singular and is not a metric; both regularizers must produce
    something ``DenseMetric`` accepts, at every window size the schedule emits.
3.  Ledoit-Wolf is the estimator it claims to be, not merely a matrix in the
    right shape -- the intensity has to be right, which is checked against the
    loss it minimizes rather than against a hard-coded number.
"""

import numpy as np
import pytest

from mcmc.adapt import (
    WindowMoments,
    estimate_covariance,
    ledoit_wolf,
    stan_shrink,
    whitened_condition_number,
)
from mcmc.hmc import hmc
from mcmc.metric import DenseMetric
from mcmc.targets import Gaussian


def ar1(d, rho, sd=1.0):
    """AR(1) covariance: Sigma_ij = sd^2 rho^|i-j|. Strongly correlated,
    positive definite for any |rho| < 1, and its condition number grows like
    ((1+rho)/(1-rho))^1 -- a target the diagonal metric provably cannot fix."""
    i = np.arange(d)
    return (sd**2) * rho ** np.abs(i[:, None] - i[None, :])


# -- 1. the refactor changes nothing ----------------------------------------

def test_window_variance_equals_the_inline_expression_exactly():
    rng = np.random.default_rng(0)
    dim, n_chains = 5, 4
    w = WindowMoments(dim, mode="diag")
    acc_n, acc_sum, acc_sumsq = 0, np.zeros(dim), np.zeros(dim)
    for _ in range(37):
        x = rng.standard_normal((n_chains, dim)) * np.array([1.0, 3.0, 0.1, 7.0, 1.0])
        w.add(x)
        acc_n += n_chains
        acc_sum += x.sum(axis=0)
        acc_sumsq += np.sum(x * x, axis=0)
    mean = acc_sum / acc_n
    var = acc_sumsq / acc_n - mean * mean
    assert w.n == acc_n
    # exactly, not np.allclose: a one-ulp change moves a leapfrog trajectory
    assert np.array_equal(w.variance(), var)

    inline = (acc_n / (acc_n + 5.0)) * var + 1e-3 * (5.0 / (acc_n + 5.0))
    assert np.array_equal(stan_shrink(var, acc_n), inline)


def test_reset_makes_the_window_memoryless():
    rng = np.random.default_rng(1)
    w = WindowMoments(3, mode="diag")
    w.add(rng.standard_normal((4, 3)) * 50.0)      # a "pre-convergence" burst
    w.reset()
    fresh = rng.standard_normal((4, 3))
    w.add(fresh)
    assert w.n == 4
    assert np.allclose(w.variance(), fresh.var(axis=0))


def test_dense_window_covariance_matches_numpy():
    rng = np.random.default_rng(2)
    w = WindowMoments(4, mode="dense")
    blocks = [rng.standard_normal((3, 4)) for _ in range(20)]
    for b in blocks:
        w.add(b)
    x = np.concatenate(blocks, axis=0)
    assert w.n == x.shape[0]
    assert np.allclose(w.covariance(), np.cov(x.T, bias=True))
    # and its diagonal is the variance the diagonal mode would have reported
    assert np.allclose(np.diag(w.covariance()), x.var(axis=0))


def test_modes_reject_the_other_mode_s_accessor():
    with pytest.raises(ValueError, match="mode='diag'"):
        WindowMoments(2, mode="dense").variance()
    with pytest.raises(ValueError, match="mode='dense'"):
        WindowMoments(2, mode="diag").covariance()
    with pytest.raises(ValueError, match="diag"):
        WindowMoments(2, mode="full")


# -- 2. what the shrinkage is for -------------------------------------------

def test_a_rank_deficient_window_fails_as_a_metric_but_not_reliably_loudly():
    """The premise of the whole module -- and the unpleasant half of it.

    A window of n <= d draws gives a covariance of rank at most n - 1, so it is
    not a metric. What it does about that depends on the sign of a roundoff
    error. At n = d here the smallest eigenvalue lands at +2e-16, Cholesky
    succeeds, and ``DenseMetric`` cheerfully returns a metric conditioned at
    1e16; one draw earlier the same eigenvalue is -2e-16 and it raises. So the
    constructor's PSD check is a backstop and not a defense: the reason to
    regularize is that the unregularized failure is *sometimes silent*.
    """
    rng = np.random.default_rng(3)
    d = 12
    w = WindowMoments(d, mode="dense")
    for _ in range(3):                      # 3 x 4 chains = 12 draws, n = d
        w.add(rng.standard_normal((4, d)))
    S = w.covariance()
    assert np.linalg.matrix_rank(S) < d
    assert np.linalg.cond(S) > 1e12         # accepted, and useless

    w_short = WindowMoments(d, mode="dense")
    for _ in range(2):                      # n = 8 < d: the loud case
        w_short.add(rng.standard_normal((4, d)))
    with pytest.raises(ValueError, match="not positive definite"):
        DenseMetric(w_short.covariance())

    # both regularizers repair it, and by very different amounts: the Stan
    # ridge only makes it invertible, Ledoit-Wolf makes it well conditioned.
    stan = np.linalg.cond(estimate_covariance(w_short, shrinkage="stan"))
    lw = np.linalg.cond(estimate_covariance(w_short, shrinkage="ledoit-wolf"))
    assert 1e2 < stan < 1e6
    assert lw < 10.0


@pytest.mark.parametrize("shrinkage", ["stan", "ledoit-wolf"])
@pytest.mark.parametrize("n_iter", [3, 8, 40, 300])
def test_shrunk_estimates_are_always_usable_metrics(shrinkage, n_iter):
    rng = np.random.default_rng(4)
    d = 12
    cov = ar1(d, 0.95, sd=2.0)
    chol = np.linalg.cholesky(cov)
    w = WindowMoments(d, mode="dense")
    for _ in range(n_iter):
        w.add(rng.standard_normal((4, d)) @ chol.T)
    est = estimate_covariance(w, shrinkage=shrinkage)
    DenseMetric(est)                                   # must not raise
    assert np.linalg.eigvalsh(est).min() > 0
    assert np.allclose(est, est.T)


def test_the_eigenvalue_floor_never_binds_on_a_real_window():
    """`floor` is a last resort, and the claim that it is unused is checkable."""
    rng = np.random.default_rng(5)
    d = 10
    chol = np.linalg.cholesky(ar1(d, 0.99))
    for n_iter in (3, 10, 100):
        w = WindowMoments(d, mode="dense")
        for _ in range(n_iter):
            w.add(rng.standard_normal((4, d)) @ chol.T)
        for shrinkage in ("stan", "ledoit-wolf"):
            floored = estimate_covariance(w, shrinkage=shrinkage, floor=1e-12)
            unfloored = estimate_covariance(w, shrinkage=shrinkage, floor=-np.inf)
            assert np.allclose(floored, unfloored, atol=0, rtol=1e-12)


def test_stan_shrink_interpolates_between_the_estimate_and_a_unit_ridge():
    S = np.array([[4.0, 1.0], [1.0, 9.0]])
    assert np.allclose(stan_shrink(S, 0), 1e-3 * np.eye(2))       # no data
    big = stan_shrink(S, 10_000_000)
    assert np.allclose(big, S, atol=1e-5)                          # all data
    mid = stan_shrink(S, 100)
    w = 100 / 105.0
    assert np.allclose(mid, w * S + 1e-3 * (5.0 / 105.0) * np.eye(2))


def test_estimate_covariance_rejects_an_unknown_shrinkage():
    w = WindowMoments(2, mode="dense")
    w.add(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="ledoit-wolf"):
        estimate_covariance(w, shrinkage="oracle")


# -- 3. Ledoit-Wolf is the estimator it says it is --------------------------

def test_ledoit_wolf_reduces_frobenius_error_where_it_is_supposed_to():
    """The estimator's own claim: lower *expected* squared error than S.

    Averaged over replicates, because a single draw can go either way -- the
    optimality is in expectation and the test says so by measuring it that way.
    """
    rng = np.random.default_rng(6)
    d = 10
    cov = ar1(d, 0.9, sd=1.5)
    chol = np.linalg.cholesky(cov)
    for n, expect_gain in [(15, True), (40, True), (2000, False)]:
        err_S, err_lw, intens = [], [], []
        for _ in range(40):
            x = rng.standard_normal((n, d)) @ chol.T
            xc = x - x.mean(axis=0)
            S = (xc.T @ xc) / n
            lw, a = ledoit_wolf(x)
            err_S.append(np.sum((S - cov) ** 2))
            err_lw.append(np.sum((lw - cov) ** 2))
            intens.append(a)
        if expect_gain:
            assert np.mean(err_lw) < np.mean(err_S)
        # the intensity always lives in [0, 1] and shrinks toward 0 with n
        assert 0.0 <= np.mean(intens) <= 1.0
    assert np.mean(intens) < 0.05          # n = 2000: nearly no shrinkage left


def test_ledoit_wolf_intensity_beats_neighbouring_intensities_on_average():
    """Not just 'better than S': near the *minimizer* of the loss it targets."""
    rng = np.random.default_rng(7)
    d, n = 8, 25
    cov = ar1(d, 0.8)
    chol = np.linalg.cholesky(cov)
    grid = np.linspace(0.0, 1.0, 21)
    losses = np.zeros_like(grid)
    chosen = []
    for _ in range(200):
        x = rng.standard_normal((n, d)) @ chol.T
        xc = x - x.mean(axis=0)
        S = (xc.T @ xc) / n
        m = np.trace(S) / d
        for j, a in enumerate(grid):
            est = a * m * np.eye(d) + (1 - a) * S
            losses[j] += np.sum((est - cov) ** 2)
        chosen.append(ledoit_wolf(x)[1])
    best = grid[int(np.argmin(losses))]
    assert abs(np.mean(chosen) - best) < 0.15


def test_ledoit_wolf_shrinks_a_spherical_sample_all_the_way():
    d = 5
    x = np.sqrt(d) * np.vstack([np.eye(d), -np.eye(d)])   # sample cov = I exactly
    est, a = ledoit_wolf(x)
    assert a == pytest.approx(1.0)
    assert np.allclose(est, np.cov(x.T, bias=True))

def test_ledoit_wolf_rejects_degenerate_input():
    with pytest.raises(ValueError, match="at least 2"):
        ledoit_wolf(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="shape"):
        ledoit_wolf(np.zeros(3))


# -- the loss a metric is judged by ------------------------------------------

def test_whitened_condition_number_hits_its_two_closed_forms():
    """For a Gaussian: the exact metric gives 1, the best diagonal gives kappa(R)."""
    d, rho = 6, 0.9
    cov = ar1(d, rho)
    assert whitened_condition_number(cov, cov) == pytest.approx(1.0, rel=1e-10)

    R = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    kappa_R = np.linalg.cond(R)          # R symmetric PD, so cond == eigenvalue ratio
    assert whitened_condition_number(np.diag(cov), cov) == pytest.approx(
        kappa_R, rel=1e-10
    )
    # ... and in 2-D that closed form is (1+rho)/(1-rho)
    two = np.array([[1.0, rho], [rho, 1.0]])
    assert whitened_condition_number(np.diag(two), two) == pytest.approx(
        (1 + rho) / (1 - rho), rel=1e-10
    )
    # scaling the marginals changes the diagonal metric's job but not its floor
    s = np.diag([1.0, 30.0, 0.1, 5.0, 1.0, 100.0])
    scaled = s @ cov @ s
    assert whitened_condition_number(np.diag(scaled), scaled) == pytest.approx(
        kappa_R, rel=1e-8
    )


# -- end to end through the sampler ------------------------------------------

def test_dense_adaptation_recovers_the_covariance_of_a_correlated_gaussian():
    d = 4
    cov = ar1(d, 0.9, sd=2.0)
    target = Gaussian(mean=np.zeros(d), cov=cov)
    res = hmc(target, np.zeros((4, d)), n_samples=200, step_size=0.4,
              n_leapfrog=20, rng=np.random.default_rng(8), n_warmup=3000,
              adapt_step_size=True, adapt_mass="dense")
    est = res.extras["inv_mass"]
    assert est.shape == (d, d)
    assert isinstance(res.extras["metric"], DenseMetric)
    # the estimate is the posterior covariance, to sampling error
    assert np.allclose(est, cov, atol=0.45)
    # and it does the job the diagonal cannot: kappa near 1, well under kappa(R)
    kappa_dense = whitened_condition_number(est, cov)
    kappa_diag = whitened_condition_number(np.diag(cov), cov)
    assert kappa_dense < 0.5 * kappa_diag


def test_dense_and_diagonal_adaptation_both_still_sample_the_right_target():
    """A metric changes efficiency only. Both must recover the same mean."""
    d = 3
    cov = ar1(d, 0.85, sd=1.5)
    mean = np.array([1.0, -2.0, 0.5])
    target = Gaussian(mean=mean, cov=cov)
    for mode in ("diag", "dense"):
        res = hmc(target, np.zeros((4, d)), n_samples=4000, step_size=0.4,
                  n_leapfrog=15, rng=np.random.default_rng(9), n_warmup=1000,
                  adapt_step_size=True, adapt_mass=mode)
        got = res.pooled().mean(axis=0)
        assert np.allclose(got, mean, atol=0.12), (mode, got)
        assert res.accept_rate.mean() > 0.6


def test_short_warmup_leaves_the_dense_metric_at_identity():
    """No window closes, so nothing is estimated -- and nothing is broken."""
    target = Gaussian(mean=[0.0, 0.0], cov=[[1.0, 0.5], [0.5, 1.0]])
    res = hmc(target, np.zeros((2, 2)), n_samples=50, step_size=0.3,
              n_leapfrog=8, rng=np.random.default_rng(10), n_warmup=5,
              adapt_mass="dense")
    assert np.array_equal(res.extras["inv_mass"], np.ones(2))


def test_adapt_mass_true_is_still_the_diagonal_metric():
    target = Gaussian(mean=[0.0, 0.0], cov=np.diag([1.0, 9.0]))
    kw = dict(n_samples=200, step_size=0.3, n_leapfrog=10, n_warmup=600,
              adapt_step_size=True)
    a = hmc(target, np.zeros((4, 2)), rng=np.random.default_rng(11),
            adapt_mass=True, **kw)
    b = hmc(target, np.zeros((4, 2)), rng=np.random.default_rng(11),
            adapt_mass="diag", **kw)
    assert np.array_equal(a.samples, b.samples)
    assert np.array_equal(a.extras["inv_mass"], b.extras["inv_mass"])
    assert a.extras["inv_mass"].shape == (2,)


def test_hmc_rejects_a_bad_adapt_mass_and_a_double_metric():
    target = Gaussian(mean=[0.0], cov=[[1.0]])
    kw = dict(n_samples=10, step_size=0.3, n_leapfrog=5,
              rng=np.random.default_rng(12), n_warmup=10)
    with pytest.raises(ValueError, match="adapt_mass must be"):
        hmc(target, np.zeros((2, 1)), adapt_mass="full", **kw)
    with pytest.raises(ValueError, match="not both"):
        hmc(target, np.zeros((2, 1)), adapt_mass="dense",
            metric=np.eye(1), **kw)
