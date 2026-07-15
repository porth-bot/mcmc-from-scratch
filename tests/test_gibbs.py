"""Gibbs on a correlated Gaussian: exact conditionals, exact moments."""

import numpy as np
import pytest

from mcmc.diagnostics import ess
from mcmc.gibbs import gibbs, make_gaussian_gibbs_updates
from mcmc.targets import Gaussian


def test_gibbs_recovers_gaussian_moments():
    mean = np.array([2.0, -1.0, 0.0])
    rho = 0.8
    sd = np.array([1.0, 2.0, 0.5])
    corr = np.array([[1, rho, 0.2], [rho, 1, 0.1], [0.2, 0.1, 1]])
    cov = corr * np.outer(sd, sd)
    Gaussian(mean, cov)  # validates cov is SPD (Cholesky inside)

    rng = np.random.default_rng(7)
    updates = make_gaussian_gibbs_updates(mean, cov)
    init = {"x": rng.standard_normal((4, 3)) * 3.0}
    res = gibbs(updates, init, n_samples=15_000, rng=rng, n_warmup=1_000)

    pooled = res.pooled()
    np.testing.assert_allclose(pooled.mean(axis=0), mean, atol=0.1)
    np.testing.assert_allclose(np.cov(pooled.T), cov, rtol=0.1, atol=0.05)
    assert np.all(res.accept_rate == 1.0)


def test_gibbs_mixing_slows_as_correlation_grows():
    """Classic Gibbs pathology: for a 2D Gaussian with correlation rho, the
    lag-1 autocorrelation of each coordinate chain is rho^2, so mixing time
    blows up as rho -> 1. Checks the phenomenon, not just the moments."""
    rng = np.random.default_rng(11)
    lag1 = {}
    for rho in (0.5, 0.95):
        cov = np.array([[1.0, rho], [rho, 1.0]])
        res = gibbs(
            make_gaussian_gibbs_updates(np.zeros(2), cov),
            {"x": np.zeros((1, 2))},
            n_samples=20_000,
            rng=rng,
            n_warmup=500,
        )
        x = res.samples[0, :, 0]
        x = x - x.mean()
        lag1[rho] = (x[:-1] @ x[1:]) / (x @ x)
    assert abs(lag1[0.5] - 0.25) < 0.05      # rho^2 = 0.25
    assert abs(lag1[0.95] - 0.9025) < 0.03   # rho^2 = 0.9025


def test_random_scan_recovers_moments():
    """Random scan is a mixture of the same pi-invariant kernels, so it is
    exact too -- it must recover the target moments like systematic scan."""
    mean = np.array([2.0, -1.0, 0.5])
    rho = 0.7
    cov = np.array([[1.0, rho, 0.3], [rho, 1.0, 0.2], [0.3, 0.2, 1.0]])
    Gaussian(mean, cov)  # SPD check

    rng = np.random.default_rng(3)
    res = gibbs(
        make_gaussian_gibbs_updates(mean, cov),
        {"x": rng.standard_normal((4, 3)) * 2.0},
        n_samples=20_000,
        rng=rng,
        n_warmup=1_000,
        scan="random",
    )
    pooled = res.pooled()
    np.testing.assert_allclose(pooled.mean(axis=0), mean, atol=0.1)
    np.testing.assert_allclose(np.cov(pooled.T), cov, rtol=0.1, atol=0.05)
    assert np.all(res.accept_rate == 1.0)


def test_random_scan_is_seed_deterministic():
    """Same seed -> identical draws (the block indices come from the passed rng)."""
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    updates = make_gaussian_gibbs_updates(np.zeros(2), cov)
    kw = dict(init_state={"x": np.zeros((2, 2))}, n_samples=500, n_warmup=50,
              scan="random")
    a = gibbs(updates, rng=np.random.default_rng(0), **kw)
    b = gibbs(updates, rng=np.random.default_rng(0), **kw)
    np.testing.assert_array_equal(a.samples, b.samples)


def test_systematic_scan_mixes_better_than_random_on_correlated_gaussian():
    """At matched work (both do d block updates per recorded sweep), systematic
    scan is measurably more efficient on the correlated Gaussian: random scan
    sometimes updates one coordinate twice and leaves the other stale, raising
    autocorrelation. Measured ratio is ~1.9x; assert a conservative > 1.4."""
    cov = np.array([[1.0, 0.95], [0.95, 1.0]])
    updates = make_gaussian_gibbs_updates(np.zeros(2), cov)
    kw = dict(init_state={"x": np.zeros((4, 2))}, n_samples=15_000, n_warmup=1_000)

    sys = gibbs(updates, rng=np.random.default_rng(0), scan="systematic", **kw)
    rand = gibbs(updates, rng=np.random.default_rng(0), scan="random", **kw)
    ess_sys = ess(sys.samples[:, :, 0])
    ess_rand = ess(rand.samples[:, :, 0])
    assert ess_sys > 1.4 * ess_rand


def test_unknown_scan_raises():
    updates = make_gaussian_gibbs_updates(np.zeros(2), np.eye(2))
    with pytest.raises(ValueError, match="scan must be"):
        gibbs(updates, {"x": np.zeros((1, 2))}, n_samples=10,
              rng=np.random.default_rng(0), scan="sweep")
