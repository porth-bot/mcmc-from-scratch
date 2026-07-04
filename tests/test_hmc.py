"""HMC structural guarantees (reversibility, energy-error order) and
statistical correctness (exact Gaussian moments, adaptation)."""

import numpy as np

from mcmc.hmc import hmc, leapfrog
from mcmc.targets import Gaussian


def _correlated_gaussian():
    mean = np.array([1.0, -1.0])
    cov = np.array([[1.0, 1.8], [1.8, 4.0]])  # rho = 0.9
    return Gaussian(mean, cov)


def test_leapfrog_is_reversible():
    """Integrate forward, flip momentum, integrate back: must return to the
    start to float roundoff. This is the property that makes the HMC proposal
    a valid involution -- if it fails, HMC is silently biased."""
    g = _correlated_gaussian()
    rng = np.random.default_rng(3)
    x0 = rng.standard_normal((6, 2))
    p0 = rng.standard_normal((6, 2))
    xf, pf = leapfrog(g.grad_logpdf, x0, p0, step_size=0.15, n_steps=30)
    xb, pb = leapfrog(g.grad_logpdf, xf, -pf, step_size=0.15, n_steps=30)
    np.testing.assert_allclose(xb, x0, atol=1e-10)
    np.testing.assert_allclose(pb, -p0, atol=1e-10)


def test_leapfrog_energy_error_is_second_order():
    """Halving eps at fixed trajectory time T = L*eps must cut the peak
    energy error |Delta H| by ~4x (leapfrog is O(eps^2))."""
    g = _correlated_gaussian()
    rng = np.random.default_rng(4)
    x0 = rng.standard_normal((1, 2))
    p0 = rng.standard_normal((1, 2))

    def H(x, p):
        return -g.logpdf(x) + 0.5 * np.sum(p**2, axis=1)

    def max_energy_error(eps, L):
        x, p, h0 = x0, p0, H(x0, p0)[0]
        worst = 0.0
        for _ in range(L):
            x, p = leapfrog(g.grad_logpdf, x, p, eps, 1)
            worst = max(worst, abs(H(x, p)[0] - h0))
        return worst

    e1 = max_energy_error(0.2, 20)
    e2 = max_energy_error(0.1, 40)
    assert e1 > 1e-6  # above roundoff, so the ratio is meaningful
    assert 2.8 < e1 / e2 < 5.5


def test_hmc_recovers_gaussian_moments():
    g = _correlated_gaussian()
    rng = np.random.default_rng(5)
    x0 = rng.standard_normal((4, 2)) * 3.0
    res = hmc(
        g, x0, n_samples=5_000, step_size=0.25, n_leapfrog=20, rng=rng, n_warmup=500
    )
    pooled = res.pooled()
    np.testing.assert_allclose(pooled.mean(axis=0), g.mean, atol=0.1)
    np.testing.assert_allclose(np.cov(pooled.T), g.cov, rtol=0.12, atol=0.08)
    assert res.accept_rate.mean() > 0.85  # long-trajectory HMC on a smooth target
    assert res.extras["n_divergent"] == 0


def test_dual_averaging_hits_target_acceptance():
    g = Gaussian(np.zeros(10), np.eye(10))
    rng = np.random.default_rng(6)
    x0 = rng.standard_normal((4, 10))
    res = hmc(
        g,
        x0,
        n_samples=2_000,
        step_size=1e-3,  # deliberately far too small; adaptation must find ~O(1)
        n_leapfrog=15,
        rng=rng,
        n_warmup=1_000,
        adapt_step_size=True,
        target_accept=0.8,
    )
    assert 0.65 < res.accept_rate.mean() < 0.95
    assert res.extras["step_size"] > 0.1
