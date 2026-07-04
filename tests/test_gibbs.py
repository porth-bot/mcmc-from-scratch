"""Gibbs on a correlated Gaussian: exact conditionals, exact moments."""

import numpy as np

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
