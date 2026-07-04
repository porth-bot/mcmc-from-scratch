"""RWMH must reproduce exact moments of a correlated Gaussian.

Tolerances are set from the Monte Carlo standard error sqrt(var * tau / N)
with a generous multiple, then run with a fixed seed so the test is
deterministic; it probes correctness of the kernel, not luck.
"""

import numpy as np

from mcmc.metropolis import random_walk_metropolis
from mcmc.targets import Gaussian


def test_rwmh_recovers_gaussian_moments():
    mean = np.array([1.0, -1.0])
    cov = np.array([[1.0, 0.9 * 1.0 * 2.0], [0.9 * 1.0 * 2.0, 4.0]])  # rho = 0.9
    target = Gaussian(mean, cov)

    rng = np.random.default_rng(42)
    x0 = rng.standard_normal((4, 2)) * 3.0  # overdispersed starts
    res = random_walk_metropolis(
        target, x0, n_samples=20_000, step_size=0.8, rng=rng, n_warmup=2_000
    )

    pooled = res.pooled()
    # tau_int for this chain is O(50); MCSE for the mean ~ sigma*sqrt(50/80000) ~ 0.025*sigma
    np.testing.assert_allclose(pooled.mean(axis=0), mean, atol=0.15)
    np.testing.assert_allclose(np.cov(pooled.T), cov, rtol=0.15, atol=0.1)
    # in a mid-range dimension-2 problem the tuned rate should be moderate,
    # and identical dynamics should give similar rates across chains
    assert 0.15 < res.accept_rate.mean() < 0.6
    assert res.accept_rate.std() < 0.05


def test_rwmh_rejects_stay_put():
    """With an enormous step size almost everything is rejected; the chain
    must then repeat states rather than drift (regression guard on the
    accept/copy bookkeeping)."""
    target = Gaussian([0.0], [[1.0]])
    rng = np.random.default_rng(0)
    res = random_walk_metropolis(
        target, np.zeros((1, 1)), n_samples=500, step_size=500.0, rng=rng
    )
    assert res.accept_rate[0] < 0.05
    diffs = np.diff(res.samples[0, :, 0])
    assert (diffs == 0).mean() > 0.9
