"""Parallel tempering: correctness on a unimodal target, and the multimodal
win it exists for (recovering both modes where a single chain is trapped)."""

import numpy as np

from mcmc.metropolis import random_walk_metropolis
from mcmc.targets import Gaussian, GaussianMixture
from mcmc.tempering import geometric_ladder, parallel_tempering


def test_geometric_ladder_is_descending_from_one():
    b = geometric_ladder(6, beta_min=0.01)
    assert b[0] == 1.0
    assert np.isclose(b[-1], 0.01)
    assert np.all(np.diff(b) < 0)  # strictly cooling-to-hot
    assert geometric_ladder(1)[0] == 1.0


def test_recovers_unimodal_gaussian_moments():
    """A valid sampler first of all must not distort an easy target: the cold
    chain reproduces a correlated Gaussian's mean and covariance."""
    g = Gaussian(mean=[1.0, -1.0], cov=[[1.0, 0.6], [0.6, 1.0]])
    betas = geometric_ladder(4, beta_min=0.1)
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((4, 2))
    res = parallel_tempering(
        g, x0, n_samples=12_000, step_sizes=1.2 / np.sqrt(betas), betas=betas,
        rng=rng, n_warmup=3_000,
    )
    pooled = res.pooled()
    np.testing.assert_allclose(pooled.mean(axis=0), g.mean, atol=0.1)
    np.testing.assert_allclose(np.cov(pooled.T), g.cov, atol=0.15)


def _wide_bimodal():
    # modes 12 apart: the barrier is uncrossable for a unit-scale random walk
    return GaussianMixture(
        weights=[0.35, 0.65], means=[[-6.0, 0.0], [6.0, 0.0]],
        covs=[np.eye(2), np.eye(2)],
    )


def test_single_chain_is_trapped_but_tempering_escapes():
    """The headline: started entirely in the left mode, a plain random walk
    never reaches the right mode (wrong by a factor of ~3 in the mean), while
    parallel tempering recovers both the mixture mean and the mode weights."""
    gm = _wide_bimodal()
    rng = np.random.default_rng(0)
    K = 8
    betas = geometric_ladder(K, beta_min=0.01)
    x0 = np.tile([-6.0, 0.0], (K, 1)) + 0.3 * rng.standard_normal((K, 2))

    res = parallel_tempering(
        gm, x0, n_samples=10_000, step_sizes=1.2 / np.sqrt(betas), betas=betas,
        rng=rng, n_warmup=4_000,
    )
    cold = res.samples[0]
    left_frac = float((cold[:, 0] < 0).mean())
    np.testing.assert_allclose(cold.mean(axis=0), gm.mean(), atol=0.5)
    assert abs(left_frac - 0.35) < 0.08                 # weights recovered
    assert np.all(res.extras["swap_rates"] > 0.1)       # ladder actually mixes

    # negative control: the same start with one untempered chain stays trapped
    rw = random_walk_metropolis(
        gm, x0[:1], n_samples=10_000, step_size=1.2, rng=np.random.default_rng(1),
        n_warmup=4_000,
    )
    assert float((rw.samples[0][:, 0] < 0).mean()) > 0.95  # never left the left mode


def test_rejects_hot_cold_chain():
    gm = _wide_bimodal()
    import pytest

    with pytest.raises(ValueError):
        parallel_tempering(
            gm, np.zeros((3, 2)), n_samples=10, step_sizes=1.0,
            betas=[0.5, 0.2, 0.1], rng=np.random.default_rng(0),
        )
