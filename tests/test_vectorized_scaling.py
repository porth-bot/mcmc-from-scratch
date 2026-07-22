"""Batched chains are statistically valid at any chain count.

``experiments/vectorized_scaling.py`` argues the batched design (all chains
advanced as one NumPy computation) makes extra chains nearly free in wall-clock.
That timing is machine-dependent and not asserted here; what this pins is the
property the whole design -- and that figure -- rests on: HMC recovers the target
identically well whether it runs 1 chain or many, so pooling more chains is a
real variance-reduction, not an artifact of the batching. (The experiment imports
matplotlib via ``common``; this test stays package-only so the numpy-only CI job
can run it.)
"""

import numpy as np

from mcmc.hmc import hmc
from mcmc.targets import Gaussian


def _correlated_gaussian(dim):
    cov = 0.6 * np.ones((dim, dim)) + 0.4 * np.eye(dim)
    return Gaussian(mean=np.zeros(dim), cov=cov)


def _run(target, n_chains, n_samples, seed):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal((n_chains, target.dim))
    return hmc(target, x0, n_samples=n_samples, step_size=0.25,
               n_leapfrog=20, rng=rng, n_warmup=300)


def test_mean_recovered_at_any_chain_count():
    """1 long chain and 32 batched chains both recover the mean; the 32-chain
    pool, with far more total draws, is at least as accurate."""
    target = _correlated_gaussian(6)
    single = _run(target, n_chains=1, n_samples=6000, seed=0)
    batched = _run(target, n_chains=32, n_samples=1000, seed=1)

    err_single = np.abs(single.pooled().mean(axis=0) - target.mean).max()
    err_batched = np.abs(batched.pooled().mean(axis=0) - target.mean).max()

    assert err_single < 0.15
    assert err_batched < 0.10
    # 32k pooled draws beat 6k: more chains is genuine extra signal, not padding.
    assert err_batched <= err_single


def test_covariance_recovered_when_batched():
    """The batched pool recovers the full covariance, correlations included."""
    target = _correlated_gaussian(4)
    res = _run(target, n_chains=16, n_samples=2000, seed=2)
    cov = np.cov(res.pooled(), rowvar=False)
    assert np.abs(cov - target.cov).max() < 0.12


def test_shapes_scale_with_chain_count():
    target = _correlated_gaussian(3)
    for n_chains in (1, 8, 64):
        res = _run(target, n_chains=n_chains, n_samples=50, seed=n_chains)
        assert res.samples.shape == (n_chains, 50, 3)
        assert res.accept_rate.shape == (n_chains,)
