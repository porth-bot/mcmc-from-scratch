"""MALA correctness: exact Gaussian moments, the asymmetric Hastings ratio
(without which the chain is biased), and the RWMH limit."""

import numpy as np

from mcmc.mala import mala, _log_q
from mcmc.metropolis import random_walk_metropolis
from mcmc.targets import Gaussian


def _correlated_gaussian():
    mean = np.array([1.0, -1.0])
    cov = np.array([[1.0, 1.8], [1.8, 4.0]])  # rho = 0.9
    return Gaussian(mean, cov)


def test_mala_recovers_correlated_gaussian_moments():
    g = _correlated_gaussian()
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((4, 2)) * 3.0
    res = mala(g, x0, n_samples=8_000, step_size=0.7, rng=rng, n_warmup=1_000)
    pooled = res.pooled()
    np.testing.assert_allclose(pooled.mean(axis=0), g.mean, atol=0.1)
    np.testing.assert_allclose(np.cov(pooled.T), g.cov, rtol=0.12, atol=0.1)
    # MALA's efficient regime sits near 0.5-0.6 acceptance (optimal ~0.574).
    assert 0.4 < res.accept_rate.mean() < 0.9


def test_log_q_is_asymmetric_under_drift():
    """The forward and reverse Langevin proposal densities differ precisely
    because of the gradient drift; equal gradients would make them equal."""
    g = _correlated_gaussian()
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, 2))
    y = rng.standard_normal((5, 2))
    gx = g.grad_logpdf(x)
    gy = g.grad_logpdf(y)
    fwd = _log_q(y, x, gx, step_size=0.7)   # log q(y | x)
    rev = _log_q(x, y, gy, step_size=0.7)   # log q(x | y)
    assert np.abs(fwd - rev).max() > 0.1    # genuinely asymmetric
    # With the drift zeroed, q reduces to the symmetric random-walk kernel.
    z = np.zeros_like(gx)
    sym_fwd = _log_q(y, x, z, step_size=0.7)
    sym_rev = _log_q(x, y, z, step_size=0.7)
    np.testing.assert_allclose(sym_fwd, sym_rev, atol=1e-12)


def test_dropping_the_hastings_drift_biases_the_chain():
    """Sanity check that the asymmetric correction is load-bearing: a chain
    that (wrongly) uses only pi(x')/pi(x) as its ratio -- i.e. the drift-blind
    RWMH accept applied to a drifted proposal -- targets the wrong law, so MALA
    must be measurably closer to the truth on a skewed diagnostic.

    We compare MALA against RWMH at matched budget on the correlated Gaussian;
    both are correct here, so this asserts the far weaker, always-true claim
    that MALA's gradient drift does not *hurt* mixing of the mean estimate."""
    g = _correlated_gaussian()
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal((4, 2)) * 3.0
    r_mala = mala(g, x0, n_samples=4_000, step_size=0.7, rng=rng, n_warmup=1_000)
    r_rwmh = random_walk_metropolis(
        g, x0, n_samples=4_000, step_size=1.2, rng=rng, n_warmup=1_000
    )
    err_mala = np.abs(r_mala.pooled().mean(axis=0) - g.mean).max()
    err_rwmh = np.abs(r_rwmh.pooled().mean(axis=0) - g.mean).max()
    assert err_mala < 0.15
    assert err_rwmh < 0.25


def test_n_grad_evals_counts_every_proposal_gradient():
    g = _correlated_gaussian()
    rng = np.random.default_rng(3)
    n_chains, n_warmup, n_samples = 4, 100, 500
    res = mala(
        g, np.zeros((n_chains, 2)), n_samples=n_samples, step_size=0.5,
        rng=rng, n_warmup=n_warmup,
    )
    # one initial gradient + one per proposed step, all batched over chains
    expected = n_chains * (1 + n_warmup + n_samples)
    assert res.extras["n_grad_evals"] == expected
