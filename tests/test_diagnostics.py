"""Diagnostics validated against processes with known answers.

The key case is AR(1): x_{t+1} = rho x_t + sqrt(1-rho^2) eps_t has
rho_k = rho^k exactly, so tau = 1 + 2 sum rho^k = (1+rho)/(1-rho) in closed
form -- a ground truth for the ESS estimator itself.
"""

import numpy as np

from mcmc.diagnostics import autocorrelation, ess, integrated_autocorr_time, split_rhat


def ar1(rho, m, n, rng):
    x = np.empty((m, n))
    x[:, 0] = rng.standard_normal(m)
    innov = np.sqrt(1 - rho**2) * rng.standard_normal((m, n))
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + innov[:, t]
    return x


def test_autocorrelation_of_ar1_matches_rho_k():
    rng = np.random.default_rng(0)
    x = ar1(0.8, 4, 100_000, rng)
    rho_hat = autocorrelation(x, max_lag=5)
    np.testing.assert_allclose(rho_hat, 0.8 ** np.arange(6), atol=0.02)


def test_tau_matches_ar1_closed_form():
    rng = np.random.default_rng(1)
    for rho, tol in [(0.5, 0.15), (0.9, 0.2)]:
        x = ar1(rho, 4, 50_000, rng)
        tau_true = (1 + rho) / (1 - rho)
        tau_hat = integrated_autocorr_time(x)
        assert abs(tau_hat - tau_true) / tau_true < tol


def test_ess_of_iid_samples_is_close_to_n():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 20_000))
    assert 0.85 * 80_000 < ess(x) < 1.15 * 80_000


def test_rhat_near_one_for_mixed_chains():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 5_000))
    assert split_rhat(x) < 1.01


def test_rhat_flags_unmixed_chains():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((4, 5_000))
    x += np.array([0.0, 0.0, 3.0, 3.0])[:, None]  # two chains stuck elsewhere
    assert split_rhat(x) > 1.5
