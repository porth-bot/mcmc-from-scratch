"""Diagnostics validated against processes with known answers.

The key case is AR(1): x_{t+1} = rho x_t + sqrt(1-rho^2) eps_t has
rho_k = rho^k exactly, so tau = 1 + 2 sum rho^k = (1+rho)/(1-rho) in closed
form -- a ground truth for the ESS estimator itself.
"""

import numpy as np
import pytest

from mcmc.diagnostics import (
    autocorr_summary,
    autocorrelation,
    ess,
    integrated_autocorr_time,
    plot_autocorrelation,
    split_rhat,
)


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


def test_autocorr_summary_matches_the_scalar_diagnostics():
    # The plot helper's data must agree with the standalone estimators, and its
    # curve must be the same autocorrelation, just truncated for display.
    rng = np.random.default_rng(5)
    x = ar1(0.85, 4, 40_000, rng)
    s = autocorr_summary(x, max_lag=30)
    assert s["tau"] == integrated_autocorr_time(x)
    assert s["ess"] == ess(x)
    np.testing.assert_allclose(s["rho"], autocorrelation(x, max_lag=30))
    assert s["rho"][0] == pytest.approx(1.0)
    # a positively-correlated chain truncates well past lag 0; the cutoff is
    # computed from the full autocorrelation, not the display window, so it may
    # exceed max_lag.
    assert s["cutoff_lag"] > 0
    # more correlation -> later cutoff and larger tau than a near-iid chain
    s_iid = autocorr_summary(rng.standard_normal((4, 40_000)))
    assert s["tau"] > s_iid["tau"]
    assert s["cutoff_lag"] >= s_iid["cutoff_lag"]


def test_plot_autocorrelation_draws_the_summary_curve():
    plt = pytest.importorskip("matplotlib.pyplot")  # experiments-only dep
    rng = np.random.default_rng(6)
    x = ar1(0.7, 4, 20_000, rng)
    s = autocorr_summary(x, max_lag=25)
    ax = plot_autocorrelation(x, max_lag=25, label="AR(1)")
    line = ax.get_lines()[0]
    np.testing.assert_allclose(line.get_ydata(), s["rho"])
    np.testing.assert_array_equal(line.get_xdata(), s["lags"])
    plt.close("all")
