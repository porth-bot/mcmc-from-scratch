"""Diagnostics validated against processes with known answers.

The key case is AR(1): x_{t+1} = rho x_t + sqrt(1-rho^2) eps_t has
rho_k = rho^k exactly, so tau = 1 + 2 sum rho^k = (1+rho)/(1-rho) in closed
form -- a ground truth for the ESS estimator itself.
"""

import numpy as np
import pytest

from mcmc.diagnostics import (
    _average_ranks,
    _standard_normal_ppf,
    autocorr_summary,
    autocorrelation,
    efficiency_summary,
    ess,
    integrated_autocorr_time,
    plot_autocorrelation,
    rank_normalize,
    rank_normalized_rhat,
    split_rhat,
    tail_ess,
    thinning_variance_ratio,
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


def test_tail_ess_of_iid_is_close_to_n():
    # For iid draws the tail indicators are iid Bernoulli, so tail-ESS ~ N.
    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 20_000))
    N = 80_000
    assert 0.8 * N < tail_ess(x) < 1.2 * N


def test_tail_ess_decreases_with_autocorrelation():
    # A slower AR(1) lingers in the tail longer, so the tail indicator is more
    # autocorrelated and tail-ESS falls monotonically with rho.
    rng = np.random.default_rng(6)
    te = [tail_ess(ar1(rho, 4, 50_000, rng)) for rho in (0.0, 0.5, 0.9)]
    assert te[0] > te[1] > te[2] > 0


def test_tail_ess_is_the_min_over_both_tails():
    # By construction tail-ESS is the smaller of the two per-quantile ESSs;
    # it must not exceed either one.
    rng = np.random.default_rng(7)
    x = ar1(0.8, 4, 40_000, rng)
    lo, hi = np.quantile(x, [0.05, 0.95])
    lower = ess((x <= lo).astype(float))
    upper = ess((x >= hi).astype(float))
    assert tail_ess(x) == pytest.approx(min(lower, upper))


def test_tail_ess_rejects_bad_probability():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((2, 1000))
    for bad in (0.0, 0.5, 0.7, -0.1):
        with pytest.raises(ValueError):
            tail_ess(x, prob=bad)


def test_tail_ess_flags_a_stuck_tail_via_the_min_over_sides():
    # The motivating case: one tail mixes fine while the other is reached only
    # in rare, long sticky excursions. tail-ESS takes the min over the two
    # sides, so it reports the bad side even when the other looks healthy --
    # and it lands well below the bulk-ESS, which averages over the whole run.
    rng = np.random.default_rng(9)
    m, n = 4, 60_000
    x = rng.standard_normal((m, n))
    for c in range(m):  # scatter a few long, stuck *upper*-tail excursions
        for s in rng.integers(0, n, size=6):
            L = int(rng.integers(150, 300))
            x[c, s : s + L] = 4.0
    lo, hi = np.quantile(x, [0.05, 0.95])
    lower = ess((x <= lo).astype(float))   # untouched, well-mixed side
    upper = ess((x >= hi).astype(float))   # the sticky side
    assert upper < 0.1 * lower             # the two sides disagree by 100x
    assert tail_ess(x) == pytest.approx(min(lower, upper))
    assert tail_ess(x) < 0.7 * ess(x)      # tail is worse than the bulk


def test_rhat_near_one_for_mixed_chains():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 5_000))
    assert split_rhat(x) < 1.01


def test_rhat_flags_unmixed_chains():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((4, 5_000))
    x += np.array([0.0, 0.0, 3.0, 3.0])[:, None]  # two chains stuck elsewhere
    assert split_rhat(x) > 1.5


def test_probit_matches_known_normal_quantiles():
    # Phi^{-1} at values with exact/tabulated answers, including the far tails
    # (where Acklam's raw approximation is weakest and the Halley step earns it).
    p = np.array([1e-4, 0.025, 0.1, 0.5, 0.8413447460685429, 0.975, 1 - 1e-4])
    known = np.array([-3.719016485455709, -1.959963984540054,
                      -1.2815515594605868, 0.0, 1.0, 1.959963984540054,
                      3.719016485455709])
    np.testing.assert_allclose(_standard_normal_ppf(p), known, atol=1e-7)


def test_average_ranks_share_the_mean_rank_within_ties():
    # Repeated states (a Metropolis chain sits still on rejection) must not bias
    # the rank transform: tied values get the average of the ranks they span.
    a = np.array([10.0, 10.0, 20.0, 5.0, 5.0, 5.0])
    # sorted: 5,5,5 (ranks 1,2,3 -> 2), 10,10 (ranks 4,5 -> 4.5), 20 (rank 6)
    np.testing.assert_allclose(_average_ranks(a),
                               [4.5, 4.5, 6.0, 2.0, 2.0, 2.0])


def test_rank_normalized_draws_are_standard_normal():
    # The Blom transform maps pooled draws of ANY continuous target onto ~N(0,1)
    # scores; a heavy-tailed input is no exception (that is the robustness).
    rng = np.random.default_rng(0)
    z = rank_normalize(rng.standard_cauchy((4, 10_000)))
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1.0) < 0.02


def test_rank_rhat_agrees_with_classic_on_well_mixed_gaussian():
    # Where the classic statistic is valid, rank-R-hat must not disagree: both
    # sit at ~1 for four mixed light-tailed chains.
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 5_000))
    assert split_rhat(x) < 1.01
    assert rank_normalized_rhat(x)["rhat"] < 1.01


def test_rank_rhat_flags_unmixed_gaussian_like_the_classic():
    # It must still catch the easy case the classic statistic already catches.
    rng = np.random.default_rng(4)
    x = rng.standard_normal((4, 5_000)) + np.array([0.0, 0.0, 3.0, 3.0])[:, None]
    assert rank_normalized_rhat(x)["rhat"] > 1.2


def test_rank_rhat_catches_a_cauchy_location_shift_the_classic_misses():
    # THE motivating case. Two of four Cauchy chains are shifted by 6 (three
    # inter-quartile ranges): genuinely unmixed. But a standard Cauchy routinely
    # throws draws of magnitude 50+, so the within-chain "variance" W is enormous
    # and noisy and the shift vanishes into it -- classic split-R-hat reads ~1.00.
    # Ranks are bounded, so the shift shows up cleanly in the bulk term.
    rng = np.random.default_rng(0)
    x = rng.standard_cauchy((4, 4_000)) + np.array([0.0, 0.0, 6.0, 6.0])[:, None]
    assert split_rhat(x) < 1.01               # classic is fooled
    r = rank_normalized_rhat(x)
    assert r["bulk"] > 1.2                     # the location term catches it
    assert r["rhat"] > 1.2


def test_folded_rhat_catches_a_scale_difference_the_bulk_term_misses():
    # Same median (0), different spread: two Cauchy chains at scale 1, two at
    # scale 6. The rank/bulk term is blind to this (the rank distributions are
    # symmetric about the pooled median either way), and so is the classic
    # statistic. Folding to |x - median| turns the scale gap into a location gap
    # of the absolute deviations, which the folded term then flags.
    rng = np.random.default_rng(0)
    x = rng.standard_cauchy((4, 4_000)) * np.array([1.0, 1.0, 6.0, 6.0])[:, None]
    assert split_rhat(x) < 1.01               # classic is fooled
    r = rank_normalized_rhat(x)
    assert r["bulk"] < 1.05                    # so is the plain bulk term
    assert r["folded"] > 1.1                   # only folding sees the scale gap
    assert r["rhat"] == max(r["bulk"], r["folded"])


def test_efficiency_summary_is_ess_normalized_by_cost():
    # ESS must match the standalone estimator, and the two rate columns must be
    # exactly ESS divided by the wall-clock and (per-1k) evaluation budgets.
    rng = np.random.default_rng(7)
    x = ar1(0.6, 4, 20_000, rng)
    seconds, n_evals = 2.5, 500_000
    s = efficiency_summary(x, seconds, n_evals)
    assert s["ess"] == pytest.approx(ess(x))
    assert s["tau"] == integrated_autocorr_time(x)
    assert s["ess_per_sec"] == pytest.approx(s["ess"] / seconds)
    assert s["ess_per_keval"] == pytest.approx(1000.0 * s["ess"] / n_evals)


def test_efficiency_summary_handles_degenerate_budgets():
    # Zero time / zero evals must not raise (a not-yet-run sampler): report NaN.
    s = efficiency_summary(np.random.default_rng(8).standard_normal((4, 2_000)),
                           seconds=0.0, n_evals=0)
    assert np.isnan(s["ess_per_sec"]) and np.isnan(s["ess_per_keval"])
    assert s["ess"] > 0


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


# ---------------------------------------------------------------------------
# Thinning (Sec. 6.3): the closed form, its limits, and the empirical check
# ---------------------------------------------------------------------------
def test_thinning_ratio_is_one_when_not_thinning():
    for rho in (0.0, 0.5, 0.95):
        assert thinning_variance_ratio(rho, 1) == pytest.approx(1.0)


def test_thinning_always_costs_and_costs_more_the_more_you_thin():
    """R > 1 for k > 1, and monotone increasing in k -- thinning never helps."""
    for rho in (0.0, 0.3, 0.7, 0.9, 0.99):
        ratios = [thinning_variance_ratio(rho, k) for k in range(1, 21)]
        assert all(r > 1.0 for r in ratios[1:])
        assert all(b > a for a, b in zip(ratios, ratios[1:]))  # strictly increasing


def test_thinning_iid_draws_wastes_exactly_the_factor_you_discard():
    # rho = 0: keeping 1 in k independent draws inflates the variance by k.
    for k in (2, 5, 10):
        assert thinning_variance_ratio(0.0, k) == pytest.approx(float(k))


def test_thinning_a_very_sticky_chain_is_nearly_free_but_never_helps():
    # rho -> 1: the discarded draws were near-duplicates, so R -> 1 from above.
    assert thinning_variance_ratio(0.999, 5) == pytest.approx(1.0, abs=0.02)
    assert thinning_variance_ratio(0.999, 5) > 1.0


def test_thinning_rejects_bad_arguments():
    with pytest.raises(ValueError):
        thinning_variance_ratio(1.0, 2)  # rho must be < 1
    with pytest.raises(ValueError):
        thinning_variance_ratio(-0.1, 2)
    with pytest.raises(ValueError):
        thinning_variance_ratio(0.5, 0)  # k >= 1


def test_thinning_ratio_matches_the_empirical_variance_of_the_mean():
    """The formula against brute force: 4000 independent AR(1) chains.

    Estimate Var(sample mean) across replicates for the full chain and for the
    thinned chain, and check the measured ratio matches the closed form. This
    is the test that makes the closed form more than algebra on a page.
    """
    rng = np.random.default_rng(11)
    rho, n_rep, n = 0.9, 4000, 2000
    x = ar1(rho, n_rep, n, rng)  # (n_rep, n): each row an independent chain

    var_full = np.var(x.mean(axis=1))
    for k in (2, 5, 10):
        var_thin = np.var(x[:, ::k].mean(axis=1))
        empirical = var_thin / var_full
        predicted = thinning_variance_ratio(rho, k)
        assert empirical == pytest.approx(predicted, rel=0.10)


def test_thinning_ratio_matches_the_measured_ess_loss():
    """ESS_thinned / ESS_full should be 1 / R, using the repo's own estimator."""
    rng = np.random.default_rng(12)
    rho = 0.8
    x = ar1(rho, 4, 100_000, rng)
    for k in (2, 5):
        measured = ess(x[:, ::k]) / ess(x)
        assert measured == pytest.approx(1.0 / thinning_variance_ratio(rho, k), rel=0.10)
