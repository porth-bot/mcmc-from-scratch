"""HMC structural guarantees (reversibility, energy-error order) and
statistical correctness (exact Gaussian moments, adaptation)."""

import numpy as np

from mcmc.diagnostics import ess
from mcmc.hmc import _mass_adaptation_schedule, hmc, leapfrog
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


# --- diagonal mass-matrix adaptation ------------------------------------


def test_leapfrog_reversible_with_diagonal_metric():
    """The metric only rescales the drift, which must not break the involution
    that makes the accept step exact: forward then momentum-flipped-backward
    must still return to the start with a non-identity inv_mass."""
    g = _correlated_gaussian()
    rng = np.random.default_rng(11)
    x0 = rng.standard_normal((5, 2))
    p0 = rng.standard_normal((5, 2))
    inv_mass = np.array([0.3, 4.0])
    xf, pf = leapfrog(g.grad_logpdf, x0, p0, 0.15, 30, inv_mass)
    xb, pb = leapfrog(g.grad_logpdf, xf, -pf, 0.15, 30, inv_mass)
    np.testing.assert_allclose(xb, x0, atol=1e-10)
    np.testing.assert_allclose(pb, -p0, atol=1e-10)


def test_inv_mass_none_matches_ones():
    """inv_mass=None (identity) must be bit-identical to inv_mass=ones, so the
    default path is provably unchanged by the mass-matrix machinery."""
    g = _correlated_gaussian()
    rng = np.random.default_rng(12)
    x0 = rng.standard_normal((3, 2))
    p0 = rng.standard_normal((3, 2))
    xa, pa = leapfrog(g.grad_logpdf, x0, p0, 0.2, 17, None)
    xb, pb = leapfrog(g.grad_logpdf, x0, p0, 0.2, 17, np.ones(2))
    np.testing.assert_array_equal(xa, xb)
    np.testing.assert_array_equal(pa, pb)


def test_mass_adaptation_schedule_structure():
    """Windows must expand, stay inside the buffers, and the last one must close
    exactly at n_warmup - term_buffer (so the terminal buffer re-tunes eps)."""
    W = 1000
    init_buffer, term_buffer, ends = _mass_adaptation_schedule(W)
    assert ends, "warmup of 1000 is long enough to adapt a metric"
    assert ends[0] >= init_buffer
    assert ends[-1] == W - term_buffer
    assert all(a < b for a, b in zip(ends, ends[1:]))  # strictly increasing
    widths = [b - a for a, b in zip([init_buffer] + ends, ends)]
    # memoryless windows double (except the final one, extended to the buffer)
    assert all(widths[i + 1] >= widths[i] for i in range(len(widths) - 2))
    # too-short warmup adapts nothing (identity metric kept)
    assert _mass_adaptation_schedule(30)[2] == []


def test_mass_adaptation_recovers_moments():
    """With the metric on, an axis-aligned target of unequal scales must still
    be sampled correctly -- adaptation changes efficiency, never the target."""
    g = Gaussian(np.array([2.0, -3.0]), np.diag([0.25, 16.0]))  # sd 0.5 and 4
    rng = np.random.default_rng(13)
    x0 = rng.standard_normal((4, 2))
    res = hmc(
        g, x0, n_samples=4_000, step_size=0.5, n_leapfrog=20, rng=rng,
        n_warmup=1_000, adapt_step_size=True, adapt_mass=True,
    )
    pooled = res.pooled()
    np.testing.assert_allclose(pooled.mean(axis=0), g.mean, atol=0.1)
    np.testing.assert_allclose(np.diag(np.cov(pooled.T)), [0.25, 16.0], rtol=0.15)
    assert res.extras["n_divergent"] == 0


def test_adapted_inv_mass_tracks_marginal_variances():
    """The adapted diagonal metric should recover the target's marginal
    variances (that is what M^{-1} = diag(Var) means)."""
    var = np.array([0.25, 4.0, 25.0])
    g = Gaussian(np.zeros(3), np.diag(var))
    rng = np.random.default_rng(14)
    x0 = rng.standard_normal((4, 3))
    res = hmc(
        g, x0, n_samples=10, step_size=0.5, n_leapfrog=20, rng=rng,
        n_warmup=2_000, adapt_step_size=True, adapt_mass=True,
    )
    np.testing.assert_allclose(res.extras["inv_mass"], var, rtol=0.25)


def test_mass_adaptation_improves_ess_on_anisotropic_target():
    """The payoff: on an axis-aligned Gaussian whose scales span two orders of
    magnitude, one step size cannot fit both directions under the identity
    metric, so the wide coordinate mixes slowly. The adapted diagonal metric
    preconditions both to unit scale -> far higher worst-coordinate ESS at a
    matched gradient budget. Both runs adapt the step size, so the metric is
    the only difference."""
    var = np.array([0.01, 100.0])  # sd 0.1 and 10, ratio 100
    g = Gaussian(np.zeros(2), np.diag(var))
    x0 = np.zeros((4, 2))

    common = dict(n_samples=3_000, step_size=0.1, n_leapfrog=25,
                  n_warmup=1_000, adapt_step_size=True)
    res_id = hmc(g, x0, rng=np.random.default_rng(15), adapt_mass=False, **common)
    res_ad = hmc(g, x0, rng=np.random.default_rng(15), adapt_mass=True, **common)

    def worst_ess(res):
        return min(ess(res.samples[:, :, d]) for d in range(2))

    assert res_ad.extras["n_divergent"] == 0
    # a clear, not marginal, win on the binding (worst) coordinate
    assert worst_ess(res_ad) > 3.0 * worst_ess(res_id)
