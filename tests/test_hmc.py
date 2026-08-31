"""HMC structural guarantees (reversibility, energy-error order) and
statistical correctness (exact Gaussian moments, adaptation)."""

import numpy as np
import pytest

from mcmc.diagnostics import ess
from mcmc.hmc import _mass_adaptation_schedule, hmc, leapfrog
from mcmc.metric import DenseMetric
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


# -- the dense metric (Sec. 4.10) --------------------------------------------
#
# The algebra itself is tested in tests/test_metric.py; what these test is the
# *wiring* -- that hmc() drifts, weighs and draws with the metric it was handed,
# and that having the option at all changes nothing about a run without one.


def _dense_target(rho=0.99, sd=(1.0, 4.0)):
    """A Gaussian the diagonal metric provably cannot fix: kappa(R) = 199 at
    rho = 0.99 whatever the marginals are scaled to (Sec. 4.10)."""
    sd = np.asarray(sd)
    cov = np.array([[1.0, rho], [rho, 1.0]]) * np.outer(sd, sd)
    return Gaussian(np.array([0.5, -0.5]), cov)


def test_a_dense_identity_metric_reproduces_the_default_run_bit_for_bit():
    """The check that an option cannot silently move published numbers: a dense
    metric holding the exact-float identity must give the same chain, not a
    close one. (Multiplying by an exact identity is exact; the momentum draw
    goes through L^-1 = I and consumes the same normals in the same order.)"""
    g = _correlated_gaussian()
    common = dict(n_samples=200, step_size=0.3, n_leapfrog=8, n_warmup=100,
                  adapt_step_size=True)
    a = hmc(g, np.zeros((3, 2)), rng=np.random.default_rng(21), metric=None, **common)
    b = hmc(g, np.zeros((3, 2)), rng=np.random.default_rng(21), metric=np.eye(2), **common)
    np.testing.assert_array_equal(a.samples, b.samples)
    np.testing.assert_array_equal(a.extras["delta_H"], b.extras["delta_H"])
    np.testing.assert_array_equal(a.accept_rate, b.accept_rate)
    assert a.extras["step_size"] == b.extras["step_size"]
    assert a.extras["n_grad_evals"] == b.extras["n_grad_evals"]


def test_a_dense_diagonal_matrix_traces_the_diagonal_metric_to_roundoff():
    """diag(v) as a matrix and v as a vector are the same metric, and one
    trajectory under them agrees to roundoff -- but *not* bit for bit, and the
    difference is worth stating rather than hiding behind a tolerance.

    The dense path computes `eps * (p @ diag(v))` and `||diag(sqrt v)^T p||^2`
    where the diagonal path computes `(eps * v) * p` and `sum(v * p**2)`; the
    products are the same real numbers in a different association, so they
    differ in the last bit (measured: 1.4e-17 on one drift, 5.6e-17 on one
    kinetic energy). Over a trajectory that stays at 1e-14 relative -- leapfrog
    does not amplify it -- but over a *chain* it is unbounded, because one
    accept comparison eventually falls on the other side of its uniform draw
    and the two chains separate completely. Only an exact-float identity
    survives at the chain level (the test above); this is the honest statement
    for every other metric.
    """
    g = _correlated_gaussian()
    v = np.array([0.4, 2.5])
    rng = np.random.default_rng(22)
    x0, p0 = rng.standard_normal((3, 2)), rng.standard_normal((3, 2))
    for n_steps in (1, 30):
        xa, pa = leapfrog(g.grad_logpdf, x0, p0, 0.2, n_steps, v)
        xb, pb = leapfrog(g.grad_logpdf, x0, p0, 0.2, n_steps, np.diag(v))
        np.testing.assert_allclose(xb, xa, rtol=1e-12, atol=0)
        np.testing.assert_allclose(pb, pa, rtol=1e-12, atol=0)


def test_leapfrog_reversible_with_a_dense_metric():
    """Reversibility is what makes the proposal a valid involution, and it is
    the property a wrong transpose in the drift would break: Sigma is symmetric
    so `p @ Sigma` and `Sigma @ p` agree, and this test would not notice -- but
    an asymmetric M^-1 would fail it, which is why DenseMetric symmetrizes."""
    g = _correlated_gaussian()
    rng = np.random.default_rng(23)
    sigma = np.array([[2.0, -1.1], [-1.1, 0.9]])
    x0 = rng.standard_normal((6, 2))
    p0 = rng.standard_normal((6, 2))
    m = DenseMetric(sigma)
    xf, pf = leapfrog(g.grad_logpdf, x0, p0, 0.12, 30, m)
    xb, pb = leapfrog(g.grad_logpdf, xf, -pf, 0.12, 30, m)
    np.testing.assert_allclose(xb, x0, atol=1e-10)
    np.testing.assert_allclose(pb, -p0, atol=1e-10)


def test_dense_metric_energy_error_is_still_second_order():
    """The kinetic energy and the drift have to be the same metric. If
    K(p) used one matrix and dx/dt another, the pair would not be a Hamiltonian
    system and |Delta H| would not fall as eps^2 -- so this is the test that
    catches a mismatched (or inverted) matrix in one of the two places, which
    a moments test would only see as slow bias."""
    g = _correlated_gaussian()
    m = DenseMetric(np.array([[2.0, -1.1], [-1.1, 0.9]]))
    rng = np.random.default_rng(24)
    x0 = rng.standard_normal((1, 2))
    p0 = m.draw_momentum(rng, (1,))

    def H(x, p):
        return -g.logpdf(x) + m.kinetic(p)

    def max_energy_error(eps, L):
        x, p, h0 = x0, p0, H(x0, p0)[0]
        worst = 0.0
        for _ in range(L):
            x, p = leapfrog(g.grad_logpdf, x, p, eps, 1, m)
            worst = max(worst, abs(H(x, p)[0] - h0))
        return worst

    e1 = max_energy_error(0.1, 20)
    e2 = max_energy_error(0.05, 40)
    assert e1 > 1e-6
    assert 2.8 < e1 / e2 < 5.5


def test_hmc_with_a_dense_metric_targets_the_right_distribution():
    """Invariance is not up for negotiation: a metric changes efficiency only.
    Run the correlated target with M^-1 = Sigma and recover both moments."""
    g = _dense_target()
    res = hmc(g, np.zeros((4, 2)), n_samples=4_000, step_size=1.0, n_leapfrog=12,
              rng=np.random.default_rng(25), n_warmup=500, adapt_step_size=True,
              metric=g.cov)
    pooled = res.pooled()
    np.testing.assert_allclose(pooled.mean(axis=0), g.mean, atol=0.15)
    np.testing.assert_allclose(np.cov(pooled.T), g.cov, rtol=0.08, atol=0.05)
    assert res.extras["n_divergent"] == 0


def test_the_moments_check_would_catch_an_inverted_momentum_draw():
    """The one bug in this module that runs perfectly while sampling the wrong
    joint: drawing p ~ N(0, Sigma) instead of N(0, Sigma^-1) = N(0, M). It
    breaks nothing structural -- reversibility, volume preservation and the
    accept rule are all still exact -- so it has to be caught statistically.
    This pins that the test above has the teeth to do it."""
    class InvertedDraw(DenseMetric):
        def draw_momentum(self, rng, shape):
            return rng.standard_normal((*shape, self.dim)) @ self.chol.T

    g = _dense_target()
    res = hmc(g, np.zeros((4, 2)), n_samples=4_000, step_size=1.0, n_leapfrog=12,
              rng=np.random.default_rng(25), n_warmup=500, adapt_step_size=True,
              metric=InvertedDraw(g.cov))
    # not marginally wrong: the momenta are Sigma-scaled twice over
    assert np.cov(res.pooled().T)[1, 1] > 10.0 * g.cov[1, 1]


def test_the_dense_metric_buys_the_rotation_the_diagonal_cannot():
    """Sec. 4.10's whole point, measured through the sampler rather than the
    Hessian: at rho = 0.99 the best diagonal metric still leaves
    kappa(R) = 199, and the exact-covariance dense metric leaves 1. Both runs
    adapt the step size from the same seed, so the metric is the only
    difference."""
    g = _dense_target()
    common = dict(n_samples=1_500, step_size=0.4, n_leapfrog=10, n_warmup=300,
                  adapt_step_size=True)
    res_diag = hmc(g, np.zeros((4, 2)), rng=np.random.default_rng(26),
                   metric=np.diag(g.cov), **common)
    res_dense = hmc(g, np.zeros((4, 2)), rng=np.random.default_rng(26),
                    metric=g.cov, **common)

    def ess_per_grad(res):
        worst = min(ess(res.samples[:, :, d]) for d in range(2))
        return 1000.0 * worst / res.extras["n_grad_evals"]

    assert ess_per_grad(res_dense) > 1.5 * ess_per_grad(res_diag)
    # and the mechanism is visible in the step size the two can afford
    assert res_dense.extras["step_size"] > 5.0 * res_diag.extras["step_size"]


def test_extras_report_the_metric_that_was_used():
    """A dense run reports the whole M^-1, not its diagonal: reporting the
    diagonal would name a metric the sampler never ran."""
    g = _dense_target()
    common = dict(n_samples=50, step_size=0.5, n_leapfrog=5,
                  rng=np.random.default_rng(27))
    res = hmc(g, np.zeros((2, 2)), metric=g.cov, **common)
    np.testing.assert_array_equal(res.extras["inv_mass"], g.cov)
    assert isinstance(res.extras["metric"], DenseMetric)
    res = hmc(g, np.zeros((2, 2)), metric=np.array([0.5, 2.0]), **common)
    np.testing.assert_array_equal(res.extras["inv_mass"], [0.5, 2.0])
    res = hmc(g, np.zeros((2, 2)), **common)
    np.testing.assert_array_equal(res.extras["inv_mass"], np.ones(2))


def test_a_fixed_metric_and_adaptation_are_mutually_exclusive():
    """Adaptation overwrites the metric at its first window boundary, so
    accepting both would silently discard one of them."""
    g = _correlated_gaussian()
    with pytest.raises(ValueError, match="not both"):
        hmc(g, np.zeros((2, 2)), n_samples=10, step_size=0.2, n_leapfrog=4,
            rng=np.random.default_rng(28), n_warmup=100, adapt_mass=True,
            metric=np.eye(2))


def test_a_metric_of_the_wrong_dimension_is_rejected_at_the_top():
    g = _correlated_gaussian()
    with pytest.raises(ValueError, match="dimensional"):
        hmc(g, np.zeros((2, 2)), n_samples=10, step_size=0.2, n_leapfrog=4,
            rng=np.random.default_rng(29), metric=np.eye(3))
