"""NUTS: statistical correctness (exact Gaussian moments), the structural
guarantees that make it valid and terminating (leapfrog agreement, bounded
tree depth, U-turn adaptivity), and that its energy errors are on par with
fixed-L HMC at a matched step size."""

import numpy as np

from mcmc.diagnostics import ess
from mcmc.hmc import hmc, leapfrog
from mcmc.nuts import _is_turning, _leapfrog_step, nuts
from mcmc.targets import Gaussian


def _correlated_gaussian():
    mean = np.array([1.0, -1.0])
    cov = np.array([[1.0, 1.8], [1.8, 4.0]])  # rho = 0.9
    return Gaussian(mean, cov)


# --- the integrator NUTS is built on -------------------------------------


def test_leapfrog_step_matches_multistep_leapfrog():
    """The gradient-caching single step must be bit-for-bit the same arithmetic
    as hmc.leapfrog(..., n_steps=1); the only difference is that it reuses the
    known gradient at x instead of recomputing it. If they diverge, NUTS is
    integrating a different (untested) dynamics than HMC."""
    g = _correlated_gaussian()
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.standard_normal(2)
        p = rng.standard_normal(2)
        grad_x = g.grad_logpdf(x[None, :])[0]
        # forward and backward, identity and diagonal metric
        for inv_mass in (None, np.array([0.4, 3.0])):
            for eps in (0.17, -0.17):
                x1, p1, g1 = _leapfrog_step(g.grad_logpdf, x, p, grad_x, eps, inv_mass)
                xr, pr = leapfrog(g.grad_logpdf, x[None, :], p[None, :], eps, 1, inv_mass)
                np.testing.assert_allclose(x1, xr[0], atol=1e-12)
                np.testing.assert_allclose(p1, pr[0], atol=1e-12)
                np.testing.assert_allclose(g1, g.grad_logpdf(x1[None, :])[0], atol=1e-12)


def test_is_turning_criterion():
    """Two states moving apart along the span are not turning; flip one
    momentum so it points back and the criterion must fire."""
    x_minus = np.array([0.0, 0.0])
    x_plus = np.array([2.0, 0.0])
    p_out = np.array([1.0, 0.0])   # points along +x, away from x_minus
    p_back = np.array([-1.0, 0.0])  # points back toward x_minus
    assert not _is_turning(x_minus, x_plus, p_out, p_out, None)
    assert _is_turning(x_minus, x_plus, p_back, p_out, None)   # minus end turned
    assert _is_turning(x_minus, x_plus, p_out, p_back, None)   # plus end turned


# --- statistical correctness ---------------------------------------------


def test_nuts_recovers_gaussian_moments():
    """The point of the whole thing: no L to tune, and the correlated Gaussian's
    mean and full covariance come out right."""
    g = _correlated_gaussian()
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((4, 2)) * 3.0
    res = nuts(
        g, x0, n_samples=4_000, step_size=0.4, rng=rng,
        n_warmup=500, adapt_step_size=True,
    )
    pooled = res.pooled()
    np.testing.assert_allclose(pooled.mean(axis=0), g.mean, atol=0.1)
    np.testing.assert_allclose(np.cov(pooled.T), g.cov, rtol=0.12, atol=0.08)
    assert res.extras["n_divergent"] == 0


def test_nuts_recovers_moments_with_diagonal_metric():
    """A non-identity diagonal metric only rescales the dynamics; the target is
    unchanged, so an anisotropic axis-aligned Gaussian is still sampled right."""
    var = np.array([0.25, 16.0])  # sd 0.5 and 4
    g = Gaussian(np.array([2.0, -3.0]), np.diag(var))
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal((4, 2))
    res = nuts(
        g, x0, n_samples=4_000, step_size=0.6, rng=rng, n_warmup=500,
        adapt_step_size=True, inv_mass=var,  # metric = target marginal variances
    )
    pooled = res.pooled()
    np.testing.assert_allclose(pooled.mean(axis=0), g.mean, atol=0.1)
    np.testing.assert_allclose(np.diag(np.cov(pooled.T)), var, rtol=0.15)
    assert res.extras["n_divergent"] == 0


# --- structural guarantees -----------------------------------------------


def test_trees_terminate_within_max_depth():
    """Doubling must respect the depth cap even when set very low, and the
    recorded depths must never exceed it. This is the termination guarantee:
    without a cap a non-turning direction could double forever."""
    g = Gaussian(np.zeros(3), np.eye(3))
    rng = np.random.default_rng(3)
    x0 = rng.standard_normal((4, 3))
    max_depth = 4
    res = nuts(
        g, x0, n_samples=500, step_size=0.5, rng=rng,
        n_warmup=100, adapt_step_size=True, max_tree_depth=max_depth,
    )
    assert res.extras["tree_depth"].max() <= max_depth
    # a healthy step size on a standard normal U-turns well before the cap
    assert res.extras["tree_depth"].mean() < max_depth


def test_nuts_adapts_trajectory_length_to_geometry():
    """On a standard normal with a sane step size, NUTS should terminate by the
    U-turn criterion at a modest depth for most iterations -- i.e. it is not
    just always running to max_tree_depth. Adaptive length is the whole idea."""
    g = Gaussian(np.zeros(2), np.eye(2))
    rng = np.random.default_rng(4)
    x0 = rng.standard_normal((4, 2))
    res = nuts(
        g, x0, n_samples=2_000, step_size=0.7, rng=rng,
        n_warmup=500, adapt_step_size=True, max_tree_depth=10,
    )
    depths = res.extras["tree_depth"]
    assert depths.max() < 10  # never forced to the cap
    assert depths.mean() < 6  # short adaptive trajectories on easy geometry


def test_nuts_energy_error_comparable_to_fixed_L_hmc():
    """At a matched step size on the same target, NUTS's per-iteration energy
    error of the selected state should be in the same ballpark as fixed-L HMC's
    -- both integrate the same leapfrog dynamics, so neither should be an order
    of magnitude worse. (NUTS additionally adapts the LENGTH, which HMC cannot.)"""
    g = _correlated_gaussian()
    eps = 0.25
    x0 = np.zeros((4, 2))

    res_hmc = hmc(
        g, x0, n_samples=2_000, step_size=eps, n_leapfrog=20,
        rng=np.random.default_rng(5), n_warmup=0,
    )
    res_nuts = nuts(
        g, x0, n_samples=2_000, step_size=eps, rng=np.random.default_rng(5),
        n_warmup=0, adapt_step_size=False,
    )
    med_hmc = float(np.median(np.abs(res_hmc.extras["delta_H"])))
    med_nuts = float(np.median(np.abs(res_nuts.extras["delta_H"])))
    assert med_hmc > 1e-6 and med_nuts > 1e-6  # both above roundoff
    assert med_nuts < 3.0 * med_hmc  # same order of magnitude, not worse


def test_nuts_dual_averaging_reaches_target_and_mixes():
    """Dual averaging must recover an O(1) step size from a tiny initial value,
    and the resulting sampler must mix (positive ESS on both coordinates)."""
    g = _correlated_gaussian()
    rng = np.random.default_rng(6)
    x0 = rng.standard_normal((4, 2))
    res = nuts(
        g, x0, n_samples=2_000, step_size=1e-3,  # far too small; must grow
        rng=rng, n_warmup=1_000, adapt_step_size=True, target_accept=0.8,
    )
    assert res.extras["step_size"] > 0.05
    assert min(ess(res.samples[:, :, d]) for d in range(2)) > 100.0


def test_nuts_flags_divergences_with_huge_step_size():
    """An absurdly large fixed step size drives the integrator off the typical
    set; those leaves must be caught as divergences (weight zero, expansion
    halted) rather than silently biasing the chain or crashing."""
    g = Gaussian(np.zeros(5), np.eye(5))
    rng = np.random.default_rng(7)
    x0 = rng.standard_normal((4, 5))
    res = nuts(
        g, x0, n_samples=300, step_size=8.0, rng=rng,
        n_warmup=0, adapt_step_size=False,
    )
    assert res.extras["n_divergent"] > 0
    assert np.all(np.isfinite(res.samples))  # divergences rejected, not stored


def test_nuts_per_iteration_divergence_mask():
    """The per-iteration ``divergent`` mask must have the sampling shape and be
    consistent with the scalar count: at least one flagged iteration when the
    step size is huge, none when it is well tuned. The mask is what lets the
    benchmark plot *where* divergences land (the funnel neck)."""
    g = Gaussian(np.zeros(5), np.eye(5))

    # huge step: divergences happen, and the mask flags whole iterations
    res_bad = nuts(
        g, np.random.default_rng(7).standard_normal((4, 5)), n_samples=300,
        step_size=8.0, rng=np.random.default_rng(7), n_warmup=0,
        adapt_step_size=False,
    )
    mask = res_bad.extras["divergent"]
    assert mask.shape == res_bad.samples.shape[:2]
    assert mask.dtype == bool
    assert mask.any()  # some iteration hit a divergence
    # the count sums leaves, the mask sums iterations, so 0 < iters <= leaves
    assert 0 < int(mask.sum()) <= res_bad.extras["n_divergent"]

    # well-tuned run on the same easy target: no divergences flagged
    res_ok = nuts(
        g, np.random.default_rng(8).standard_normal((4, 5)), n_samples=500,
        step_size=0.6, rng=np.random.default_rng(8), n_warmup=200,
        adapt_step_size=True,
    )
    assert not res_ok.extras["divergent"].any()
    assert res_ok.extras["n_divergent"] == 0
