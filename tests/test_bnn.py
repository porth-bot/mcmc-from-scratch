"""Bayesian neural network: hand-derived backprop gradient (vs finite
differences) and end-to-end HMC inference on the gapped-sine toy."""

import numpy as np

from mcmc.bnn import BayesianNNRegression, make_gapped_sine, train_map
from mcmc.hmc import hmc
from mcmc.targets import finite_difference_grad


def _toy(seed=0, n_hidden=16):
    rng = np.random.default_rng(seed)
    X, y = make_gapped_sine(rng, n=40)
    model = BayesianNNRegression(X, y, n_hidden=n_hidden, noise_std=0.1, prior_std=1.0)
    return model, rng


def test_gradient_matches_finite_differences():
    """The money test: the log-posterior gradient is a hand-written backprop
    pass through the tanh net plus the Gaussian-prior term. Finite differences
    are the independent oracle. A single wrong (1 - z^2) factor or a mispacked
    block shows up here."""
    model, rng = _toy()
    theta = 0.5 * rng.standard_normal((5, model.dim))
    np.testing.assert_allclose(
        model.grad_logpdf(theta),
        finite_difference_grad(model.logpdf, theta),
        rtol=1e-5,
        atol=1e-5,
    )


def test_gradient_check_at_larger_scale():
    """tanh saturates for large pre-activations, where 1 - z^2 -> 0; check the
    gradient there too, not just near the origin."""
    model, rng = _toy(seed=1)
    theta = 2.0 * rng.standard_normal((4, model.dim))
    np.testing.assert_allclose(
        model.grad_logpdf(theta),
        finite_difference_grad(model.logpdf, theta),
        rtol=1e-5,
        atol=1e-5,
    )


def test_pack_unpack_shapes_and_forward():
    model, rng = _toy()
    theta = rng.standard_normal((3, model.dim))
    W1, b1, w2, b2 = model._unpack(theta)
    assert W1.shape == b1.shape == w2.shape == (3, model.H)
    assert b2.shape == (3,)
    grid = np.linspace(-2, 2, 17)
    assert model.forward(theta, grid).shape == (3, grid.size)
    # a single flat vector is accepted (n_chains = 1)
    assert model.forward(theta[0]).shape == (1, model.X.size)


def test_logpdf_matches_explicit_likelihood_plus_prior():
    """logpdf equals the residual sum-of-squares likelihood plus the Gaussian
    prior computed independently from forward() -- guards the assembly of the
    two terms (signs, the 1/2 factors, which variance divides which)."""
    model, rng = _toy()
    theta = 0.4 * rng.standard_normal((6, model.dim))
    f = model.forward(theta)  # (C, N)
    resid = model.y[None, :] - f
    expected = (
        -0.5 * np.sum(resid**2, axis=1) / model.noise_var
        - 0.5 * np.sum(theta**2, axis=1) / model.prior_var
    )
    np.testing.assert_allclose(model.logpdf(theta), expected, rtol=1e-12)


def test_hmc_fits_toy_and_widens_in_the_gap():
    """End-to-end: HMC over the ~49-dim weight posterior recovers sin(3x) on
    the observed region (predictive RMSE below the noise floor) and reports
    substantially larger predictive uncertainty across the held-out gap --
    the qualitative payoff of Bayesian inference here.

    Convergence is judged in *function* space on purpose: weight-space
    permutation/sign symmetries make raw-weight R-hat meaningless."""
    model, rng = _toy(seed=0)
    x0 = 0.1 * rng.standard_normal((4, model.dim))
    res = hmc(
        model, x0, n_samples=1500, step_size=0.01, n_leapfrog=30, rng=rng,
        n_warmup=1500, adapt_step_size=True, target_accept=0.9,
    )
    assert res.accept_rate.mean() > 0.7

    grid = np.linspace(-2.0, 2.0, 200)
    mean, std = model.posterior_predictive(res.samples, grid)
    truth = np.sin(3.0 * grid)
    observed = (grid < -0.5) | (grid > 0.5)
    gap = (grid > -0.5) & (grid < 0.5)

    rmse = np.sqrt(np.mean((mean[observed] - truth[observed]) ** 2))
    assert rmse < 0.15  # noise_std is 0.1; the function is recovered
    assert std[gap].mean() > 2.0 * std[observed].mean()


def test_train_map_ascends_the_log_posterior_and_fits_observed_data():
    """Adam MAP training must (1) monotone-ish increase the log-posterior it
    optimizes -- final logpdf strictly above the initialization for every
    ensemble member -- and (2) actually fit the observed data (low RMSE on the
    inputs it was shown). This is the point-estimate / deep-ensemble baseline
    the Day-5 experiment compares HMC against, so it has to genuinely train."""
    model, rng = _toy(seed=2)
    x0 = 0.1 * rng.standard_normal((5, model.dim))  # a 5-member ensemble
    lp0 = model.logpdf(x0)
    theta = train_map(model, x0, n_steps=2000, lr=0.01)
    lp1 = model.logpdf(theta)
    assert theta.shape == x0.shape
    assert np.all(lp1 > lp0)  # every member climbed

    preds = model.forward(theta, model.X)  # (5, n_data)
    rmse = np.sqrt(np.mean((preds - model.y[None, :]) ** 2, axis=1))
    assert np.all(rmse < 0.15)  # noise floor is 0.1; each member fits the data


def test_train_map_members_diverge_in_the_gap():
    """Different inits give a deep ensemble its epistemic spread: trained
    members agree where they saw data but disagree across the held-out gap.
    (The experiment's finding is that they still disagree *less* than HMC --
    here we only assert the spread is non-trivial and gap-concentrated.)"""
    model, rng = _toy(seed=3)
    x0 = 0.5 * rng.standard_normal((8, model.dim))
    theta = train_map(model, x0, n_steps=2000, lr=0.01)
    grid = np.linspace(-2.0, 2.0, 200)
    preds = model.forward(theta, grid)  # (8, 200)
    spread = preds.std(axis=0)
    observed = (grid < -0.5) | (grid > 0.5)
    gap = (grid > -0.5) & (grid < 0.5)
    assert spread[gap].mean() > spread[observed].mean()


def test_make_gapped_sine_respects_the_gap():
    rng = np.random.default_rng(4)
    X, y = make_gapped_sine(rng, n=40, gap=(-0.5, 0.5))
    assert X.size == 40
    assert not np.any((X > -0.5) & (X < 0.5))
    assert np.all((X >= -2.0) & (X <= 2.0))
