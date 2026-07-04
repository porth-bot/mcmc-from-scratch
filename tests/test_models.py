"""Model log-posteriors: hand-derived gradients and cross-method agreement."""

import numpy as np

from mcmc.gibbs import gibbs
from mcmc.hmc import hmc
from mcmc.models import (
    ConjugateLinearRegression,
    EightSchoolsNonCentered,
    make_eight_schools_gibbs_updates,
)
from mcmc.targets import finite_difference_grad


def make_linreg(rng):
    X = rng.standard_normal((50, 3))
    beta_true = np.array([1.5, -2.0, 0.5])
    y = X @ beta_true + 0.7 * rng.standard_normal(50)
    return ConjugateLinearRegression(X, y, noise_var=0.49, prior_var=10.0)


def test_linreg_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    model = make_linreg(rng)
    beta = rng.standard_normal((6, 3))
    np.testing.assert_allclose(
        model.grad_logpdf(beta),
        finite_difference_grad(model.logpdf, beta),
        rtol=1e-5,
        atol=1e-6,
    )


def test_linreg_exact_posterior_satisfies_normal_equations():
    rng = np.random.default_rng(1)
    model = make_linreg(rng)
    post = model.exact_posterior()
    precision = model.X.T @ model.X / model.noise_var + np.eye(3) / model.prior_var
    np.testing.assert_allclose(
        precision @ post.mean, model.X.T @ model.y / model.noise_var, rtol=1e-10
    )
    # posterior mode = mean for a Gaussian: gradient must vanish there
    np.testing.assert_allclose(model.grad_logpdf(post.mean[None, :]), 0.0, atol=1e-8)


def test_eight_schools_gradient_matches_finite_differences():
    """The money test for the non-centered model: the hand-derived gradient
    includes the InvGamma-with-Jacobian term in log tau, the easiest place
    to make a sign/factor error."""
    model = EightSchoolsNonCentered()
    rng = np.random.default_rng(2)
    z = rng.standard_normal((8, model.dim))
    z[:, 1] = rng.uniform(-1.0, 2.5, size=8)  # spread over realistic log tau
    np.testing.assert_allclose(
        model.grad_logpdf(z),
        finite_difference_grad(model.logpdf, z),
        rtol=1e-5,
        atol=1e-6,
    )


def test_eight_schools_gibbs_and_hmc_agree():
    """Two independent inference routes -- conjugate Gibbs on the centered
    model, HMC on the non-centered one -- must agree on posterior means.
    Neither knows the answer a priori; agreement is the evidence."""
    rng = np.random.default_rng(3)

    updates = make_eight_schools_gibbs_updates()
    init = {
        "theta": rng.standard_normal((4, 8)) * 5.0,
        "mu": rng.standard_normal(4) * 5.0,
        "tau2": np.full(4, 4.0),
    }
    res_g = gibbs(updates, init, n_samples=8_000, rng=rng, n_warmup=1_000)
    parts = res_g.extras["unpack"]()
    mu_g = parts["mu"].mean()
    tau_g = np.sqrt(parts["tau2"]).mean()

    model = EightSchoolsNonCentered()
    z0 = 0.1 * rng.standard_normal((4, model.dim))
    res_h = hmc(
        model, z0, n_samples=4_000, step_size=0.1, n_leapfrog=20, rng=rng,
        n_warmup=1_000, adapt_step_size=True,
    )
    params = model.transform(res_h.samples)
    mu_h = params["mu"].mean()
    tau_h = params["tau"].mean()

    assert abs(mu_g - mu_h) < 0.5   # posterior sd of mu is ~4; MCSE << 0.5
    assert abs(tau_g - tau_h) < 0.5
