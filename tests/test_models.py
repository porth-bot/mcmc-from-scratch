"""Model log-posteriors: hand-derived gradients and cross-method agreement."""

import numpy as np

from mcmc.gibbs import gibbs
from mcmc.hmc import hmc
from mcmc.models import (
    ConjugateLinearRegression,
    EightSchoolsNonCentered,
    make_eight_schools_gibbs_updates,
)
from mcmc.targets import finite_difference_grad, finite_difference_hess


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


def test_eight_schools_hessian_matches_finite_differences():
    """The Hessian is hand-derived from the hand-derived gradient, so nothing
    upstream would catch a slip in it: the log-tau row carries an e^t and an
    e^{2t} term of opposite sign, and the eta block's 1 + tau^2/sigma_j^2 is
    the entry the metric study reads. Checked at a spread of log tau, since
    every error term in the derivation is a function of it."""
    model = EightSchoolsNonCentered()
    rng = np.random.default_rng(11)
    z = rng.standard_normal((8, model.dim))
    z[:, 1] = rng.uniform(-1.0, 2.0, size=8)
    H = model.hess_logpdf(z)
    np.testing.assert_allclose(
        H, finite_difference_hess(model.grad_logpdf, z), rtol=2e-5, atol=2e-6
    )
    # symmetric exactly, not just to the finite-difference tolerance: the two
    # off-diagonal blocks are assigned from one array each, so a typo that
    # transposed one of them would show here and nowhere else.
    np.testing.assert_array_equal(H, np.transpose(H, (0, 2, 1)))


def test_eight_schools_eta_block_is_the_prior_plus_tau_over_sigma():
    """The closed form the metric study is built on: d2L/deta_j deta_k is
    diagonal with entries -(1 + tau^2/sigma_j^2). It is asserted separately
    from the finite-difference check because the *shape* of this block is the
    claim -- that the eta curvature depends on position only through tau -- and
    a Hessian that was right to 1e-5 with a spurious eta_j eta_k coupling would
    pass the check above and refute the section."""
    model = EightSchoolsNonCentered()
    rng = np.random.default_rng(12)
    z = rng.standard_normal((5, model.dim))
    z[:, 1] = np.linspace(-1.5, 1.5, 5)
    block = model.hess_logpdf(z)[:, 2:, 2:]
    tau2 = np.exp(2.0 * z[:, 1])
    expected = -(1.0 + tau2[:, None] / model.sigma2)
    for b, e in zip(block, expected):
        np.testing.assert_allclose(np.diag(b), e, rtol=1e-12)
        np.testing.assert_allclose(b - np.diag(np.diag(b)), 0.0, atol=0.0)


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
