"""Targets must have correct log-densities and hand-derived gradients."""

import numpy as np
import pytest

from mcmc.targets import Gaussian, NealsFunnel, Rosenbrock, finite_difference_grad

RNG = np.random.default_rng(0)


def make_gaussian():
    mean = np.array([1.0, -2.0, 0.5])
    A = RNG.standard_normal((3, 3))
    cov = A @ A.T + 3.0 * np.eye(3)
    return Gaussian(mean, cov)


def test_gaussian_logpdf_matches_direct_formula():
    g = make_gaussian()
    x = RNG.standard_normal((5, 3))
    sign, logdet = np.linalg.slogdet(g.cov)
    assert sign > 0
    delta = x - g.mean
    quad = np.einsum("ni,ni->n", delta @ np.linalg.inv(g.cov), delta)
    expected = -0.5 * (3 * np.log(2 * np.pi) + logdet + quad)
    np.testing.assert_allclose(g.logpdf(x), expected, rtol=1e-12)


def test_gaussian_exact_sampler_moments():
    g = make_gaussian()
    xs = g.sample(200_000, np.random.default_rng(1))
    np.testing.assert_allclose(xs.mean(axis=0), g.mean, atol=0.03)
    np.testing.assert_allclose(np.cov(xs.T), g.cov, atol=0.08)


def test_funnel_v_marginal_is_exact_normal():
    f = NealsFunnel(dim=10, sigma_v=3.0)
    zs = f.sample(200_000, np.random.default_rng(2))
    v = zs[:, 0]
    assert abs(v.mean()) < 0.05
    assert abs(v.std() - 3.0) < 0.05


@pytest.mark.parametrize(
    "target,scale",
    [
        (make_gaussian(), 2.0),
        (NealsFunnel(dim=6, sigma_v=3.0), 1.5),
        (Rosenbrock(a=1.0, b=10.0), 1.0),
    ],
    ids=["gaussian", "funnel", "rosenbrock"],
)
def test_gradients_match_finite_differences(target, scale):
    x = RNG.standard_normal((8, target.dim)) * scale
    analytic = target.grad_logpdf(x)
    numeric = finite_difference_grad(target.logpdf, x)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_rosenbrock_is_normalized():
    """The stated log-normalizer makes exp(logpdf) integrate to 1 (2D grid)."""
    t = Rosenbrock(a=1.0, b=10.0)
    xs = np.linspace(-3.0, 5.0, 900)
    ys = np.linspace(-3.0, 14.0, 1400)
    X, Y = np.meshgrid(xs, ys)
    p = np.exp(t.logpdf(np.column_stack([X.ravel(), Y.ravel()]))).reshape(X.shape)
    integral = np.trapezoid(np.trapezoid(p, ys, axis=0), xs)
    assert abs(integral - 1.0) < 1e-3


def test_rosenbrock_exact_sampler_matches_closed_form_moments():
    """The generative sampler (x1 ~ N(a,1/2), x2|x1 ~ N(x1^2,1/2b)) reproduces
    the hand-derived mean and covariance."""
    t = Rosenbrock(a=1.0, b=10.0)
    xs = t.sample(2_000_000, np.random.default_rng(7))
    mean, cov = t.moments()
    np.testing.assert_allclose(xs.mean(axis=0), mean, atol=0.02)
    np.testing.assert_allclose(np.cov(xs.T), cov, atol=0.03)


def test_rosenbrock_x1_marginal_is_normal():
    """The b-term integrates out, so x1 is exactly N(a, 1/2) regardless of b."""
    t = Rosenbrock(a=1.0, b=10.0)
    x1 = t.sample(500_000, np.random.default_rng(8))[:, 0]
    assert abs(x1.mean() - 1.0) < 0.01
    assert abs(x1.std() - np.sqrt(0.5)) < 0.01
