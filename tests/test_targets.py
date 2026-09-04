"""Targets must have correct log-densities and hand-derived gradients."""

import numpy as np
import pytest

from mcmc.targets import (
    Gaussian,
    GaussianMixture,
    NealsFunnel,
    Rosenbrock,
    StudentT,
    finite_difference_grad,
    finite_difference_hess,
)

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
        (StudentT(mean=[0.5, -1.0], scale=[[1.0, 0.3], [0.3, 2.0]], dof=5.0), 2.0),
    ],
    ids=["gaussian", "funnel", "rosenbrock", "studentt"],
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


def make_studentt(dof=5.0):
    return StudentT(mean=[0.5, -1.0], scale=[[1.0, 0.3], [0.3, 2.0]], dof=dof)


def test_studentt_is_normalized():
    """The multivariate-t normalizer makes exp(logpdf) integrate to 1."""
    t = make_studentt()
    xs = np.linspace(-30.0, 31.0, 1600)
    ys = np.linspace(-40.0, 38.0, 1600)
    X, Y = np.meshgrid(xs, ys)
    p = np.exp(t.logpdf(np.column_stack([X.ravel(), Y.ravel()]))).reshape(X.shape)
    integral = np.trapezoid(np.trapezoid(p, ys, axis=0), xs)
    assert abs(integral - 1.0) < 2e-3


def test_studentt_exact_sampler_matches_moments():
    """The Gaussian scale-mixture sampler reproduces mean = mu and
    cov = dof/(dof-2) * scale (finite because dof=5 > 2)."""
    t = make_studentt(dof=5.0)
    xs = t.sample(4_000_000, np.random.default_rng(11))
    mean, cov = t.moments()
    np.testing.assert_allclose(xs.mean(axis=0), mean, atol=0.02)
    np.testing.assert_allclose(np.cov(xs.T), cov, rtol=0.03)


def test_studentt_has_heavier_tails_than_matched_gaussian():
    """Beyond the matched Gaussian's 99.9% radius, the t carries far more mass --
    the property that makes it a heavy-tail mixing test."""
    t = make_studentt(dof=5.0)
    _, cov = t.moments()
    g = Gaussian(mean=[0.5, -1.0], cov=cov)  # same first two moments
    mu = np.array([0.5, -1.0])
    xt = t.sample(2_000_000, np.random.default_rng(12))
    xg = g.sample(2_000_000, np.random.default_rng(13))
    rt = np.linalg.norm(xt - mu, axis=1)
    rg = np.linalg.norm(xg - mu, axis=1)
    thr = np.quantile(rg, 0.999)
    assert (rt > thr).mean() > 5 * (rg > thr).mean()


def test_studentt_covariance_undefined_below_dof_2():
    with pytest.raises(ValueError):
        StudentT(mean=[0.0], scale=[[1.0]], dof=1.5).moments()


def make_mixture():
    return GaussianMixture(
        weights=[0.3, 0.7],
        means=[[-4.0, 0.0], [4.0, 1.0]],
        covs=[[[1.0, 0.4], [0.4, 1.0]], [[0.6, 0.0], [0.0, 1.5]]],
    )


def test_mixture_gradient_matches_finite_differences():
    """The responsibility-weighted score is the easy place to drop the
    softmax normalizer or a Sigma^{-1}; finite differences catch it."""
    gm = make_mixture()
    rng = np.random.default_rng(1)
    x = rng.uniform(-6, 6, size=(10, 2))  # spans both modes and the valley
    np.testing.assert_allclose(
        gm.grad_logpdf(x), finite_difference_grad(gm.logpdf, x), rtol=1e-5, atol=1e-6
    )


def test_mixture_logpdf_normalizes_and_matches_bruteforce():
    gm = make_mixture()
    # against an explicit sum of scipy-free Gaussian densities at a few points
    pts = np.array([[-4.0, 0.0], [4.0, 1.0], [0.0, 0.0]])
    brute = np.zeros(len(pts))
    for k in range(gm.n_comp):
        d = pts - gm.means[k]
        prec = np.linalg.inv(gm.covs[k])
        lognorm = -0.5 * (2 * np.log(2 * np.pi) + np.log(np.linalg.det(gm.covs[k])))
        brute += gm.weights[k] * np.exp(lognorm - 0.5 * np.einsum("ni,ij,nj->n", d, prec, d))
    np.testing.assert_allclose(gm.logpdf(pts), np.log(brute), rtol=1e-10)


def test_mixture_exact_moments_match_sampler():
    gm = make_mixture()
    draws = gm.sample(400_000, np.random.default_rng(2))
    np.testing.assert_allclose(draws.mean(axis=0), gm.mean(), atol=0.03)
    np.testing.assert_allclose(np.cov(draws.T), gm.cov(), atol=0.06)


# -- Neal's funnel: the closed forms Sec. 4.12 argues from -----------------


def test_funnel_hessian_matches_finite_differences():
    """The hand-derived Hessian, against a difference of the exact gradient.

    Checked at draws from the funnel itself rather than at a standard normal:
    the e^{-v} entries span orders of magnitude across the target, and a
    Hessian that were right only near v = 0 would pass an easier test.
    """
    f = NealsFunnel(dim=6, sigma_v=1.5)
    z = f.sample(12, np.random.default_rng(11))
    H = f.hess_logpdf(z)
    fd = finite_difference_hess(f.grad_logpdf, z, eps=1e-5)
    scale = np.maximum(np.abs(fd).max(axis=(1, 2), keepdims=True), 1.0)
    assert np.allclose(H / scale, fd / scale, atol=2e-6)
    # not symmetrized on the way out, so this checks the derivation too
    assert np.allclose(H, H.transpose(0, 2, 1), atol=0.0)


def test_funnel_hessian_obeys_its_scaling_congruence():
    """A(T_c z) = D_c A(z) D_c exactly, for the funnel's own scaling map.

    This is the identity Sec. 4.12's whole argument rests on -- moving along
    the funnel's spine is the same as rescaling a metric's x block -- so it is
    asserted to machine precision rather than plotted.
    """
    f = NealsFunnel(dim=7, sigma_v=3.0)
    rng = np.random.default_rng(12)
    z = f.sample(8, rng)
    for c in (-2.5, -0.4, 0.0, 1.7, 4.0):
        shifted = np.column_stack([z[:, 0] + c, np.exp(0.5 * c) * z[:, 1:]])
        D = np.diag([1.0] + [np.exp(-0.5 * c)] * (f.dim - 1))
        lhs = -f.hess_logpdf(shifted)
        rhs = D @ (-f.hess_logpdf(z)) @ D
        assert np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_funnel_is_not_log_concave_where_it_lives():
    """-H is positive definite iff e^{-v}||x||^2 < 2/sigma_v^2, and it isn't.

    Both halves: the Schur-complement criterion agrees with a direct eigenvalue
    test, and essentially no draw from the funnel satisfies it. The second half
    is what forces ``whitened_abs_condition_numbers`` to exist.
    """
    f = NealsFunnel(dim=10, sigma_v=3.0)
    z = f.sample(20_000, np.random.default_rng(13))
    A = -f.hess_logpdf(z)
    pd = np.linalg.eigvalsh(A)[:, 0] > 0.0
    criterion = np.exp(-z[:, 0]) * np.sum(z[:, 1:] ** 2, axis=1) < 2.0 / f.sigma_v**2
    assert np.array_equal(pd, criterion)
    assert pd.mean() < 1e-3          # measured: 0 of 20k at these settings


def test_funnel_moments_are_diagonal_and_match_the_sampler():
    """The exact covariance, against a large exact-sampler estimate.

sigma_v = 1.2 rather than the default 3 for a reason that is itself part
    of the result: Var(x_i^2) = 3 e^{2 sigma_v^2}, so at sigma_v = 3 even a
    million exact draws estimate Var(x_i) only to 5%, 33%, 9% at three seeds.
    A sharp check of the *formula* needs a setting where Monte Carlo error is
    small; the slow-estimator fact gets its own assertion below.
    """
    f = NealsFunnel(dim=5, sigma_v=1.2)
    mean, cov = f.moments()
    assert np.allclose(cov, np.diag(np.diag(cov)), atol=0.0)     # exactly diagonal
    assert cov[0, 0] == pytest.approx(1.2**2)
    assert cov[1, 1] == pytest.approx(np.exp(0.5 * 1.2**2))
    z = f.sample(400_000, np.random.default_rng(14))
    assert np.allclose(z.mean(axis=0), mean, atol=0.05)
    emp = np.cov(z, rowvar=False)
    assert np.allclose(np.diag(emp), np.diag(cov), rtol=0.03)
    off = emp[np.triu_indices(f.dim, 1)]
    assert np.abs(off).max() < 0.05


def test_funnel_covariance_estimator_is_dominated_by_its_largest_draw():
    """Why an estimated dense metric for the funnel is noise, quantified.

    At sigma_v = 3 the sample variance of x_i inherits a variance of
    3 e^{18}, so a warmup-length window's answer is set by its single largest
    e^v. Two consequences, both measured across 40 independent 1000-draw
    windows at each of three seeds:

    - the spread between the largest and smallest window estimate ran 9.9x,
      51.4x and 54.1x -- itself that variable, so the assertion is the weakest
      of the three and the point is the order of magnitude, not the number;
    - the *median* window estimate ran 55.9, 56.7, 58.4 against a true
      e^{4.5} = 90.0. A typical window underestimates by ~1.6x, because the
      mean it is estimating lives in draws most windows do not contain. The
      bias is downward and it does not shrink with a smaller tolerance.
    """
    f = NealsFunnel(dim=10, sigma_v=3.0)
    truth = np.exp(0.5 * 3.0**2)
    for seed in (15, 16, 17):
        rng = np.random.default_rng(seed)
        ests, drop_ratio = [], []
        for _ in range(40):
            z = f.sample(1_000, rng)
            s = z[:, 1] ** 2
            ests.append(s.mean())
            drop_ratio.append(np.sort(s)[:-1].mean() / s.mean())
        assert max(ests) / min(ests) > 8.0
        assert np.median(ests) < 0.75 * truth       # measured 0.62-0.65x
        assert min(drop_ratio) < 0.7   # one draw carries >30% of some estimate


def _max_step(f, z, inv_mass):
    """Largest stable leapfrog step at z under metric M^-1 = inv_mass."""
    L = np.linalg.cholesky(inv_mass)
    A = -f.hess_logpdf(np.atleast_2d(z))[0]
    return 2.0 / np.sqrt(np.abs(np.linalg.eigvalsh(L.T @ A @ L)).max())


def test_funnel_step_size_limit_transfers_to_the_metric_exactly():
    """eps_max(T_c z; Sigma) = eps_max(z; D_c Sigma D_c), Sec. 4.12.

    The integrator form of the congruence: moving up the funnel's spine and
    rescaling the metric's x block are the *same* operation, so every metric
    faces one one-parameter family of problems and differs only in where along
    it it sits. Checked at the identity, the exact covariance, and a random
    dense metric, since a claim quantified over all Sigma should not be tested
    at one.
    """
    f = NealsFunnel(dim=10, sigma_v=3.0)
    rng = np.random.default_rng(21)
    z = f.sample(1, rng)[0]
    C = rng.standard_normal((10, 10))
    metrics = [np.eye(10), f.moments()[1], C @ C.T + np.eye(10)]
    for S in metrics:
        for c in (-6.0, -2.0, 0.0, 3.0, 7.0):
            shifted = np.concatenate([[z[0] + c], np.exp(0.5 * c) * z[1:]])
            D = np.diag([1.0] + [np.exp(-0.5 * c)] * 9)
            assert _max_step(f, shifted, S) == pytest.approx(
                _max_step(f, z, D @ S @ D), rel=1e-12
            )


def test_funnel_neck_step_size_collapses_like_exp_half_v_at_any_metric():
    """In the neck eps_max(v) ~ e^{v/2}, with the metric setting only the constant.

    Measured, not assumed: the deviation from the e^{(v2-v1)/2} law is under 3%
    from v = -9 down through v = -5 and under 0.3% below v = -7, at all three
    metrics. That is the sense in which no global metric fixes the funnel -- it
    slides the curve, it does not flatten it.
    """
    f = NealsFunnel(dim=10, sigma_v=3.0)
    rng = np.random.default_rng(22)
    z = f.sample(1, rng)[0]
    C = rng.standard_normal((10, 10))
    vs = np.array([-9.0, -7.0, -5.0])
    for S in (np.eye(10), f.moments()[1], C @ C.T + np.eye(10)):
        e = np.array([
            _max_step(f, np.concatenate([[z[0] + c], np.exp(0.5 * c) * z[1:]]), S)
            for c in vs
        ])
        law = (e / e[0]) / np.exp((vs - vs[0]) / 2.0)
        assert np.allclose(law, 1.0, rtol=0.03)
        assert law[1] == pytest.approx(1.0, rel=3e-3)   # v = -7
