"""The metric algebra of Sec. 4.10, against closed forms and independent code.

Three things have to be right before a dense metric can be wired into a
sampler, and each fails in a way that is easy to miss: the kinetic energy (a
transpose in the wrong place still returns a plausible positive number), the
momentum draw (a wrong factor still produces a Gaussian, just of the wrong
covariance -- and the chain still runs, slightly wrong), and the identity case
(a metric option that perturbs results computed without one).
"""

import numpy as np
import pytest

from mcmc.hmc import leapfrog
from mcmc.metric import (
    DenseMetric,
    DiagonalMetric,
    IdentityMetric,
    invert_lower,
    metric_from,
)


def spd(rng, d, jitter=2.0):
    a = rng.standard_normal((d, d))
    return a @ a.T + jitter * np.eye(d)


# -- the triangular inverse --------------------------------------------------

def test_invert_lower_matches_a_general_solve():
    rng = np.random.default_rng(0)
    L = np.linalg.cholesky(spd(rng, 6))
    inv = invert_lower(L)
    assert np.allclose(inv @ L, np.eye(6), atol=1e-12)
    assert np.allclose(inv, np.linalg.solve(L, np.eye(6)), atol=1e-12)
    assert np.allclose(np.triu(inv, 1), 0.0)      # the inverse stays lower


def test_invert_lower_rejects_a_singular_factor():
    with pytest.raises(ValueError, match="singular"):
        invert_lower(np.array([[1.0, 0.0], [2.0, 0.0]]))
    with pytest.raises(ValueError, match="square"):
        invert_lower(np.ones((2, 3)))


# -- the three operations, against direct formulas ---------------------------

@pytest.mark.parametrize("d", [1, 3, 7])
def test_dense_operations_match_the_direct_quadratic_forms(d):
    rng = np.random.default_rng(d)
    sigma = spd(rng, d)
    m = DenseMetric(sigma)
    p = rng.standard_normal((5, d))
    assert np.allclose(m.velocity(p), p @ sigma)
    assert np.allclose(m.kinetic(p),
                       0.5 * np.einsum("ij,jk,ik->i", p, sigma, p))
    # and the batching is elementwise: one row alone gives the same answer
    assert np.allclose(m.kinetic(p[2]), m.kinetic(p)[2])
    assert np.allclose(m.velocity(p[2]), m.velocity(p)[2])


def test_the_momentum_draw_has_covariance_M_not_M_inverse():
    """The failure this catches: drawing from N(0, Sigma) instead of
    N(0, Sigma^-1) leaves a perfectly good sampler targeting the wrong joint.
    Checked against the algebra (exact) and against a large sample (noisy)."""
    rng = np.random.default_rng(1)
    sigma = spd(rng, 4)
    m = DenseMetric(sigma)
    mass = np.linalg.inv(sigma)                       # M = Sigma^-1
    # exact: Cov(z @ L^-1) = L^-T L^-1 = (L L^T)^-1
    linv = m._inv_chol
    assert np.allclose(linv.T @ linv, mass, atol=1e-10)
    draws = m.draw_momentum(rng, (200_000,))
    assert draws.shape == (200_000, 4)
    assert np.allclose(np.cov(draws.T), mass, atol=0.02 * np.abs(mass).max() + 0.01)


def test_the_augmented_target_factorizes_correctly():
    """The point of drawing p ~ N(0, M): the mean kinetic energy is then dim/2,
    whatever the metric, since K = z^T z / 2 in whitened coordinates."""
    rng = np.random.default_rng(2)
    for m in (IdentityMetric(5), DiagonalMetric(np.array([0.1, 4.0, 1.0, 9.0, 0.5])),
              DenseMetric(spd(rng, 5))):
        k = m.kinetic(m.draw_momentum(rng, (100_000,)))
        assert k.mean() == pytest.approx(2.5, rel=0.02)


# -- the identity case -------------------------------------------------------

def test_a_dense_identity_metric_reproduces_the_identity_metric_exactly():
    """A metric option must not perturb results computed without one; with an
    exact-float identity the mat-vecs are bit-for-bit, not merely close."""
    rng = np.random.default_rng(3)
    p = rng.standard_normal((4, 6))
    ident, dense = IdentityMetric(6), DenseMetric(np.eye(6))
    diag = DiagonalMetric(np.ones(6))
    for m in (dense, diag):
        assert np.array_equal(m.velocity(p), ident.velocity(p))
        assert np.array_equal(m.kinetic(p), ident.kinetic(p))
    # and the momentum draws agree draw for draw from the same seed
    a = ident.draw_momentum(np.random.default_rng(7), (3,))
    b = dense.draw_momentum(np.random.default_rng(7), (3,))
    c = diag.draw_momentum(np.random.default_rng(7), (3,))
    assert np.array_equal(a, b) and np.array_equal(a, c)


def test_diagonal_metric_agrees_with_the_inline_arithmetic_the_samplers_use():
    """hmc.py computed these three quantities inline as `inv_mass * p`,
    `(eps * inv_mass) * p` and `sum(inv_mass * p**2) / 2` before this module
    existed, and the object must reproduce them *exactly* -- association
    included. A different-but-equivalent association is a one-ulp change, and
    an HMC chain is chaotic enough to turn that into a different chain (see
    tests/test_hmc.py), so every measurement already committed under the
    diagonal metric depends on these being equalities and not tolerances."""
    rng = np.random.default_rng(4)
    v = np.exp(rng.standard_normal(5))
    p = rng.standard_normal((3, 5))
    m = DiagonalMetric(v)
    assert np.array_equal(m.velocity(p), v * p)
    assert np.array_equal(m.scaled_velocity(p, 0.3), (0.3 * v) * p)
    assert np.array_equal(m.kinetic(p), 0.5 * np.sum(v * p**2, axis=-1))
    # a dense metric holding the same diagonal agrees to roundoff, not exactly:
    # (eps * v) * p against eps * (p @ diag(v)) is the same real number
    # reassociated, and `sum((p @ diag(sqrt v))**2)` against `sum(v * p**2)`
    # likewise.
    dense = DenseMetric(np.diag(v))
    assert np.allclose(dense.kinetic(p), m.kinetic(p), rtol=1e-14, atol=0)
    assert np.allclose(dense.scaled_velocity(p, 0.3),
                       m.scaled_velocity(p, 0.3), rtol=1e-14, atol=0)


# -- what the metric is for --------------------------------------------------

def whitened_conditioning(inv_mass, precision):
    """kappa of the Hessian leapfrog actually integrates under a metric.

    Sec. 4.10: metric M turns the target into identity-metric HMC on
    y = M^1/2 x, whose Hessian M^-1/2 A M^-1/2 is *symmetric* -- so its
    condition number is the eigenvalue ratio, and that is what sets the step
    size. Not `cond(M^-1 A)`: the product is not symmetric, and its singular
    values are not its eigenvalues (at scale ratio 3000 they differ by six
    orders of magnitude, which is how this was found).
    """
    root = np.linalg.cholesky(inv_mass)               # M^-1 = R R^T
    h = root.T @ precision @ root                     # similar to M^-1 A
    ev = np.linalg.eigvalsh(h)
    return ev.max() / ev.min()


def test_the_diagonal_metric_leaves_exactly_the_correlation_conditioning():
    """Sec. 4.10's closed form: with M^-1 = diag(Sigma) the whitened Hessian is
    similar to R^-1, so the leftover conditioning is kappa(R), the correlation
    matrix's own -- (1+rho)/(1-rho) in 2-D, at any marginal scales. The dense
    metric gives exactly 1."""
    for rho in (0.5, 0.9, 0.99):
        for s1, s2 in ((1.0, 1.0), (0.01, 30.0)):        # any marginal scales
            sigma = np.array([[s1**2, rho * s1 * s2],
                              [rho * s1 * s2, s2**2]])
            a = np.linalg.inv(sigma)                      # target precision
            diag = np.diag(np.diag(sigma))                # Sec. 4.8's metric
            kappa_diag = whitened_conditioning(diag, a)
            assert kappa_diag == pytest.approx((1 + rho) / (1 - rho), rel=1e-8)
            assert whitened_conditioning(sigma, a) == pytest.approx(1.0, rel=1e-8)
            # and it is the correlation matrix's own conditioning, in any d
            corr = sigma / np.outer([s1, s2], [s1, s2])
            assert kappa_diag == pytest.approx(np.linalg.cond(corr), rel=1e-8)
            # the scale disparity IS removed: the unit metric is far worse
            unit = whitened_conditioning(np.eye(2), a)
            assert unit > kappa_diag or s1 == s2


def test_the_correlation_conditioning_result_holds_in_higher_dimensions():
    """kappa(R) is not a 2-D coincidence: for a random SPD Sigma at any scales,
    the diagonally-whitened Hessian conditions exactly as the correlation
    matrix does."""
    rng = np.random.default_rng(11)
    for d in (3, 6):
        sigma = spd(rng, d)
        scales = np.exp(rng.standard_normal(d) * 2)      # wildly unequal
        sigma = sigma * np.outer(scales, scales)
        a = np.linalg.inv(sigma)
        sd = np.sqrt(np.diag(sigma))
        corr = sigma / np.outer(sd, sd)
        assert whitened_conditioning(np.diag(np.diag(sigma)), a) == pytest.approx(
            np.linalg.cond(corr), rel=1e-6)
        assert whitened_conditioning(sigma, a) == pytest.approx(1.0, rel=1e-6)


def test_dense_metric_whitens_a_correlated_gaussian_trajectory():
    """Exercise 4(b) for dense M, as an executable statement: leapfrog with
    metric M on the correlated target and identity leapfrog on the whitened
    target trace the same trajectory, coordinate for coordinate."""
    rng = np.random.default_rng(5)
    sigma = spd(rng, 3)
    a = np.linalg.inv(sigma)
    m = DenseMetric(sigma)
    chol = m.chol                                   # Sigma = L L^T, M^1/2 = L^-T
    x0 = rng.standard_normal(3)
    p0 = m.draw_momentum(rng, ())

    # metric-M leapfrog, written out with the metric's own operations
    x, p = x0.copy(), p0.copy()
    eps, n = 0.05, 20
    p = p + 0.5 * eps * (-x @ a)
    for k in range(n):
        x = x + eps * m.velocity(p)
        if k < n - 1:
            p = p + eps * (-x @ a)
    p = p + 0.5 * eps * (-x @ a)

    # identity leapfrog on y = M^1/2 x = L^-1 x, whose precision is L^T A L
    y0 = np.linalg.solve(chol, x0)
    q0 = chol.T @ p0
    a_y = chol.T @ a @ chol
    y, q = leapfrog(lambda z: -z @ a_y, y0, q0, eps, n)

    assert np.allclose(np.linalg.solve(chol, x), y, atol=1e-10)
    assert np.allclose(chol.T @ p, q, atol=1e-10)


# -- construction ------------------------------------------------------------

def test_metric_from_dispatches_on_shape():
    assert isinstance(metric_from(None, dim=3), IdentityMetric)
    assert isinstance(metric_from(np.ones(3)), DiagonalMetric)
    assert isinstance(metric_from(np.eye(3)), DenseMetric)
    with pytest.raises(ValueError, match="dim is required"):
        metric_from(None)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        metric_from(np.ones((2, 2, 2)))


def test_bad_metrics_fail_at_construction_not_inside_a_trajectory():
    with pytest.raises(ValueError, match="positive definite"):
        DenseMetric(np.array([[1.0, 2.0], [2.0, 1.0]]))       # indefinite
    with pytest.raises(ValueError, match="positive"):
        DiagonalMetric(np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="square"):
        DenseMetric(np.ones((2, 3)))


def test_an_asymmetric_estimate_is_symmetrized_rather_than_trusted():
    """A sample covariance is symmetric only up to roundoff; a metric built
    from one must not depend on which triangle it happened to read."""
    base = np.array([[2.0, 0.5], [0.5, 1.0]])
    skew = base + np.array([[0.0, 1e-9], [-1e-9, 0.0]])
    assert np.allclose(DenseMetric(skew).cov, base, atol=1e-12)
