"""Test distributions with analytically known structure.

Each target implements the protocol from ``mcmc.base`` (batched ``logpdf`` and
``grad_logpdf``) and exposes whatever ground truth it has (moments, exact
marginals) so sampler output can be checked against exact answers rather than
eyeballed.
"""

import numpy as np


class Gaussian:
    """Multivariate normal N(mu, cov), the exactly-solvable benchmark.

    log p(x) = -1/2 [ d log(2 pi) + log|Sigma| + (x-mu)^T Sigma^{-1} (x-mu) ]
    grad log p(x) = -Sigma^{-1} (x - mu)

    The precision matrix is precomputed from a Cholesky factorization
    (Sigma = L L^T), which also gives log|Sigma| = 2 sum_i log L_ii.
    Explicitly forming Sigma^{-1} is fine at the small dimensions used here;
    at scale you would keep L and use triangular solves instead.
    """

    def __init__(self, mean, cov):
        self.mean = np.atleast_1d(np.asarray(mean, dtype=float))
        self.cov = np.atleast_2d(np.asarray(cov, dtype=float))
        self.dim = self.mean.shape[0]
        L = np.linalg.cholesky(self.cov)
        self._chol = L
        self.precision = np.linalg.inv(self.cov)
        self._logdet = 2.0 * np.sum(np.log(np.diag(L)))
        self._lognorm = -0.5 * (self.dim * np.log(2.0 * np.pi) + self._logdet)

    def logpdf(self, x):
        delta = np.atleast_2d(x) - self.mean
        quad = np.einsum("ni,ij,nj->n", delta, self.precision, delta)
        return self._lognorm - 0.5 * quad

    def grad_logpdf(self, x):
        delta = np.atleast_2d(x) - self.mean
        return -delta @ self.precision  # precision is symmetric

    def sample(self, n, rng):
        """Exact i.i.d. draws x = mu + L z, z ~ N(0, I). Used as a reference."""
        z = rng.standard_normal((n, self.dim))
        return self.mean + z @ self._chol.T


class NealsFunnel:
    """Neal's funnel (Neal 2003, "Slice sampling", Ann. Statist.).

    v ~ N(0, sigma_v^2),   x_i | v ~ N(0, e^v)  for i = 1..dim-1.

    The state is z = (v, x_1, ..., x_{dim-1}). The conditional scale of the
    x_i varies over ~e^{3 sigma_v} across the prior range of v, so no single
    proposal scale (RWMH) or step size (unit-metric HMC) fits both the wide
    mouth and the narrow neck. This mimics the geometry of hierarchical
    posteriors (funnel in (log tau, theta)) and is the standard hard case.

    Ground truth: the marginal of v is exactly N(0, sigma_v^2), which gives a
    sharp correctness check even though the joint has no closed-form moments
    a sampler finds easy.

    log p(z) = log N(v; 0, sigma_v^2) + sum_i log N(x_i; 0, e^v)
             = const - v^2/(2 sigma_v^2) - (dim-1) v/2 - e^{-v}/2 * sum_i x_i^2
    d log p / dv  = -v/sigma_v^2 - (dim-1)/2 + e^{-v}/2 * sum_i x_i^2
    d log p / dx_i = -x_i e^{-v}
    """

    def __init__(self, dim=10, sigma_v=3.0):
        if dim < 2:
            raise ValueError("funnel needs dim >= 2 (one v plus at least one x)")
        self.dim = dim
        self.sigma_v = float(sigma_v)

    # States far into the neck can overflow e^{-v} in float64; the resulting
    # -inf log-density is rejected by the Metropolis step, which is the
    # correct outcome -- silence only the warning (see mcmc.models for the
    # same pattern, explained).

    def logpdf(self, z):
        z = np.atleast_2d(z)
        v, x = z[:, 0], z[:, 1:]
        sumsq = np.sum(x * x, axis=1)
        k = self.dim - 1
        with np.errstate(over="ignore", invalid="ignore"):
            return (
                -0.5 * v**2 / self.sigma_v**2
                - 0.5 * k * v
                - 0.5 * np.exp(-v) * sumsq
                - 0.5 * k * np.log(2.0 * np.pi)
                - 0.5 * np.log(2.0 * np.pi * self.sigma_v**2)
            )

    def grad_logpdf(self, z):
        z = np.atleast_2d(z)
        v, x = z[:, 0], z[:, 1:]
        with np.errstate(over="ignore", invalid="ignore"):
            e_neg_v = np.exp(-v)
            g = np.empty_like(z)
            g[:, 0] = (
                -v / self.sigma_v**2
                - 0.5 * (self.dim - 1)
                + 0.5 * e_neg_v * np.sum(x * x, axis=1)
            )
            g[:, 1:] = -x * e_neg_v[:, None]
        return g

    def sample(self, n, rng):
        """Exact draws via the generative process (v first, then x | v)."""
        v = rng.standard_normal(n) * self.sigma_v
        x = rng.standard_normal((n, self.dim - 1)) * np.exp(0.5 * v)[:, None]
        return np.column_stack([v, x])


class Rosenbrock:
    """The banana-shaped Rosenbrock density, a curved-geometry stress test.

    log p(x1, x2) = -(x1 - a)^2 - b (x2 - x1^2)^2 + const

    The exp of minus the classic Rosenbrock valley: a thin parabolic ridge
    x2 ~ x1^2. Its curvature is exactly the pathology gradient methods on this
    density must cope with -- the local covariance rotates along the arc, so a
    single fixed HMC step size and mass are never ideal (same lesson as the
    funnel, but from banana curvature rather than a varying scale).

    Ground truth is unusually complete here because the b-term is Gaussian in
    x2 and integrates out cleanly:

        x1        ~ N(a, 1/2)             (marginal; the b-term contributes a
                                           constant sqrt(pi/b) independent of x1)
        x2 | x1   ~ N(x1^2, 1/(2b))       (Gaussian conditional)

    so the exact moments are known in closed form (derived in
    theory/derivations.md via E[x1^2], Var[x1^2], Cov[x1, x1^2] for a normal):

        E[x1] = a,                  Var[x1] = 1/2
        E[x2] = a^2 + 1/2,          Var[x2] = 1/(2b) + 1/2 + 2 a^2
        Cov[x1, x2] = a

    and ``sample`` draws exact reference points via that generative process
    (x1 first, then x2 | x1) -- the answer key the samplers are checked against.

    Gradient (hand-derived):
        d log p / dx1 = -2 (x1 - a) + 4 b x1 (x2 - x1^2)
        d log p / dx2 = -2 b (x2 - x1^2)
    """

    dim = 2

    def __init__(self, a=1.0, b=10.0):
        # b sets the ridge thinness: x2|x1 has sd 1/sqrt(2b). b=10 is a clearly
        # curved but sample-able banana; the classic Rosenbrock uses b=100, a
        # ridge so thin that fixed-step HMC struggles badly (that is the point).
        self.a = float(a)
        self.b = float(b)
        # log normalizer: integral of exp(log p) = sqrt(pi) * sqrt(pi/b)
        self._lognorm = -0.5 * (np.log(np.pi) + np.log(np.pi / self.b))

    # A divergent leapfrog trajectory can send x1 far enough that x1**2
    # overflows float64; the resulting +/-inf log-density is rejected by the
    # Metropolis step (the correct outcome), so only the warning is silenced --
    # the same pattern used by NealsFunnel above.

    def logpdf(self, x):
        x = np.atleast_2d(x)
        x1, x2 = x[:, 0], x[:, 1]
        with np.errstate(over="ignore", invalid="ignore"):
            return self._lognorm - (x1 - self.a) ** 2 - self.b * (x2 - x1**2) ** 2

    def grad_logpdf(self, x):
        x = np.atleast_2d(x)
        x1, x2 = x[:, 0], x[:, 1]
        g = np.empty_like(x)
        with np.errstate(over="ignore", invalid="ignore"):
            resid = x2 - x1**2
            g[:, 0] = -2.0 * (x1 - self.a) + 4.0 * self.b * x1 * resid
            g[:, 1] = -2.0 * self.b * resid
        return g

    def moments(self):
        """Exact (mean, cov) from the closed-form marginal/conditional above."""
        a, b = self.a, self.b
        mean = np.array([a, a**2 + 0.5])
        cov = np.array(
            [[0.5, a], [a, 1.0 / (2.0 * b) + 0.5 + 2.0 * a**2]]
        )
        return mean, cov

    def sample(self, n, rng):
        """Exact draws: x1 ~ N(a, 1/2), then x2 | x1 ~ N(x1^2, 1/(2b))."""
        x1 = self.a + rng.standard_normal(n) / np.sqrt(2.0)
        x2 = x1**2 + rng.standard_normal(n) / np.sqrt(2.0 * self.b)
        return np.column_stack([x1, x2])


def finite_difference_grad(logpdf, x, eps=1e-6):
    """Central-difference gradient of a batched logpdf, for gradient checks.

    (f(x + eps e_i) - f(x - eps e_i)) / (2 eps) has O(eps^2) truncation error;
    with eps ~ 1e-6 in float64, truncation and roundoff are balanced near 1e-9.
    """
    x = np.atleast_2d(x)
    g = np.empty_like(x)
    for i in range(x.shape[1]):
        dx = np.zeros_like(x)
        dx[:, i] = eps
        g[:, i] = (logpdf(x + dx) - logpdf(x - dx)) / (2.0 * eps)
    return g
