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
