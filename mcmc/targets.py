"""Test distributions with analytically known structure.

Each target implements the protocol from ``mcmc.base`` (batched ``logpdf`` and
``grad_logpdf``) and exposes whatever ground truth it has (moments, exact
marginals) so sampler output can be checked against exact answers rather than
eyeballed.
"""

from __future__ import annotations

import math
from typing import Callable

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

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = np.atleast_1d(np.asarray(mean, dtype=float))
        self.cov = np.atleast_2d(np.asarray(cov, dtype=float))
        self.dim = self.mean.shape[0]
        L = np.linalg.cholesky(self.cov)
        self._chol = L
        self.precision = np.linalg.inv(self.cov)
        self._logdet = 2.0 * np.sum(np.log(np.diag(L)))
        self._lognorm = -0.5 * (self.dim * np.log(2.0 * np.pi) + self._logdet)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        delta = np.atleast_2d(x) - self.mean
        quad = np.einsum("ni,ij,nj->n", delta, self.precision, delta)
        return self._lognorm - 0.5 * quad

    def grad_logpdf(self, x: np.ndarray) -> np.ndarray:
        delta = np.atleast_2d(x) - self.mean
        return -delta @ self.precision  # precision is symmetric

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Exact i.i.d. draws x = mu + L z, z ~ N(0, I). Used as a reference."""
        z = rng.standard_normal((n, self.dim))
        return self.mean + z @ self._chol.T


class GaussianMixture:
    """Mixture of Gaussians sum_k w_k N(mu_k, Sigma_k) -- the standard
    *multimodal* benchmark.

    Well-separated components are the classic failure case for a single-chain
    random walk or unit-metric HMC: the sampler falls into whichever mode it
    starts in and essentially never crosses the low-density barrier between
    modes, so it reports one mode with wildly wrong weights. It is the target
    parallel tempering (``mcmc.tempering``) is built to handle.

    Log-density via log-sum-exp for stability:

        log p(x) = logsumexp_k [ log w_k + log N(x; mu_k, Sigma_k) ].

    Gradient (hand-derived; checked against finite differences). With the
    responsibilities r_k(x) = w_k N_k(x) / sum_j w_j N_j(x) (a softmax of the
    per-component log-densities, so the normalizer cancels),

        grad log p(x) = -sum_k r_k(x) Sigma_k^{-1} (x - mu_k),

    a responsibility-weighted average of the per-component score functions.
    """

    def __init__(self, weights: np.ndarray, means: np.ndarray, covs: np.ndarray):
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / self.weights.sum()
        self.means = np.atleast_2d(np.asarray(means, dtype=float))
        self.n_comp, self.dim = self.means.shape
        covs = np.asarray(covs, dtype=float)
        if covs.ndim == 2:  # one shared covariance -> broadcast
            covs = np.broadcast_to(covs, (self.n_comp, self.dim, self.dim))
        self.covs = covs
        self._prec = np.stack([np.linalg.inv(c) for c in covs])
        self._lognorm = np.array([
            -0.5 * (self.dim * np.log(2.0 * np.pi)
                    + 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(c)))))
            for c in covs
        ])
        self._logw = np.log(self.weights)

    def _log_components(self, x: np.ndarray) -> np.ndarray:
        """Per-component log(w_k N_k(x)); shape (n_chains, n_comp)."""
        x = np.atleast_2d(x)
        delta = x[:, None, :] - self.means[None, :, :]        # (n, K, d)
        quad = np.einsum("nki,kij,nkj->nk", delta, self._prec, delta)
        return self._logw[None, :] + self._lognorm[None, :] - 0.5 * quad

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        lc = self._log_components(x)
        m = lc.max(axis=1, keepdims=True)
        return (m[:, 0] + np.log(np.sum(np.exp(lc - m), axis=1)))

    def grad_logpdf(self, x: np.ndarray) -> np.ndarray:
        lc = self._log_components(x)
        r = np.exp(lc - lc.max(axis=1, keepdims=True))
        r = r / r.sum(axis=1, keepdims=True)                  # responsibilities (n, K)
        delta = np.atleast_2d(x)[:, None, :] - self.means[None, :, :]
        scores = -np.einsum("kij,nkj->nki", self._prec, delta)  # per-comp score (n,K,d)
        return np.einsum("nk,nki->ni", r, scores)

    def mean(self) -> np.ndarray:
        return self.weights @ self.means

    def cov(self) -> np.ndarray:
        """Exact covariance: sum_k w_k (Sigma_k + mu_k mu_k^T) - mu mu^T."""
        mu = self.mean()
        second = sum(
            self.weights[k] * (self.covs[k] + np.outer(self.means[k], self.means[k]))
            for k in range(self.n_comp)
        )
        return second - np.outer(mu, mu)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Exact draws: pick a component by weight, then sample it."""
        comp = rng.choice(self.n_comp, size=n, p=self.weights)
        out = np.empty((n, self.dim))
        for k in range(self.n_comp):
            m = comp == k
            L = np.linalg.cholesky(self.covs[k])
            out[m] = self.means[k] + rng.standard_normal((int(m.sum()), self.dim)) @ L.T
        return out


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

    def __init__(self, dim: int = 10, sigma_v: float = 3.0):
        if dim < 2:
            raise ValueError("funnel needs dim >= 2 (one v plus at least one x)")
        self.dim = dim
        self.sigma_v = float(sigma_v)

    # States far into the neck can overflow e^{-v} in float64; the resulting
    # -inf log-density is rejected by the Metropolis step, which is the
    # correct outcome -- silence only the warning (see mcmc.models for the
    # same pattern, explained).

    def logpdf(self, z: np.ndarray) -> np.ndarray:
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

    def grad_logpdf(self, z: np.ndarray) -> np.ndarray:
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

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
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

    def __init__(self, a: float = 1.0, b: float = 10.0):
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

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        x1, x2 = x[:, 0], x[:, 1]
        with np.errstate(over="ignore", invalid="ignore"):
            return self._lognorm - (x1 - self.a) ** 2 - self.b * (x2 - x1**2) ** 2

    def grad_logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        x1, x2 = x[:, 0], x[:, 1]
        g = np.empty_like(x)
        with np.errstate(over="ignore", invalid="ignore"):
            resid = x2 - x1**2
            g[:, 0] = -2.0 * (x1 - self.a) + 4.0 * self.b * x1 * resid
            g[:, 1] = -2.0 * self.b * resid
        return g

    def moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact (mean, cov) from the closed-form marginal/conditional above."""
        a, b = self.a, self.b
        mean = np.array([a, a**2 + 0.5])
        cov = np.array(
            [[0.5, a], [a, 1.0 / (2.0 * b) + 0.5 + 2.0 * a**2]]
        )
        return mean, cov

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Exact draws: x1 ~ N(a, 1/2), then x2 | x1 ~ N(x1^2, 1/(2b))."""
        x1 = self.a + rng.standard_normal(n) / np.sqrt(2.0)
        x2 = x1**2 + rng.standard_normal(n) / np.sqrt(2.0 * self.b)
        return np.column_stack([x1, x2])


class StudentT:
    """Multivariate Student-t: N(mu, Sigma)-shaped centre with polynomial tails.

    The heavy tails are the point. Where a Gaussian proposal or unit-metric HMC
    tuned to the bulk is calibrated, the same sampler badly under-visits a
    Student-t's tails: the density there decays like a power law, not an
    exponential, so occasional far excursions carry real probability mass that a
    bulk-scaled step rarely reaches. This is the standard heavy-tail mixing
    cautionary target, and a useful contrast to the (light-tailed) Gaussian.

    Density (scale matrix Sigma, dof nu), with q = (x-mu)^T Sigma^{-1} (x-mu):

        p(x) = C * (1 + q/nu)^{-(nu+d)/2},
        C = Gamma((nu+d)/2) / [ Gamma(nu/2) (nu pi)^{d/2} |Sigma|^{1/2} ].

    log p(x) = log C - (nu+d)/2 * log(1 + q/nu)
    grad log p(x) = -((nu+d)/nu) * Sigma^{-1}(x - mu) / (1 + q/nu)

    Note the score is the Gaussian score -Sigma^{-1}(x-mu) divided by
    (1 + q/nu): near the mode it matches a Gaussian, but far out the 1/(1+q/nu)
    factor softens the restoring force -- exactly why the tails are fat.

    Ground truth: Sigma is the *scale*, not the covariance. The mean is mu for
    nu > 1; the covariance is nu/(nu-2) * Sigma for nu > 2 (and is undefined
    below that). ``sample`` draws exact reference points via the Gaussian
    scale-mixture representation x = mu + (L z) / sqrt(w/nu), z ~ N(0, I),
    w ~ chi^2_nu, L = chol(Sigma).
    """

    def __init__(self, mean: np.ndarray, scale: np.ndarray, dof: float):
        self.mean = np.atleast_1d(np.asarray(mean, dtype=float))
        self.scale = np.atleast_2d(np.asarray(scale, dtype=float))
        self.dof = float(dof)
        self.dim = self.mean.shape[0]
        L = np.linalg.cholesky(self.scale)
        self._chol = L
        self.precision = np.linalg.inv(self.scale)
        self._logdet = 2.0 * np.sum(np.log(np.diag(L)))
        d, nu = self.dim, self.dof
        self._lognorm = (
            math.lgamma(0.5 * (nu + d))
            - math.lgamma(0.5 * nu)
            - 0.5 * d * math.log(nu * math.pi)
            - 0.5 * self._logdet
        )

    def _quad(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta = np.atleast_2d(x) - self.mean
        return np.einsum("ni,ij,nj->n", delta, self.precision, delta), delta

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        q, _ = self._quad(x)
        d, nu = self.dim, self.dof
        return self._lognorm - 0.5 * (nu + d) * np.log1p(q / nu)

    def grad_logpdf(self, x: np.ndarray) -> np.ndarray:
        q, delta = self._quad(x)
        d, nu = self.dim, self.dof
        coeff = -((nu + d) / nu) / (1.0 + q / nu)          # (n,)
        return coeff[:, None] * (delta @ self.precision)   # precision symmetric

    def moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact (mean, cov); cov requires nu > 2 (else raises)."""
        if self.dof <= 2.0:
            raise ValueError("covariance is undefined for dof <= 2")
        cov = self.dof / (self.dof - 2.0) * self.scale
        return self.mean.copy(), cov

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Exact draws via the Gaussian scale mixture (chi^2 mixing)."""
        z = rng.standard_normal((n, self.dim))
        w = rng.chisquare(self.dof, size=n)
        return self.mean + (z @ self._chol.T) * np.sqrt(self.dof / w)[:, None]


def finite_difference_grad(
    logpdf: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
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
