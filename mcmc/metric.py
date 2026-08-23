"""The HMC metric: kinetic energy, drift velocity, and the momentum draw.

Every sampler here that uses a mass matrix does the same three things with it --
compute ``K(p) = p^T M^-1 p / 2``, drift by ``M^-1 p``, and draw ``p ~ N(0, M)``
-- and until now all three were written inline for a *diagonal* M, as elementwise
arithmetic on ``inv_mass`` (``hmc.py``, ``nuts.py``). A dense metric needs the
same three operations and cannot be written elementwise, so they move here,
behind one interface with a diagonal and a dense implementation.

The derivation is Sec. 4.10 of ``theory/derivations.md``; the two facts worth
repeating at the code are these.

**The metric is parameterized by M^-1, not M.** Warmup estimates a *covariance*
(Sec. 4.8), and M^-1 is what the drift and the kinetic energy want. Only the
momentum draw wants M, and it wants a square root of it. So the dense metric
stores one Cholesky factor of the covariance, ``Sigma = M^-1 = L L^T``, and
reads all three operations off it:

    velocity   M^-1 p          =  Sigma @ p
    kinetic    p^T M^-1 p / 2  =  ||L^T p||^2 / 2
    momentum   p ~ N(0, M)     =  L^-T z,   z ~ N(0, I)

The last line is the one to check rather than believe: Cov(L^-T z) = L^-T L^-1 =
(L L^T)^-1 = Sigma^-1 = M. It never forms M, which would be both wasted work and
a needless squaring of the condition number.

**Nothing about validity changes.** The drift map (x, p) -> (x + eps M^-1 p, p)
is a shear whatever M^-1 is -- Jacobian upper-triangular with unit diagonal,
determinant 1 -- so symplecticity, reversibility and hence the min(1, e^-dH)
accept rule are untouched (Exercise 4). A metric changes efficiency only.

``IdentityMetric`` exists so the samplers can hold a metric unconditionally
instead of branching on ``None``, and it is deliberately *arithmetic-free*: it
returns the arrays it is handed. A dense metric built from the identity matrix
also reproduces it bit for bit (multiplying by an exact-float identity is
exact), which is the check that a metric option cannot silently change results
that were computed without one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def invert_lower(L: np.ndarray) -> np.ndarray:
    """Inverse of a lower-triangular matrix, by forward substitution.

    Solves ``L X = I`` column by column. NumPy exposes no triangular solve, and
    handing ``np.linalg.solve`` a triangular matrix runs a general LU with
    partial pivoting -- 2/3 d^3 where substitution is d^3/3, on a matrix that
    is already factored. (``gp-from-scratch`` measured that exact waste at 5x
    when it happened inside a fit loop; here it happens once per metric, so the
    point is that the code says what it means.)

    Raises on a zero diagonal, which is the singular case a covariance estimate
    can reach in high dimensions and must not silently produce infinities.
    """
    L = np.asarray(L, dtype=float)
    d = L.shape[0]
    if L.shape != (d, d):
        raise ValueError(f"expected a square matrix, got {L.shape}")
    diag = np.diag(L)
    if not np.all(np.abs(diag) > 0):
        raise ValueError("triangular matrix is singular (zero on the diagonal)")
    X = np.zeros((d, d))
    for i in range(d):
        # row i of the inverse: X[i] = (e_i - L[i, :i] @ X[:i]) / L[i, i]
        X[i] = -(L[i, :i] @ X[:i]) / L[i, i]
        X[i, i] += 1.0 / L[i, i]
    return X


class Metric(ABC):
    """A mass matrix, used only through the three operations HMC needs.

    Momentum arrays are shaped ``(..., dim)`` throughout -- the samplers run
    several chains at once, so every operation is written to broadcast over
    leading axes and to reduce only over the last one.
    """

    dim: int

    @abstractmethod
    def velocity(self, p: np.ndarray) -> np.ndarray:
        """``M^-1 p`` -- the drift direction, and NUTS's U-turn vector."""

    @abstractmethod
    def kinetic(self, p: np.ndarray) -> np.ndarray:
        """``p^T M^-1 p / 2``, reduced over the last axis."""

    @abstractmethod
    def draw_momentum(self, rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        """A draw from ``N(0, M)`` of shape ``shape + (dim,)``."""

    @abstractmethod
    def inv_mass_matrix(self) -> np.ndarray:
        """``M^-1`` as a dense (dim, dim) array -- for reporting, not the loop."""


class IdentityMetric(Metric):
    """M = I. The arithmetic-free case, so an unadapted run costs nothing."""

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)

    def velocity(self, p: np.ndarray) -> np.ndarray:
        return p

    def kinetic(self, p: np.ndarray) -> np.ndarray:
        return 0.5 * np.sum(p * p, axis=-1)

    def draw_momentum(self, rng, shape):
        return rng.standard_normal((*shape, self.dim))

    def inv_mass_matrix(self) -> np.ndarray:
        return np.eye(self.dim)


class DiagonalMetric(Metric):
    """``M^-1 = diag(variance)`` -- Sec. 4.8's metric, as an object.

    Constructed from the per-coordinate variances warmup estimates, which is
    the same ``inv_mass`` vector the samplers already carry; the momentum sd is
    ``1/sqrt(variance)`` because ``M = diag(1/variance)``.
    """

    def __init__(self, variance: np.ndarray) -> None:
        v = np.asarray(variance, dtype=float)
        if v.ndim != 1:
            raise ValueError(f"expected a 1-D variance vector, got shape {v.shape}")
        if not np.all(v > 0):
            raise ValueError("every variance must be positive")
        self.dim = v.size
        self.variance = v
        self._momentum_sd = 1.0 / np.sqrt(v)

    def velocity(self, p: np.ndarray) -> np.ndarray:
        return self.variance * p

    def kinetic(self, p: np.ndarray) -> np.ndarray:
        return 0.5 * np.sum(self.variance * p * p, axis=-1)

    def draw_momentum(self, rng, shape):
        return self._momentum_sd * rng.standard_normal((*shape, self.dim))

    def inv_mass_matrix(self) -> np.ndarray:
        return np.diag(self.variance)


class DenseMetric(Metric):
    """``M^-1 = Sigma``, a full covariance, held as its Cholesky factor.

    Parameters
    ----------
    cov : (dim, dim) symmetric positive definite
        The estimated posterior covariance. Symmetrized on the way in (an
        estimate is symmetric only up to roundoff) and factorized once; a
        non-PSD input fails here, at construction, rather than as a NaN
        somewhere inside a trajectory.
    """

    def __init__(self, cov: np.ndarray) -> None:
        c = np.asarray(cov, dtype=float)
        if c.ndim != 2 or c.shape[0] != c.shape[1]:
            raise ValueError(f"expected a square covariance, got shape {c.shape}")
        c = 0.5 * (c + c.T)
        try:
            chol = np.linalg.cholesky(c)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "covariance is not positive definite; regularize the estimate "
                "before building a metric from it"
            ) from e
        self.dim = c.shape[0]
        self.cov = c
        self.chol = chol                      # Sigma = L L^T
        self._inv_chol = invert_lower(chol)   # L^-1, for the momentum draw

    def velocity(self, p: np.ndarray) -> np.ndarray:
        # Sigma is symmetric, so (Sigma p)_i for a batch of row vectors is p @ Sigma.
        return p @ self.cov

    def kinetic(self, p: np.ndarray) -> np.ndarray:
        # (L^T p)_j = sum_k p_k L_kj = (p @ L)_j
        return 0.5 * np.sum((p @ self.chol) ** 2, axis=-1)

    def draw_momentum(self, rng, shape):
        # p = L^-T z, i.e. z @ L^-1 for row vectors; Cov(p) = L^-T L^-1 = M.
        z = rng.standard_normal((*shape, self.dim))
        return z @ self._inv_chol

    def inv_mass_matrix(self) -> np.ndarray:
        return self.cov


def metric_from(
    inv_mass: np.ndarray | None, dim: int | None = None
) -> Metric:
    """Build the right metric from what a caller has: None / 1-D / 2-D.

    ``None`` -> identity (needs ``dim``), a vector -> diagonal, a matrix ->
    dense. This is the adapter that lets the existing ``inv_mass`` arguments
    keep working while the samplers move to holding a ``Metric``.
    """
    if inv_mass is None:
        if dim is None:
            raise ValueError("dim is required for the identity metric")
        return IdentityMetric(dim)
    arr = np.asarray(inv_mass, dtype=float)
    if arr.ndim == 1:
        return DiagonalMetric(arr)
    if arr.ndim == 2:
        return DenseMetric(arr)
    raise ValueError(f"inv_mass must be 1-D or 2-D, got {arr.ndim}-D")
