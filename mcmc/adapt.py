"""Estimating the metric from warmup draws: the covariance, and its shrinkage.

``metric.py`` gives the three operations a mass matrix has to support; this
module produces the matrix they are built from. Sec. 4.8 already estimates a
*diagonal* metric inline in ``hmc.py`` -- per-coordinate variances over a
memoryless warmup window, shrunk toward a unit metric -- and the dense case is
the same recipe with one part that does not survive the generalization.

**Why the diagonal recipe does not just carry over.** A window of ``n`` pooled
draws in ``d`` dimensions estimates ``d`` variances from ``n`` numbers each;
the same window estimates ``d(d+1)/2`` covariance entries from the same ``n``
draws. The sample covariance ``S`` of ``n`` centered draws has rank at most
``n - 1``, so for ``n <= d`` it is *singular* -- and what that does is worse
than an error. Whether ``DenseMetric`` catches it comes down to the sign of a
roundoff error on a zero eigenvalue: at ``n = d`` in the case
``tests/test_adapt.py`` pins down, the smallest eigenvalue lands at ``+2e-16``,
Cholesky succeeds, and the run gets a metric conditioned at ``1e16``; one draw
fewer and the same eigenvalue is negative and it raises. The PSD check is a
backstop, not a defense. Even for ``n`` a few times ``d`` the estimate is badly
conditioned in the way
that matters here: the extreme eigenvalues of ``S`` are biased outward
(Marchenko-Pastur: for ``n/d = 4`` the sample spectrum of a *white* population
already spans ``[0.25, 2.25]``), and a metric is used through
``Sigma_hat @ Sigma^-1``, so an eigenvalue wrong by 9x is a step-size
constraint wrong by 3x. Regularization is not a numerical nicety here; it is
what makes the estimate usable at all.

Two regularizers, both implemented, both measured in
``experiments/dense_metric_estimation.py``:

``"stan"``
    The literal generalization of the line ``hmc.py`` already runs for the
    diagonal, ``S <- (n/(n+5)) S + 1e-3 (5/(n+5)) I``. The weight on the ridge
    is ``5/(n+5)``, which is ``0.005`` at ``n = 1000``: for any window long
    enough to be worth using this is a *singularity guard*, not shrinkage. It
    keeps the estimate invertible and leaves its conditioning essentially
    alone. Stan uses exactly this, and it is the honest default here for the
    same reason -- it does not silently change a well-sampled window.

``"ledoit-wolf"``
    Ledoit & Wolf (2004): shrink toward the *scaled* identity ``m I``,
    ``m = tr(S)/d``, with the shrinkage intensity that minimizes expected
    squared Frobenius error, estimated from the same draws (derivation:
    Sec. 4.11). It is well conditioned by construction at any ``n``, including
    ``n < d``.

Which one a *sampler* should want is not settled by "which is closer in
Frobenius norm", and the measurement says so: see Sec. 4.11 and the experiment.
The loss that governs HMC is the conditioning of the whitened Hessian, and for
a Gaussian target that is ``kappa(Sigma_hat Sigma^-1)`` -- a ratio-of-
eigenvalues loss, which a shrinkage tuned for a sum-of-squares loss is under no
obligation to improve.

The accumulator below is window-scoped and memoryless (Sec. 4.8): each warmup
window estimates from its own draws only, so pre-convergence samples are
dropped rather than averaged in.
"""

from __future__ import annotations

import numpy as np

STAN_RIDGE_WEIGHT = 5.0    # the "+5" in n/(n+5)
STAN_RIDGE_VALUE = 1e-3    # what a zero-information window shrinks to


class WindowMoments:
    """Pooled moments of one warmup window, in diagonal or dense mode.

    ``add(x)`` takes the current position of every chain, ``(n_chains, dim)``,
    once per warmup iteration; ``reset()`` starts the next window.

    The two modes differ in more than the shape of the answer:

    - ``"diag"`` streams ``sum`` and ``sum of squares`` and returns
      ``sumsq/n - mean^2``. That expression, in that association, is what
      ``hmc.py`` computed inline before this class existed, and
      ``tests/test_adapt.py`` asserts the two are equal *exactly*, not close:
      a one-ulp change to an adapted variance moves a leapfrog trajectory,
      moves an accept comparison, and would move every diagonal-metric number
      already committed to this repo.
    - ``"dense"`` keeps the window's draws. It could stream ``sum`` and
      ``sum of outer products`` instead, but Ledoit-Wolf's shrinkage intensity
      needs a *centered* fourth-moment term (Sec. 4.11) whose streaming form
      would require the window mean before the window ends. Holding the draws
      is ``O(n d)`` floats -- a 1000-iteration window of 4 chains in 10
      dimensions is 320 KB -- against the ``O(d^2)`` the metric itself costs,
      so the memory is not the binding cost at any size this repo runs.
    """

    def __init__(self, dim: int, mode: str = "diag") -> None:
        if mode not in ("diag", "dense"):
            raise ValueError(f"mode must be 'diag' or 'dense', got {mode!r}")
        self.dim = int(dim)
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self._sum = np.zeros(self.dim)
        self._sumsq = np.zeros(self.dim)
        self._draws: list[np.ndarray] = []

    def add(self, x: np.ndarray) -> None:
        """Absorb one iteration's positions, ``(n_chains, dim)``."""
        x = np.asarray(x, dtype=float)
        self.n += x.shape[0]
        if self.mode == "diag":
            self._sum += x.sum(axis=0)
            self._sumsq += np.sum(x * x, axis=0)
        else:
            self._draws.append(x.copy())

    def variance(self) -> np.ndarray:
        """The window's per-coordinate variance, ``(dim,)``.

        Uncentered form, ``E[x^2] - E[x]^2``, dividing by ``n`` and not
        ``n - 1``: both choices are deliberate and both are what ``hmc.py``
        did inline. The 1/n normalization is the maximum-likelihood variance,
        and the shrinkage below dominates the 1/(n-1) difference by orders of
        magnitude at every window size the schedule produces.
        """
        if self.mode != "diag":
            raise ValueError("variance() needs mode='diag'")
        mean = self._sum / self.n
        return self._sumsq / self.n - mean * mean

    def draws(self) -> np.ndarray:
        """The window's pooled draws, ``(n, dim)``."""
        if self.mode != "dense":
            raise ValueError("draws() needs mode='dense'")
        return np.concatenate(self._draws, axis=0)

    def covariance(self) -> np.ndarray:
        """The window's sample covariance, ``(dim, dim)``, normalized by ``n``.

        Rank ``min(n - 1, dim)``: for ``n <= dim`` this matrix is singular and
        is not a usable metric on its own. That is the whole reason the
        shrinkage below exists, so it is returned unregularized and the caller
        regularizes explicitly.
        """
        x = self.draws()
        xc = x - x.mean(axis=0)
        return (xc.T @ xc) / x.shape[0]


def stan_shrink(estimate: np.ndarray, n: int) -> np.ndarray:
    """Stan's warmup regularization, for a variance vector or a covariance.

    ``est <- (n/(n+5)) est + 1e-3 (5/(n+5)) I``.

    Interpolates between the sample estimate at large ``n`` and a fixed tiny
    unit metric at ``n = 0``, so a degenerate window cannot emit a wild scale
    and a singular one cannot emit a singular metric. It is not much of a
    shrinkage: the ridge weight is ``5/(n+5)``, under a percent for any window
    the schedule in ``hmc.py`` actually closes.

    The 1-D branch is written in exactly the association ``hmc.py`` used
    inline, so an adapted diagonal run is bit-for-bit what it was.
    """
    est = np.asarray(estimate, dtype=float)
    w = n / (n + STAN_RIDGE_WEIGHT)
    ridge = STAN_RIDGE_VALUE * (STAN_RIDGE_WEIGHT / (n + STAN_RIDGE_WEIGHT))
    if est.ndim == 1:
        return w * est + ridge
    if est.ndim == 2:
        return w * est + ridge * np.eye(est.shape[0])
    raise ValueError(f"expected a 1-D or 2-D estimate, got {est.ndim}-D")


def ledoit_wolf(draws: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage of a sample covariance toward the scaled identity.

    Returns ``(sigma_hat, intensity)`` with
    ``sigma_hat = intensity * m I + (1 - intensity) * S``, ``m = tr(S)/d``,
    and ``intensity`` in ``[0, 1]`` the estimated optimal shrinkage.

    Derivation in Sec. 4.11; the four estimated quantities, in the
    dimension-normalized inner product ``<A, B> = tr(A B^T)/d`` the paper uses:

        m   = <S, I>                      the average sample eigenvalue
        d2  = ||S - m I||^2               how far S is from spherical
        b2  = min(bbar2, d2)              the estimation error in S, capped
        a2  = d2 - b2                     how far Sigma is from spherical

    with ``bbar2 = (1/n^2) sum_k ||x_k x_k^T - S||^2`` and the shrinkage
    ``intensity = b2/d2``. The cap is the paper's: ``bbar2 > d2`` is a
    small-sample artifact, and without it the intensity could exceed 1 and the
    estimate could leave the PSD cone.

    ``bbar2`` is computed from the identity

        sum_k ||x_k x_k^T - S||_F^2 = sum_k ||x_k||^4 - n ||S||_F^2

    (expand, and use ``sum_k x_k x_k^T = n S`` with ``tr(S S) = ||S||_F^2``),
    which costs one pass and no ``(d, d)`` temporaries per draw.

    ``draws`` are centered here, so the ``n`` in the formulas is the same ``n``
    the covariance was normalized by, and the estimate is exactly spherical --
    ``intensity = 1`` -- when ``S`` already is.
    """
    x = np.asarray(draws, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected (n, dim) draws, got shape {x.shape}")
    n, d = x.shape
    if n < 2:
        raise ValueError("Ledoit-Wolf needs at least 2 draws")
    xc = x - x.mean(axis=0)
    S = (xc.T @ xc) / n

    m = np.trace(S) / d
    # dimension-normalized squared Frobenius norms; the /d cancels in b2/d2 but
    # is kept so each quantity is the paper's and can be checked against it.
    d2 = np.sum((S - m * np.eye(d)) ** 2) / d
    quartic = np.sum(np.sum(xc * xc, axis=1) ** 2)          # sum_k ||x_k||^4
    bbar2 = (quartic - n * np.sum(S**2)) / (n**2 * d)
    b2 = min(bbar2, d2)
    if d2 <= 0.0:                       # S is already exactly spherical
        return S.copy(), 1.0
    intensity = float(np.clip(b2 / d2, 0.0, 1.0))
    sigma_hat = intensity * m * np.eye(d) + (1.0 - intensity) * S
    return sigma_hat, intensity


def estimate_covariance(
    moments: WindowMoments, shrinkage: str = "stan", floor: float = 1e-12
) -> np.ndarray:
    """The window's covariance, regularized, ready for ``DenseMetric``.

    ``shrinkage="stan"`` applies the ridge above to the sample covariance;
    ``"ledoit-wolf"`` applies the data-driven intensity and then the same tiny
    ridge, because Ledoit-Wolf's intensity can be estimated at 0 (when ``S``
    looks like a perfect estimate of a very non-spherical population) and a
    ridge-free ``S`` at ``n <= d`` is singular.

    ``floor`` is applied to the eigenvalues as a last resort: it is not
    expected to bind after either shrinkage, and ``tests/test_adapt.py`` checks
    that it does not on the cases the sampler actually produces. It exists so
    that a pathological window fails as a slightly-wrong metric rather than as
    a Cholesky exception in the middle of a run.
    """
    if shrinkage == "stan":
        cov = stan_shrink(moments.covariance(), moments.n)
    elif shrinkage == "ledoit-wolf":
        shrunk, _ = ledoit_wolf(moments.draws())
        cov = stan_shrink(shrunk, moments.n)
    else:
        raise ValueError(
            f"shrinkage must be 'stan' or 'ledoit-wolf', got {shrinkage!r}"
        )
    cov = 0.5 * (cov + cov.T)
    w, V = np.linalg.eigh(cov)
    if w.min() < floor:
        cov = V @ np.diag(np.maximum(w, floor)) @ V.T
        cov = 0.5 * (cov + cov.T)
    return cov


def whitened_condition_number(inv_mass: np.ndarray, cov: np.ndarray) -> float:
    """``kappa`` of the whitened Hessian for a Gaussian target of covariance ``cov``.

    The loss a metric is actually judged by (Sec. 4.10): with ``M^-1`` the
    metric and ``A = cov^-1`` the Gaussian's Hessian, HMC behaves as
    unit-metric HMC on a target whose Hessian is ``M^-1/2 A M^-1/2``, and it is
    *that* symmetric matrix's eigenvalue ratio that sets the step size.

    Computed as the eigenvalues of the symmetric ``L^T A L`` where
    ``M^-1 = L L^T``, never as ``cond(M^-1 @ A)``: the unsymmetric product has
    the right eigenvalues and the wrong singular values, and ``np.linalg.cond``
    reports singular values (Sec. 4.10 says how badly that goes -- six orders
    of magnitude at a scale ratio of 3000).

    ``inv_mass`` may be a (d,) diagonal or a (d, d) matrix.
    """
    inv_mass = np.asarray(inv_mass, dtype=float)
    if inv_mass.ndim == 1:
        inv_mass = np.diag(inv_mass)
    A = np.linalg.inv(np.asarray(cov, dtype=float))
    L = np.linalg.cholesky(0.5 * (inv_mass + inv_mass.T))
    w = np.linalg.eigvalsh(L.T @ A @ L)
    return float(w.max() / w.min())
