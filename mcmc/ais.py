"""Annealed importance sampling: the normalizing constant MCMC cannot give you.

Every sampler in this repo answers questions of the form E_pi[h(x)], and none
of them answers the one question the normalizer is: what is
Z = integral f(x) dx, for an unnormalized density f? Metropolis-Hastings sees
only ratios f(x')/f(x), and Z cancels out of every ratio. That is exactly why
MCMC works on unnormalized targets, and exactly why it cannot report Z --
which is the quantity a Bayes factor, a marginal likelihood, or a free energy
is made of.

Importance sampling can, in principle: with a proposal q that we can both draw
from and evaluate,

    Z = integral f(x) dx = E_q[ f(x) / q(x) ],

so the mean of the weights w = f/q is an unbiased estimate of Z. In practice a
single q in more than a few dimensions is hopeless: if q is anywhere narrower
than f the ratio has enormous, occasionally infinite variance, and the estimate
is dominated by whichever draw happened to land furthest into the tail. The
sample mean converges, and it converges too slowly to be usable.

Annealed importance sampling (Neal 2001) makes the jump in many small steps.
Interpolate geometrically from a tractable f_0 (samplable, with a known Z_0) to
the target f_T:

    f_j(x) = f_0(x)^{1 - beta_j} f_T(x)^{beta_j},   0 = beta_0 < ... < beta_T = 1.

Draw x_0 ~ p_0, and for j = 1..T move x through an MCMC transition T_j that
leaves p_j invariant, accumulating

    log w = sum_{j=1}^{T} [ log f_j(x_{j-1}) - log f_{j-1}(x_{j-1}) ]
          = sum_{j=1}^{T} (beta_j - beta_{j-1}) [ log f_T(x_{j-1}) - log f_0(x_{j-1}) ],

where the second line is what the geometric path gives and is what the code
computes: each increment is a small beta step times a log-density difference
evaluated *before* the move. Then E[w] = Z_T / Z_0, exactly, for any number of
temperatures, any transition kernels leaving the p_j invariant, and any number
of particles -- the unbiasedness is not asymptotic. Neal's derivation is an
importance-sampling argument in the *extended* space of whole trajectories
(x_0, ..., x_{T-1}), where the reverse-transition product supplies exactly the
missing normalizers; theory/derivations.md Sec. 7 has it.

What the annealing buys is variance. At T = 1 the path collapses to ordinary
importance sampling from p_0 (the code makes that the honest baseline rather
than a claim), and every intermediate temperature is a chance for the particles
to re-equilibrate before the weight increment gets any bigger.

Three things about this estimator that matter more than its definition, all
measured in experiments/ais.py:

1. **Unbiased in Z, biased in log Z, downward.** log E[w] >= E[log w] by
   Jensen, and log Z is what anyone actually reports. The bias is not a
   technicality -- with degenerate weights it is the whole error, and it
   always points the same way, so a too-short ladder *understates* the
   evidence rather than scattering around it.

2. **The diagnostic is the effective sample size of the weights**,
   ESS = (sum w)^2 / sum w^2, which is 1 when a single particle carries all
   the weight and N when they are equal. It costs nothing to compute and is
   the only warning available when there is no ground truth to check against.

3. **It inherits the sampler's failures.** If the transitions cannot cross
   between modes, AIS does not either; it just reports a confident, wrong Z.

Everything here works with the repo's target protocol (a batched ``logpdf``),
particles run as batched chains, and the transition is the repo's own
random-walk Metropolis, so the invariance the derivation requires is a property
of already-tested code rather than of a kernel written for this file.

References
----------
Neal (2001), Annealed importance sampling. Statistics and Computing 11:125-139.
Jarzynski (1997), Nonequilibrium equality for free energy differences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .metropolis import random_walk_metropolis


@dataclass
class AISResult:
    """Log weights from one AIS run, plus what they say about Z.

    ``log_weights`` is the raw per-particle log w; everything else is derived
    from it, so a caller who wants a different estimator (a bootstrap interval,
    a jackknife bias correction) has the primitive.
    """

    log_weights: np.ndarray
    log_z0: float
    accept_rates: np.ndarray
    betas: np.ndarray
    extras: "dict[str, Any]" = field(default_factory=dict)

    @property
    def n_particles(self) -> int:
        return int(self.log_weights.shape[0])

    @property
    def log_z(self) -> float:
        """log Z_T = log mean(w) + log Z_0, via log-sum-exp.

        The log of an unbiased estimate of Z, which is *not* an unbiased
        estimate of log Z: Jensen puts it below the truth, by an amount that
        grows as the weights spread out. `log_z_jackknife` estimates that gap
        rather than hoping it is small.
        """
        w = self.log_weights
        m = float(np.max(w))
        return float(m + np.log(np.mean(np.exp(w - m))) + self.log_z0)

    @property
    def ess(self) -> float:
        """Effective sample size of the weights, (sum w)^2 / sum w^2.

        Between 1 and N. Computed in log space because the weights routinely
        span hundreds of nats: ESS = exp(2 lse(w) - lse(2w)).
        """
        w = self.log_weights
        m = float(np.max(w))
        lse1 = m + np.log(np.sum(np.exp(w - m)))
        m2 = float(np.max(2.0 * w))
        lse2 = m2 + np.log(np.sum(np.exp(2.0 * w - m2)))
        return float(np.exp(2.0 * lse1 - lse2))

    @property
    def ess_fraction(self) -> float:
        return self.ess / self.n_particles

    def log_z_jackknife(self) -> "tuple[float, float]":
        """Jackknife-corrected log Z, and the correction that was applied.

        The bias of log-of-a-mean is O(1/N), so the leave-one-out jackknife
        removes its leading term:

            log Z_jack = N * log Z_hat - (N - 1) * mean_i log Z_hat^{(-i)}.

        Reported alongside the correction itself, because a large correction is
        a statement that the naive number was not to be trusted -- and on
        degenerate weights the jackknife is not to be trusted either.
        """
        w = self.log_weights
        n = self.n_particles
        if n < 2:
            raise ValueError("the jackknife needs at least two particles")
        m = float(np.max(w))
        shifted = np.exp(w - m)
        total = float(np.sum(shifted))
        loo = np.array([
            m + np.log((total - shifted[i]) / (n - 1)) for i in range(n)
        ])
        full = self.log_z - self.log_z0
        corrected = n * full - (n - 1) * float(np.mean(loo))
        return corrected + self.log_z0, corrected - full


class _GeometricBridge:
    """The intermediate target f_0^{1-beta} f_T^beta, as a repo target.

    A class rather than a closure so that the object handed to the sampler has
    exactly the interface `mcmc/base.py` documents, and so that ``beta`` can be
    moved without rebuilding it.
    """

    def __init__(self, log_f0: Any, log_fT: Any, beta: float):
        self.log_f0 = log_f0
        self.log_fT = log_fT
        self.beta = float(beta)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return ((1.0 - self.beta) * np.asarray(self.log_f0(x), dtype=float)
                + self.beta * np.asarray(self.log_fT(x), dtype=float))


def geometric_betas(n_temps: int, power: float = 1.0) -> np.ndarray:
    """Ladder 0 = beta_0 < ... < beta_T = 1, with ``T = n_temps`` steps.

    ``power = 1`` is the uniform spacing beta_j = j / T. Larger powers bunch
    the temperatures near beta = 0, which is where the density is changing
    fastest for a broad p_0 and a concentrated target -- the ladder shape is a
    real tuning knob and experiments/ais.py measures what it is worth rather
    than asserting a default.

    Examples
    --------
    >>> import numpy as np
    >>> geometric_betas(4)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    >>> bool(np.all(np.diff(geometric_betas(8, power=2.0)) > 0))
    True
    """
    if n_temps < 1:
        raise ValueError("need at least one temperature step")
    return np.linspace(0.0, 1.0, n_temps + 1) ** power


def annealed_importance_sampling(
    log_f0: Any,
    sample_f0: Any,
    log_fT: Any,
    betas: np.ndarray,
    n_particles: int,
    rng: np.random.Generator,
    log_z0: float = 0.0,
    step_size: float = 1.0,
    n_transition: int = 1,
) -> AISResult:
    """Estimate log Z_T for an unnormalized ``log_fT``, by annealing from p_0.

    Parameters
    ----------
    log_f0, log_fT : callables ``(n, dim) -> (n,)``
        Unnormalized log-densities of the initial and target distributions.
    sample_f0 : callable ``(n, rng) -> (n, dim)``
        Exact draws from the *normalized* p_0. AIS needs independent draws
        here; this is the one distribution that has to be tractable.
    betas : ndarray
        Increasing ladder with ``betas[0] == 0`` and ``betas[-1] == 1``.
    log_z0 : float
        log Z_0 of the initial density as written by ``log_f0``. Zero when
        ``log_f0`` is already normalized, which is the usual case.
    step_size : float
        Random-walk proposal scale for the transitions.
    n_transition : int
        Metropolis steps per temperature. More steps mean better mixing at each
        rung and a cost that scales with the product ``T * n_transition``,
        which is the budget experiments/ais.py holds fixed when it compares
        ladder lengths.

    Returns
    -------
    AISResult

    Notes
    -----
    The weight increment at rung j is evaluated at the state *before* the j-th
    transition, and the transition at rung j targets ``p_j``. Getting either
    off by one silently breaks the unbiasedness while leaving the estimate
    plausible, which is why ``tests/test_ais.py`` checks the estimator against
    a Gaussian whose Z is known in closed form rather than against a plot.

    Examples
    --------
    A 2D Gaussian target scaled by a known constant, annealed from a broad
    Gaussian. The estimate has to find the constant:

    >>> import numpy as np
    >>> from mcmc.targets import Gaussian
    >>> broad = Gaussian(mean=[0.0, 0.0], cov=[[9.0, 0.0], [0.0, 9.0]])
    >>> target = Gaussian(mean=[1.0, -1.0], cov=[[1.0, 0.5], [0.5, 1.0]])
    >>> res = annealed_importance_sampling(
    ...     log_f0=broad.logpdf, sample_f0=broad.sample,
    ...     log_fT=lambda x: target.logpdf(x) + 3.0,      # log Z_T = 3 exactly
    ...     betas=geometric_betas(50), n_particles=400,
    ...     rng=np.random.default_rng(0), step_size=1.0)
    >>> bool(abs(res.log_z - 3.0) < 0.1)
    True
    >>> round(res.ess_fraction, 2)     # 50 rungs is not many: experiments/ais.py
    0.41
    """
    betas = np.asarray(betas, dtype=float)
    if betas.ndim != 1 or betas.size < 2:
        raise ValueError("betas must be a ladder of at least two entries")
    if not (betas[0] == 0.0 and betas[-1] == 1.0):
        raise ValueError(
            f"betas must run from 0 to 1, got [{betas[0]}, {betas[-1]}]"
        )
    if np.any(np.diff(betas) <= 0):
        raise ValueError("betas must be strictly increasing")

    x = np.atleast_2d(np.asarray(sample_f0(n_particles, rng), dtype=float))
    if x.shape[0] != n_particles:
        raise ValueError(
            f"sample_f0 returned {x.shape[0]} draws, expected {n_particles}"
        )

    log_w = np.zeros(n_particles)
    accept = np.zeros(n_particles)
    bridge = _GeometricBridge(log_f0, log_fT, beta=0.0)

    for j in range(1, betas.size):
        # The weight increment uses the state before the move, with the ratio
        # f_j/f_{j-1} collapsing to a beta step times one log-density gap.
        gap = (np.asarray(log_fT(x), dtype=float)
               - np.asarray(log_f0(x), dtype=float))
        log_w += (betas[j] - betas[j - 1]) * gap

        # Then move under p_j. The last rung's transition targets the actual
        # target and does not affect the weights; it is kept so the returned
        # particles are p_T-distributed and the acceptance rate at beta = 1 is
        # reported like every other rung.
        bridge.beta = float(betas[j])
        res = random_walk_metropolis(
            bridge, x, n_samples=n_transition, step_size=step_size, rng=rng
        )
        x = res.samples[:, -1, :]
        accept += res.accept_rate

    return AISResult(
        log_weights=log_w,
        log_z0=float(log_z0),
        accept_rates=accept / (betas.size - 1),
        betas=betas,
        extras={"particles": x, "step_size": step_size,
                "n_transition": n_transition},
    )
