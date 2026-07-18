"""Hamiltonian Monte Carlo with a leapfrog integrator and dual-averaging
step-size adaptation.

Why HMC is a valid MCMC method (full derivation: theory/derivations.md, Sec. 4):

1.  Augment x with momentum p ~ N(0, I) and target the joint
    pi(x, p) proportional to exp(-H(x, p)),  H(x, p) = U(x) + K(p),
    with potential U(x) = -log pi(x) and kinetic K(p) = p.p / 2.
    The marginal of x under the joint is exactly pi: discard p and you have
    your samples.
2.  Exact Hamiltonian flow conserves H, is time-reversible, and preserves
    phase-space volume (Liouville's theorem). A proposal built from exact
    flow plus a momentum flip would therefore be a deterministic,
    volume-preserving involution with Delta H = 0: always accepted.
3.  We cannot integrate the flow exactly, so we discretize with *leapfrog*,
    chosen because it keeps two of the three properties exactly -- it is
    symplectic (hence exactly volume-preserving; each of its three substeps
    is a shear, unit Jacobian) and exactly reversible under momentum flip.
    Only conservation of H picks up discretization error, and a Metropolis
    accept with probability min(1, exp(-Delta H)) repairs precisely that
    error. Global energy error is O(step_size^2) (leapfrog is second order),
    so acceptance stays near 1 at usable step sizes.
4.  Alternating (a) exact Gibbs resampling of p and (b) the
    Metropolis-corrected trajectory step gives a kernel that leaves pi(x, p)
    invariant; distant proposals from long trajectories give far lower
    autocorrelation than a random walk.

Step size is tuned during warmup by dual averaging (Nesterov 2009 as adapted
in Hoffman & Gelman 2014, Algorithm 5), driving the mean acceptance
probability to ``target_accept``. A *diagonal mass matrix* can additionally be
adapted from windowed warmup variances (Sec. 4.8 of theory/derivations.md):
this rescales each coordinate so a single step size fits axis-aligned targets
of unequal scale, which is the common cheap win before reaching for NUTS.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .base import SamplerResult

DIVERGENCE_DELTA_H = 25.0  # exp(-25) ~ 1e-11: trajectory left the typical set


def leapfrog(
    grad_logpdf: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    p: np.ndarray,
    step_size: float,
    n_steps: int,
    inv_mass: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate Hamilton's equations with the leapfrog (Stormer-Verlet) scheme.

    With a mass matrix M the kinetic energy is K(p) = p^T M^{-1} p / 2, so the
    momentum enters the drift through M^{-1}:

    dx/dt = dK/dp = M^{-1} p,   dp/dt = -grad U(x) = +grad log pi(x)

        p_{1/2} = p_0 + (eps/2) grad log pi(x_0)        (half kick)
        x_{k+1} = x_k + eps M^{-1} p_{k+1/2}            (drift)
        p_{k+3/2} = p_{k+1/2} + eps grad log pi(x_{k+1})(kick)
        ... final half kick to resynchronize p with x.

    ``inv_mass`` is the *diagonal* of M^{-1}, shape (dim,); None means M = I and
    the drift is the plain x += eps p. Only the drift changes -- the kicks are
    gradient steps in position space and never see the metric. Rescaling p by a
    diagonal is still a shear composition, so symplecticity and reversibility
    (hence the accept step's validity) are untouched.

    Symmetric composition => second-order accurate and reversible:
    running n_steps from (x', -p') returns exactly to (x, -p) up to
    float roundoff (verified in tests/test_hmc.py).
    """
    # A trajectory that escapes the typical set can produce inf gradients and
    # then NaN positions. That IS the divergence signal: the NaN propagates to
    # a -inf acceptance ratio and the proposal is rejected (hmc() below), so
    # only the arithmetic warning is silenced here, not the failure.
    drift = step_size if inv_mass is None else step_size * inv_mass
    with np.errstate(over="ignore", invalid="ignore"):
        x = x.copy()
        p = p + 0.5 * step_size * grad_logpdf(x)
        for k in range(n_steps):
            x = x + drift * p
            if k < n_steps - 1:
                p = p + step_size * grad_logpdf(x)
        p = p + 0.5 * step_size * grad_logpdf(x)
    return x, p


def _mass_adaptation_schedule(n_warmup: int) -> tuple[int, int, list[int]]:
    """Stan-style windowing for diagonal-metric warmup (Stan Reference Manual,
    "HMC algorithm parameters" / automatic adaptation).

    Warmup splits into three phases:

    - an *initial buffer* (~15%) where the metric is left at identity and only
      the step size adapts, letting the chain first reach the typical set;
    - a sequence of expanding, *memoryless* windows (each ~2x the previous) --
      each estimates a fresh diagonal metric from just its own draws, so early
      pre-convergence samples are discarded rather than averaged in; and
    - a *terminal buffer* (~10%) where the final metric is frozen and the step
      size is re-tuned to it.

    Returns ``(init_buffer, term_buffer, window_ends)`` where ``window_ends`` are
    the (1-indexed) warmup iteration counts at which a window closes -- the
    metric is re-estimated and step-size dual averaging is restarted. The last
    entry is always ``n_warmup - term_buffer``. An empty list means warmup is
    too short to adapt a metric meaningfully; the identity metric is kept.
    """
    init_buffer = max(10, int(round(0.15 * n_warmup)))
    term_buffer = max(10, int(round(0.10 * n_warmup)))
    middle = n_warmup - init_buffer - term_buffer
    if middle < 20:
        return init_buffer, term_buffer, []
    ends: list[int] = []
    pos = init_buffer
    window = max(25, int(round(0.05 * n_warmup)))
    while pos + window < init_buffer + middle:
        pos += window
        ends.append(pos)
        window *= 2
    ends.append(init_buffer + middle)  # final window closes at n_warmup - term_buffer
    return init_buffer, term_buffer, ends


def hmc(
    target: Any,
    x0: np.ndarray,
    n_samples: int,
    step_size: float,
    n_leapfrog: int,
    rng: np.random.Generator,
    n_warmup: int = 0,
    adapt_step_size: bool = False,
    adapt_mass: bool = False,
    target_accept: float = 0.8,
    jitter: float = 0.2,
) -> SamplerResult:
    """Run batched HMC chains, optionally with an adapted diagonal mass matrix.

    Parameters
    ----------
    target : object with batched ``logpdf`` and ``grad_logpdf``.
    x0 : ndarray (n_chains, dim)
    step_size : float
        Leapfrog step size; treated as an initial value when
        ``adapt_step_size`` (frozen at the dual-averaged value afterwards).
    n_leapfrog : int
        Base number of leapfrog steps L. Each iteration uses a uniform draw
        from [ceil((1-jitter) L), L] (shared across chains): fixed-L HMC on
        near-Gaussian targets can resonate (trajectories that U-turn back to
        the start), and jitter breaks the periodicity cheaply. NUTS replaces
        this heuristic with an automatic U-turn criterion.
    adapt_mass : bool
        If True (and ``n_warmup`` is long enough), estimate a *diagonal* mass
        matrix M from windowed warmup variances: M^{-1} = diag of the per-
        coordinate variances (Sec. 4.8). With that metric the momentum is
        drawn p ~ N(0, M) and the kinetic energy is K = p^T M^{-1} p / 2, so
        each axis is preconditioned to unit scale and one step size fits an
        axis-aligned target of unequal scales. Best paired with
        ``adapt_step_size`` (the step size is re-tuned to each new metric).
        The metric is frozen at the end of warmup -- adapting during sampling
        would break invariance, exactly as for the step size.
    target_accept : float
        Dual-averaging target for the mean acceptance probability. 0.8 is a
        good default; the cost-optimal value for well-behaved targets is
        ~0.65 (Neal 2011, Sec. 5.4.4), while harder geometry rewards higher.

    Returns
    -------
    SamplerResult; ``extras`` holds per-iteration energy errors ``delta_H``
    (n_chains, n_samples), the final ``step_size``, the diagonal ``inv_mass``
    (= diag(M^{-1}), all ones when ``adapt_mass`` is off), ``n_divergent``, and
    ``n_grad_evals`` for compute-normalized efficiency comparisons.

    Examples
    --------
    Long leapfrog trajectories mix a correlated Gaussian efficiently --
    high acceptance and an accurately recovered mean:

    >>> import numpy as np
    >>> from mcmc.targets import Gaussian
    >>> target = Gaussian(mean=[1.0, -1.0], cov=[[1.0, 0.8], [0.8, 1.0]])
    >>> rng = np.random.default_rng(0)
    >>> res = hmc(target, np.zeros((4, 2)), n_samples=1000, step_size=0.3,
    ...           n_leapfrog=15, rng=rng, n_warmup=300)
    >>> res.samples.shape
    (4, 1000, 2)
    >>> bool(np.allclose(res.pooled().mean(axis=0), [1.0, -1.0], atol=0.15))
    True
    >>> bool(res.accept_rate.mean() > 0.6)
    True
    """
    x = np.array(x0, dtype=float, copy=True)
    n_chains, dim = x.shape
    lp = np.asarray(target.logpdf(x), dtype=float)

    samples = np.empty((n_chains, n_samples, dim))
    delta_H = np.empty((n_chains, n_samples))
    n_accept = np.zeros(n_chains)
    n_divergent = 0
    n_grad_evals = 0

    inv_mass = np.ones(dim)          # diag(M^{-1}); identity metric by default
    sqrt_mass = np.ones(dim)         # sd of the momentum draw, = sqrt(diag(M)) = 1/sqrt(inv_mass)

    # dual-averaging state (Hoffman & Gelman 2014, Alg. 5). Reset at each mass
    # window boundary so the step size re-tunes to the fresh metric.
    eps = float(step_size)
    gamma, t0, kappa = 0.05, 10.0, 0.75

    def reset_dual_averaging(cur_eps: float) -> tuple[float, float, float]:
        return np.log(10.0 * cur_eps), 0.0, 0.0  # mu, h_bar, log_eps_bar

    mu, h_bar, log_eps_bar = reset_dual_averaging(eps)
    da_t = 0.0  # steps since the last dual-averaging reset

    # windowed diagonal-mass adaptation (memoryless windows; see helper)
    init_buffer, term_buffer, window_ends = (
        _mass_adaptation_schedule(n_warmup) if adapt_mass else (0, 0, [])
    )
    window_end_set = set(window_ends)
    last_window_end = window_ends[-1] if window_ends else 0
    # Welford-free running moments over the current window (pooled across chains)
    acc_n = 0
    acc_sum = np.zeros(dim)
    acc_sumsq = np.zeros(dim)

    L_low = max(1, int(np.ceil((1.0 - jitter) * n_leapfrog)))

    for it in range(n_warmup + n_samples):
        p0 = rng.standard_normal((n_chains, dim)) * sqrt_mass  # p ~ N(0, M)
        L = int(rng.integers(L_low, n_leapfrog + 1))
        x_prop, p_prop = leapfrog(target.grad_logpdf, x, p0, eps, L, inv_mass)
        n_grad_evals += (L + 1) * n_chains
        lp_prop = np.asarray(target.logpdf(x_prop), dtype=float)

        # -Delta H = [log pi(x') - K(p')] - [log pi(x) - K(p)],
        # K(p) = p^T M^{-1} p / 2 = sum inv_mass * p^2 / 2.
        # Diverged trajectories carry inf/NaN through here by design; they
        # are mapped to -inf below and rejected.
        with np.errstate(over="ignore", invalid="ignore"):
            neg_dH = (lp_prop - 0.5 * np.sum(inv_mass * p_prop**2, axis=1)) - (
                lp - 0.5 * np.sum(inv_mass * p0**2, axis=1)
            )
        neg_dH = np.where(np.isnan(neg_dH), -np.inf, neg_dH)
        accept = np.log(rng.uniform(size=n_chains)) < neg_dH
        x[accept] = x_prop[accept]
        lp[accept] = lp_prop[accept]

        if it < n_warmup:
            if adapt_step_size:
                da_t += 1.0
                t = da_t
                alpha = float(np.mean(np.exp(np.minimum(0.0, neg_dH))))
                h_bar = (1 - 1 / (t + t0)) * h_bar + (target_accept - alpha) / (t + t0)
                log_eps = mu - np.sqrt(t) / gamma * h_bar
                w = t ** (-kappa)
                log_eps_bar = w * log_eps + (1 - w) * log_eps_bar
                eps = float(np.exp(log_eps))

            # accumulate window moments during the metric-adaptation region
            if adapt_mass and init_buffer <= it < last_window_end:
                acc_n += n_chains
                acc_sum += x.sum(axis=0)
                acc_sumsq += np.sum(x * x, axis=0)

            if adapt_mass and (it + 1) in window_end_set:
                mean = acc_sum / acc_n
                var = acc_sumsq / acc_n - mean * mean
                # Stan's regularization toward a unit metric when the window is
                # small, so a short/degenerate window can't produce a wild scale.
                var = (acc_n / (acc_n + 5.0)) * var + 1e-3 * (5.0 / (acc_n + 5.0))
                inv_mass = np.maximum(var, 1e-12)
                sqrt_mass = 1.0 / np.sqrt(inv_mass)
                acc_n, acc_sum, acc_sumsq = 0, np.zeros(dim), np.zeros(dim)
                if adapt_step_size:
                    mu, h_bar, log_eps_bar = reset_dual_averaging(eps)
                    da_t = 0.0

            if it == n_warmup - 1 and adapt_step_size:
                eps = float(np.exp(log_eps_bar))  # freeze at averaged value
        else:
            i = it - n_warmup
            samples[:, i, :] = x
            delta_H[:, i] = -neg_dH
            n_accept += accept
            n_divergent += int(np.sum(-neg_dH > DIVERGENCE_DELTA_H))

    return SamplerResult(
        samples=samples,
        accept_rate=n_accept / n_samples,
        extras={
            "delta_H": delta_H,
            "step_size": eps,
            "inv_mass": inv_mass,
            "n_divergent": n_divergent,
            "n_grad_evals": n_grad_evals,
        },
    )
