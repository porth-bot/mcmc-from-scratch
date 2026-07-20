"""The No-U-Turn Sampler (NUTS): HMC that chooses its own trajectory length.

Fixed-length HMC (``mcmc.hmc``) has one awkward tuning knob left after the step
size is dual-averaged: the number of leapfrog steps ``L``. Too few and the
proposal barely moves; too many and the trajectory U-turns back toward the
start, wasting gradients (and can *resonate* on near-Gaussian targets, which is
why ``hmc`` jitters ``L``). NUTS removes the knob entirely by growing the
trajectory until it starts to double back on itself.

This module implements the multinomial NUTS of Betancourt (2017), which
replaces the slice sampler of the original Hoffman & Gelman (2014) NUTS with a
cleaner canonical-distribution formulation:

1.  **Recursive doubling.** From the current point, repeatedly double the
    trajectory. Each doubling picks a random time direction (forward or
    backward in fictitious time) and appends a sub-trajectory of the same
    length as everything built so far, so after ``j`` doublings the trajectory
    holds up to ``2**j`` states, built with ``2**j - 1`` leapfrog steps. Because
    the two endpoints of every balanced sub-tree are exact leapfrog images of
    each other, the whole scheme stays reversible -- expansion never depends on
    where inside the tree the current state sits.

2.  **The no-U-turn criterion.** A balanced sub-tree with endpoints
    ``(x_minus, p_minus)`` and ``(x_plus, p_plus)`` is *turning* when advancing
    either endpoint would no longer increase the distance between them:

        (x_plus - x_minus) . (M^{-1} p_minus) < 0   or
        (x_plus - x_minus) . (M^{-1} p_plus)  < 0.

    (``M^{-1} p`` is the velocity; with the identity metric this is just ``p``.)
    The moment *any* sub-tree turns, doubling stops -- so the trajectory length
    adapts to the local geometry, long in flat directions and short in tight
    ones, with no user input.

3.  **Multinomial state selection.** Every state ``z = (x, p)`` visited carries
    canonical weight ``exp(-H(z)) = exp(log pi(x) - K(p))``. The next sample is
    drawn from the trajectory with probability proportional to that weight
    (Betancourt 2017, Sec. A.3). We realise the exact multinomial *progressively*
    while the tree is built: within a balanced doubling the newer half is chosen
    with probability equal to its share of the weight; at the top level the new
    half is chosen with the *biased* probability ``min(1, W_new / W_old)``
    (Stan's scheme), which pushes the sample outward along the trajectory for
    faster mixing while staying a valid transition. Since the whole trajectory
    is drawn from ``exp(-H)`` and ``H``'s marginal in ``x`` is ``pi``, detailed
    balance holds and the target is left invariant -- exactly as for HMC, but
    with the accept/reject step subsumed into the weighting.

Two safety valves make the recursion terminate on pathological geometry
(fully exercised on the funnel benchmark in Day 18): a **maximum tree depth**
caps work per iteration even where the criterion never fires, and a
**divergence** check marks any leaf whose energy error exceeds
``DIVERGENCE_DELTA_H`` as invalid (weight zero) and halts expansion -- the
signal that the step size is too large for the local curvature.

Step size is tuned during warmup by the same dual averaging used for HMC
(Hoffman & Gelman 2014, Alg. 5), driving the average per-leaf acceptance
statistic to ``target_accept``. Full derivation: theory/derivations.md, Sec. 4.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .base import SamplerResult
from .hmc import DIVERGENCE_DELTA_H


def _leapfrog_step(
    grad_logpdf: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    p: np.ndarray,
    grad_x: np.ndarray,
    step_size: float,
    inv_mass: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One leapfrog step that *caches* the gradient at both endpoints.

    Identical arithmetic to ``mcmc.hmc.leapfrog(..., n_steps=1)`` (verified in
    tests/test_nuts.py), but it takes the already-known gradient at ``x`` and
    returns the gradient at the new point, so the tree can reuse it as the next
    leaf's starting gradient. That makes NUTS cost exactly *one* gradient
    evaluation per leaf instead of two -- the honest count for the
    ESS-per-gradient comparison in Day 18.

        p_half = p + (eps/2) grad log pi(x)
        x'     = x + eps M^{-1} p_half
        p'     = p_half + (eps/2) grad log pi(x')

    ``step_size`` carries the time direction (negative to integrate backward).
    """
    drift = step_size if inv_mass is None else step_size * inv_mass
    with np.errstate(over="ignore", invalid="ignore"):
        p_half = p + 0.5 * step_size * grad_x
        x_new = x + drift * p_half
        grad_new = grad_logpdf(x_new[None, :])[0]
        p_new = p_half + 0.5 * step_size * grad_new
    return x_new, p_new, grad_new


def _is_turning(
    x_minus: np.ndarray,
    x_plus: np.ndarray,
    p_minus: np.ndarray,
    p_plus: np.ndarray,
    inv_mass: np.ndarray | None,
) -> bool:
    """Generalized no-U-turn criterion (Betancourt 2017, Eq. 4.2).

    True once the sub-tree spanning ``x_minus..x_plus`` starts to double back:
    the span vector no longer projects positively onto the velocity at either
    end. ``M^{-1} p`` is the velocity; identity metric -> just ``p``.
    """
    dx = x_plus - x_minus
    v_minus = p_minus if inv_mass is None else inv_mass * p_minus
    v_plus = p_plus if inv_mass is None else inv_mass * p_plus
    return bool(dx @ v_minus < 0.0 or dx @ v_plus < 0.0)


class _Tree:
    """A balanced sub-tree returned by the recursion.

    Holds only what the doubling scheme needs: the two extreme leaves
    (position, momentum, and cached gradient at each), the multinomially
    selected proposal and its joint log-density, the log total canonical weight
    ``log sum exp(-H)`` over the sub-tree (relative to the initial energy), the
    stop flag (a sub-U-turn or a divergence anywhere inside), and the running
    acceptance statistic for step-size adaptation.
    """

    __slots__ = (
        "x_minus", "p_minus", "grad_minus",
        "x_plus", "p_plus", "grad_plus",
        "x_sample", "joint_sample", "log_w",
        "stop", "alpha", "n_alpha",
    )

    def __init__(
        self,
        x_minus: np.ndarray, p_minus: np.ndarray, grad_minus: np.ndarray,
        x_plus: np.ndarray, p_plus: np.ndarray, grad_plus: np.ndarray,
        x_sample: np.ndarray, joint_sample: float, log_w: float,
        stop: bool, alpha: float, n_alpha: int,
    ) -> None:
        self.x_minus, self.p_minus, self.grad_minus = x_minus, p_minus, grad_minus
        self.x_plus, self.p_plus, self.grad_plus = x_plus, p_plus, grad_plus
        self.x_sample, self.joint_sample, self.log_w = x_sample, joint_sample, log_w
        self.stop, self.alpha, self.n_alpha = stop, alpha, n_alpha


def _nuts_transition(
    target: Any,
    x: np.ndarray,
    grad_x: np.ndarray,
    logp_x: float,
    step_size: float,
    max_tree_depth: int,
    rng: np.random.Generator,
    inv_mass: np.ndarray | None,
    sqrt_mass: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, int, int, int]:
    """One NUTS transition from ``x`` (with its cached gradient/log-density).

    Returns ``(x_new, grad_new, logp_new, accept_stat, delta_H, depth,
    n_grad, n_div)`` where ``accept_stat`` is the mean per-leaf acceptance
    probability (dual-averaging target) and ``delta_H`` is the energy error of
    the selected state (for comparison with fixed-L HMC).
    """
    dim = x.shape[0]
    p0: np.ndarray = np.asarray(rng.standard_normal(dim))
    if sqrt_mass is not None:
        p0 = p0 * sqrt_mass
    kinetic0 = 0.5 * float(np.sum((p0 * p0) if inv_mass is None else inv_mass * p0 * p0))
    joint0 = logp_x - kinetic0

    counters = {"n_grad": 0, "n_div": 0}

    def build_tree(x, p, grad, direction: int, depth: int) -> _Tree:
        if depth == 0:
            eps = direction * step_size
            x1, p1, grad1 = _leapfrog_step(
                target.grad_logpdf, x, p, grad, eps, inv_mass
            )
            counters["n_grad"] += 1
            logp1 = float(np.asarray(target.logpdf(x1[None, :]))[0])
            kinetic1 = 0.5 * np.sum(
                (p1 * p1) if inv_mass is None else inv_mass * p1 * p1
            )
            joint1 = float(logp1 - kinetic1)
            with np.errstate(over="ignore", invalid="ignore"):
                energy_err = joint0 - joint1  # = H(z) - H(z0)
            diverging = (not np.isfinite(energy_err)) or energy_err > DIVERGENCE_DELTA_H
            if diverging:
                counters["n_div"] += 1
                log_w = -np.inf
            else:
                log_w = float(joint1 - joint0)
            alpha = float(np.exp(min(0.0, -energy_err))) if np.isfinite(energy_err) else 0.0
            return _Tree(
                x1, p1, grad1, x1, p1, grad1,
                x1, joint1, log_w, diverging, alpha, 1,
            )

        left = build_tree(x, p, grad, direction, depth - 1)
        if left.stop:
            return left

        if direction == -1:
            right = build_tree(
                left.x_minus, left.p_minus, left.grad_minus, direction, depth - 1
            )
            x_minus, p_minus, grad_minus = right.x_minus, right.p_minus, right.grad_minus
            x_plus, p_plus, grad_plus = left.x_plus, left.p_plus, left.grad_plus
        else:
            right = build_tree(
                left.x_plus, left.p_plus, left.grad_plus, direction, depth - 1
            )
            x_minus, p_minus, grad_minus = left.x_minus, left.p_minus, left.grad_minus
            x_plus, p_plus, grad_plus = right.x_plus, right.p_plus, right.grad_plus

        # progressive multinomial selection within this balanced doubling:
        # take the right (newer) half's proposal with probability equal to its
        # share of the total canonical weight.
        log_w = float(np.logaddexp(left.log_w, right.log_w))
        if np.isfinite(right.log_w):
            p_right = float(np.exp(right.log_w - log_w)) if np.isfinite(log_w) else 1.0
            if rng.uniform() < p_right:
                x_sample, joint_sample = right.x_sample, right.joint_sample
            else:
                x_sample, joint_sample = left.x_sample, left.joint_sample
        else:
            x_sample, joint_sample = left.x_sample, left.joint_sample

        stop = (
            right.stop
            or _is_turning(x_minus, x_plus, p_minus, p_plus, inv_mass)
        )
        return _Tree(
            x_minus, p_minus, grad_minus, x_plus, p_plus, grad_plus,
            x_sample, joint_sample, log_w, stop, left.alpha + right.alpha,
            left.n_alpha + right.n_alpha,
        )

    # Whole-trajectory state, initialised to the single current point (weight 1).
    x_minus = x_plus = x
    p_minus = p_plus = p0
    grad_minus = grad_plus = grad_x
    x_sample = x
    joint_sample = joint0
    log_w_tree = 0.0  # log of exp(joint0 - joint0)
    sum_alpha = 0.0
    n_alpha = 0

    for depth in range(max_tree_depth):
        direction = 1 if rng.uniform() < 0.5 else -1
        if direction == -1:
            subtree = build_tree(x_minus, p_minus, grad_minus, direction, depth)
            x_minus, p_minus, grad_minus = (
                subtree.x_minus, subtree.p_minus, subtree.grad_minus
            )
        else:
            subtree = build_tree(x_plus, p_plus, grad_plus, direction, depth)
            x_plus, p_plus, grad_plus = (
                subtree.x_plus, subtree.p_plus, subtree.grad_plus
            )

        sum_alpha += subtree.alpha
        n_alpha += subtree.n_alpha

        if not subtree.stop:
            # biased progressive selection (Stan): jump to the new half's
            # proposal with probability min(1, W_new / W_old).
            if np.isfinite(subtree.log_w):
                if np.log(rng.uniform()) < subtree.log_w - log_w_tree:
                    x_sample, joint_sample = subtree.x_sample, subtree.joint_sample
            log_w_tree = float(np.logaddexp(log_w_tree, subtree.log_w))

        if subtree.stop or _is_turning(x_minus, x_plus, p_minus, p_plus, inv_mass):
            break

    logp_new = float(np.asarray(target.logpdf(x_sample[None, :]))[0])
    grad_new = np.asarray(target.grad_logpdf(x_sample[None, :]))[0]
    accept_stat = sum_alpha / n_alpha if n_alpha > 0 else 0.0
    delta_H = float(joint0 - joint_sample)  # H(selected) - H(initial)
    return (
        x_sample, grad_new, logp_new, accept_stat, delta_H,
        depth, counters["n_grad"], counters["n_div"],
    )


def nuts(
    target: Any,
    x0: np.ndarray,
    n_samples: int,
    step_size: float,
    rng: np.random.Generator,
    n_warmup: int = 0,
    adapt_step_size: bool = False,
    max_tree_depth: int = 10,
    target_accept: float = 0.8,
    inv_mass: np.ndarray | None = None,
) -> SamplerResult:
    """Run NUTS (multinomial, Betancourt 2017) with dual-averaging step size.

    Parameters
    ----------
    target : object with batched ``logpdf`` and ``grad_logpdf`` (the protocol in
        ``mcmc.base``); called one point at a time here (shape ``(1, dim)``),
        since each chain's trajectory has a data-dependent length.
    x0 : ndarray (n_chains, dim)
    step_size : float
        Leapfrog step size; an initial value when ``adapt_step_size`` is set
        (dual-averaged over warmup, then frozen).
    max_tree_depth : int
        Hard cap on doublings per iteration: at most ``2**max_tree_depth``
        leapfrog states are visited even if the no-U-turn criterion never
        fires. 10 (=> up to 1023 steps) is the Stan default.
    target_accept : float
        Dual-averaging target for the mean per-leaf acceptance statistic. NUTS
        is usually run higher than HMC (0.8; harder geometry rewards 0.9+),
        because a larger step size both lowers acceptance and inflates the tree.
    inv_mass : ndarray (dim,), optional
        Diagonal of ``M^{-1}`` (the metric); ``None`` is the identity metric.
        Enters the drift and the kinetic energy exactly as in ``mcmc.hmc``.

    Returns
    -------
    SamplerResult; ``extras`` holds per-iteration energy error of the selected
    state ``delta_H`` (n_chains, n_samples), the ``tree_depth`` reached
    (n_chains, n_samples), the final ``step_size``, ``n_divergent``,
    ``n_grad_evals`` (one per leaf, gradient cached across leaves), and the
    ``inv_mass`` used.

    Examples
    --------
    NUTS recovers a correlated Gaussian with no trajectory-length tuning:

    >>> import numpy as np
    >>> from mcmc.targets import Gaussian
    >>> target = Gaussian(mean=[1.0, -1.0], cov=[[1.0, 0.8], [0.8, 1.0]])
    >>> rng = np.random.default_rng(0)
    >>> res = nuts(target, np.zeros((2, 2)), n_samples=800, step_size=0.5,
    ...            rng=rng, n_warmup=300, adapt_step_size=True)
    >>> res.samples.shape
    (2, 800, 2)
    >>> bool(np.allclose(res.pooled().mean(axis=0), [1.0, -1.0], atol=0.2))
    True
    """
    x = np.array(x0, dtype=float, copy=True)
    n_chains, dim = x.shape
    sqrt_mass = None if inv_mass is None else 1.0 / np.sqrt(inv_mass)

    logp = np.asarray(target.logpdf(x), dtype=float)
    grad = np.asarray(target.grad_logpdf(x), dtype=float)

    samples = np.empty((n_chains, n_samples, dim))
    delta_H = np.empty((n_chains, n_samples))
    tree_depth = np.empty((n_chains, n_samples), dtype=int)
    n_divergent = 0
    n_grad_evals = 0

    # dual averaging (Hoffman & Gelman 2014, Alg. 5), a single shared step size
    # driven by the acceptance statistic averaged across chains -- mirrors hmc().
    eps = float(step_size)
    gamma, t0, kappa = 0.05, 10.0, 0.75
    mu = np.log(10.0 * eps)
    h_bar, log_eps_bar = 0.0, 0.0

    for it in range(n_warmup + n_samples):
        accept_stats = np.empty(n_chains)
        for c in range(n_chains):
            (
                x[c], grad[c], logp[c], accept_stats[c], dH, depth, ng, nd
            ) = _nuts_transition(
                target, x[c], grad[c], float(logp[c]), eps,
                max_tree_depth, rng, inv_mass, sqrt_mass,
            )
            n_grad_evals += ng
            if it >= n_warmup:
                i = it - n_warmup
                samples[c, i, :] = x[c]
                delta_H[c, i] = dH
                tree_depth[c, i] = depth
                n_divergent += nd

        if it < n_warmup and adapt_step_size:
            t = it + 1
            alpha_bar = float(np.mean(accept_stats))
            h_bar = (1 - 1 / (t + t0)) * h_bar + (target_accept - alpha_bar) / (t + t0)
            log_eps = mu - np.sqrt(t) / gamma * h_bar
            w = t ** (-kappa)
            log_eps_bar = w * log_eps + (1 - w) * log_eps_bar
            eps = float(np.exp(log_eps))
            if it == n_warmup - 1:
                eps = float(np.exp(log_eps_bar))  # freeze at the averaged value

    # per-chain acceptance is subsumed into NUTS's weighting; report 1.0 (every
    # transition advances the state via the multinomial draw, as for Gibbs).
    return SamplerResult(
        samples=samples,
        accept_rate=np.ones(n_chains),
        extras={
            "delta_H": delta_H,
            "tree_depth": tree_depth,
            "step_size": eps,
            "n_divergent": n_divergent,
            "n_grad_evals": n_grad_evals,
            "inv_mass": np.ones(dim) if inv_mass is None else inv_mass,
        },
    )
