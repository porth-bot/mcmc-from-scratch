"""Parallel tempering (replica-exchange MCMC) for multimodal targets.

A single random-walk (or HMC) chain on a well-separated multimodal target is
trapped: the low-density barrier between modes is crossed only with
exponentially small probability, so the chain reports whichever mode it
started in with badly wrong weights. Parallel tempering fixes this by running
K replicas at inverse temperatures

    1 = beta_0 > beta_1 > ... > beta_{K-1} > 0,

replica k targeting the *tempered* density pi_{beta_k}(x) proportional to
pi(x)^{beta_k}. Small beta flattens the landscape -- the barrier between modes
shrinks and the hot replicas roam freely between them. Periodic *swaps* of
configurations between adjacent replicas then ferry that mobility down the
temperature ladder to the cold (beta = 1) replica, whose draws are the ones
kept: they are exact samples of pi, but with the mode-hopping imported from
the hot chains.

Two moves, both leaving the product distribution prod_k pi_{beta_k}(x_k)
invariant:

1.  **Local.** Each replica takes a random-walk Metropolis step on its own
    tempered density (accept with pi_{beta_k}(x')/pi_{beta_k}(x) =
    exp(beta_k [log pi(x') - log pi(x)])).
2.  **Swap.** Propose exchanging the states of adjacent replicas i, j. This is
    a Metropolis move on the product target with a deterministic swap
    proposal, so it is accepted with probability min(1, exp(Delta)) where

        Delta = (beta_i - beta_j) (log pi(x_j) - log pi(x_i)).

    (Derivation: the product density picks up exactly this ratio when the two
    configurations trade places; the proposal is its own inverse, so the
    Hastings factor is 1.) Swaps are attempted on disjoint even/odd adjacent
    pairs, alternating parity each sweep, so several can be applied at once
    without interfering.

Only the target's ``logpdf`` is used (gradient-free), so tempering works on
any target in the repo. The temperature ladder is the one tuning knob: too few
rungs and adjacent tempered densities barely overlap, so swaps are rejected
and mobility never reaches the cold chain -- ``extras['swap_rates']`` reports
the per-pair acceptance to diagnose exactly that.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import SamplerResult


def geometric_ladder(n_replicas: int, beta_min: float = 0.01) -> np.ndarray:
    """Inverse temperatures 1 = beta_0 > ... > beta_{K-1} = beta_min, spaced
    geometrically. A geometric ladder keeps the overlap between adjacent
    tempered densities roughly constant, the usual first choice."""
    if n_replicas == 1:
        return np.array([1.0])
    return np.geomspace(1.0, beta_min, n_replicas)


def parallel_tempering(
    target: Any,
    x0: np.ndarray,
    n_samples: int,
    step_sizes: np.ndarray,
    betas: np.ndarray,
    rng: np.random.Generator,
    n_warmup: int = 0,
    swap_every: int = 1,
) -> SamplerResult:
    """Run replica-exchange MCMC and return the cold (beta = 1) chain.

    Parameters
    ----------
    target : object with a batched ``logpdf`` (gradient not required).
    x0 : ndarray (n_replicas, dim)
        Initial state of each replica.
    step_sizes : ndarray (n_replicas,)
        Random-walk proposal scale per replica; hotter replicas (smaller beta)
        usually take larger steps.
    betas : ndarray (n_replicas,)
        Inverse temperatures, descending, with ``betas[0] == 1`` (the cold
        replica whose samples are returned).
    swap_every : int
        Attempt a swap sweep every this many local steps.

    Returns
    -------
    SamplerResult whose ``samples`` is the cold chain, shape
    ``(1, n_samples, dim)``. ``extras`` holds ``swap_rates`` (per adjacent
    pair), ``local_accept`` (per replica), and the ``betas`` used.
    """
    x = np.array(x0, dtype=float, copy=True)
    K, dim = x.shape
    betas = np.asarray(betas, dtype=float)
    step_sizes = np.broadcast_to(np.asarray(step_sizes, dtype=float), (K,))
    if betas[0] != 1.0:
        raise ValueError("betas[0] must be 1.0 (the cold, target replica)")

    lp = np.asarray(target.logpdf(x), dtype=float)  # untempered log pi at each replica
    samples = np.empty((1, n_samples, dim))
    local_accept = np.zeros(K)
    swap_attempts = np.zeros(K - 1)
    swap_accepts = np.zeros(K - 1)
    parity = 0

    for it in range(n_warmup + n_samples):
        # 1. local random-walk Metropolis on each tempered density
        prop = x + step_sizes[:, None] * rng.standard_normal((K, dim))
        lp_prop = np.asarray(target.logpdf(prop), dtype=float)
        accept = np.log(rng.uniform(size=K)) < betas * (lp_prop - lp)
        x[accept] = prop[accept]
        lp[accept] = lp_prop[accept]

        # 2. swap sweep over disjoint adjacent pairs (alternating parity)
        if swap_every and it % swap_every == 0:
            pairs = np.arange(parity, K - 1, 2)
            parity ^= 1
            for i in pairs:
                delta = (betas[i] - betas[i + 1]) * (lp[i + 1] - lp[i])
                swap_attempts[i] += 1
                if np.log(rng.uniform()) < delta:
                    x[[i, i + 1]] = x[[i + 1, i]]
                    lp[[i, i + 1]] = lp[[i + 1, i]]
                    swap_accepts[i] += 1

        if it >= n_warmup:
            samples[0, it - n_warmup, :] = x[0]
            local_accept += accept

    with np.errstate(invalid="ignore", divide="ignore"):
        swap_rates = np.where(swap_attempts > 0, swap_accepts / swap_attempts, np.nan)
    return SamplerResult(
        samples=samples,
        accept_rate=local_accept[:1] / n_samples,  # cold-chain local acceptance
        extras={
            "swap_rates": swap_rates,
            "local_accept": local_accept / n_samples,
            "betas": betas,
        },
    )
