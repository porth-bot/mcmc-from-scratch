"""Random-walk Metropolis-Hastings.

The Metropolis-Hastings kernel: from state x, propose x' ~ q(. | x), accept
with probability

    alpha(x -> x') = min(1,  pi(x') q(x | x') / [ pi(x) q(x' | x) ])

Accepting with this probability makes the chain satisfy detailed balance
w.r.t. pi (proof in theory/derivations.md, Sec. 2), so pi is stationary; with
an everywhere-positive proposal the chain is also irreducible and aperiodic,
hence ergodic: time averages converge to E_pi[f].

Here q is a symmetric Gaussian random walk, x' = x + step_size * eps with
eps ~ N(0, I), so q(x'|x) = q(x|x') and the Hastings ratio collapses to
pi(x')/pi(x) -- computed in log space to avoid under/overflow.

Tuning fact used in the experiments: for product targets in moderate/high
dimension the asymptotically optimal acceptance rate is ~0.234 (Roberts,
Gelman & Gilks 1997), and the required step size scales like O(d^{-1/2}) --
one of the two reasons RWMH degrades in high dimension (the other: diffusive
exploration covers distance ~ step * sqrt(n) rather than ~ n).
"""

import numpy as np

from .base import SamplerResult


def random_walk_metropolis(target, x0, n_samples, step_size, rng, n_warmup=0):
    """Run batched random-walk Metropolis chains.

    Parameters
    ----------
    target : object with ``logpdf(x)`` batched over chains.
    x0 : ndarray, shape (n_chains, dim)
        Initial states (make them overdispersed if you plan to compute R-hat).
    n_samples : int
        Post-warmup draws to store per chain.
    step_size : float or ndarray broadcastable to (dim,)
        Std-dev of the Gaussian proposal in each coordinate.
    rng : numpy.random.Generator
    n_warmup : int
        Iterations discarded before storage begins.

    Returns
    -------
    SamplerResult with ``samples`` (n_chains, n_samples, dim) and per-chain
    post-warmup ``accept_rate``.
    """
    x = np.array(x0, dtype=float, copy=True)
    n_chains, dim = x.shape
    lp = np.asarray(target.logpdf(x), dtype=float)

    samples = np.empty((n_chains, n_samples, dim))
    n_accept = np.zeros(n_chains)

    for it in range(n_warmup + n_samples):
        prop = x + step_size * rng.standard_normal((n_chains, dim))
        lp_prop = np.asarray(target.logpdf(prop), dtype=float)
        # log U < log[pi(x')/pi(x)]; NaN log-density compares False -> reject.
        accept = np.log(rng.uniform(size=n_chains)) < lp_prop - lp
        x[accept] = prop[accept]
        lp[accept] = lp_prop[accept]
        if it >= n_warmup:
            samples[:, it - n_warmup, :] = x
            n_accept += accept

    return SamplerResult(
        samples=samples,
        accept_rate=n_accept / n_samples,
        extras={"step_size": step_size},
    )
