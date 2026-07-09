"""Metropolis-adjusted Langevin algorithm (MALA).

MALA is the one-step-discretized, Metropolis-corrected overdamped Langevin
diffusion

    dX = grad log pi(X) dt + sqrt(2) dW,

whose stationary law is exactly pi. Discretizing with the Euler-Maruyama scheme
at step ``eps`` gives the proposal

    x' = x + (eps^2 / 2) grad log pi(x) + eps z,     z ~ N(0, I).

The drift ``(eps^2/2) grad log pi`` pushes proposals uphill toward mass, so
MALA moves further per accepted step than random-walk Metropolis while still
being gradient-based like HMC -- it is exactly *one* leapfrog-like step, the
simplest member of the same family. Euler-Maruyama introduces O(eps) bias, so
unlike the exact Langevin diffusion the raw proposal does not preserve pi; a
Metropolis-Hastings accept repairs it exactly.

Because the proposal is not symmetric (the drift makes q(x'|x) != q(x|x')), the
full Hastings correction is required:

    log alpha = log pi(x') - log pi(x) + log q(x | x') - log q(x' | x),

    log q(y | x) = -|| y - x - (eps^2/2) grad log pi(x) ||^2 / (2 eps^2) + C,

and the eps-dependent normalizer C cancels between the two directions. Dropping
the drift (the (eps^2/2) grad terms) recovers the symmetric random walk and the
ratio collapses to pi(x')/pi(x) -- MALA is RWMH plus a gradient-informed drift.

MALA's optimal acceptance rate is ~0.574 with step scaling O(d^{-1/3})
(Roberts & Rosenthal 1998), strictly better than RWMH's 0.234 / O(d^{-1/2}):
the gradient buys a slower curse of dimensionality. This single-step Langevin
proposal is also the exact bridge to score-based generative models -- annealed
(unadjusted) Langevin sampling from a learned ``grad log pi`` is MALA without
the accept step -- which the planned diffusion-from-scratch repo builds on.

MALA is a Metropolis-Hastings kernel, so the accept rule above is exactly the
general one whose detailed-balance proof is in theory/derivations.md, Sec. 2;
the only MALA-specific piece is the asymmetric proposal density ``_log_q``.
"""

import numpy as np

from .base import SamplerResult


def _log_q(y, x, grad_x, step_size):
    """log q(y | x) for the Langevin proposal, up to the eps normalizer that
    cancels in the Hastings ratio. Batched over chains -> (n_chains,)."""
    mean = x + 0.5 * step_size**2 * grad_x
    return -np.sum((y - mean) ** 2, axis=1) / (2.0 * step_size**2)


def mala(target, x0, n_samples, step_size, rng, n_warmup=0):
    """Run batched Metropolis-adjusted Langevin chains.

    Parameters
    ----------
    target : object with batched ``logpdf(x)`` and ``grad_logpdf(x)``.
    x0 : ndarray, shape (n_chains, dim)
        Initial states (overdisperse them if you plan to compute R-hat).
    n_samples : int
        Post-warmup draws stored per chain.
    step_size : float
        Langevin step ``eps``. The proposal std is ``eps`` and the drift is
        ``eps^2 / 2`` times the gradient; too large and every proposal is
        rejected, too small and it diffuses like RWMH.
    rng : numpy.random.Generator
    n_warmup : int
        Iterations discarded before storage begins.

    Returns
    -------
    SamplerResult with ``samples`` (n_chains, n_samples, dim), per-chain
    post-warmup ``accept_rate``, and ``extras`` holding ``step_size`` and
    ``n_grad_evals`` (for ESS-per-gradient comparisons against HMC/NUTS).

    Examples
    --------
    Recover the mean of a correlated 2D Gaussian:

    >>> import numpy as np
    >>> from mcmc.targets import Gaussian
    >>> target = Gaussian(mean=[1.0, -1.0], cov=[[1.0, 0.8], [0.8, 1.0]])
    >>> rng = np.random.default_rng(0)
    >>> res = mala(target, np.zeros((4, 2)), n_samples=3000, step_size=0.6,
    ...            rng=rng, n_warmup=500)
    >>> res.samples.shape
    (4, 3000, 2)
    >>> bool(np.allclose(res.pooled().mean(axis=0), [1.0, -1.0], atol=0.15))
    True
    >>> bool(0.4 < res.accept_rate.mean() < 0.9)
    True
    """
    x = np.array(x0, dtype=float, copy=True)
    n_chains, dim = x.shape
    lp = np.asarray(target.logpdf(x), dtype=float)
    grad = np.asarray(target.grad_logpdf(x), dtype=float)

    samples = np.empty((n_chains, n_samples, dim))
    n_accept = np.zeros(n_chains)
    n_grad_evals = n_chains  # the initial gradient above

    for it in range(n_warmup + n_samples):
        noise = rng.standard_normal((n_chains, dim))
        prop = x + 0.5 * step_size**2 * grad + step_size * noise
        lp_prop = np.asarray(target.logpdf(prop), dtype=float)
        grad_prop = np.asarray(target.grad_logpdf(prop), dtype=float)
        n_grad_evals += n_chains

        # Asymmetric Hastings ratio: the forward and reverse Langevin proposal
        # densities both carry the drift, so neither cancels the way a symmetric
        # random walk's would.
        log_ratio = (
            lp_prop - lp
            + _log_q(x, prop, grad_prop, step_size)   # log q(x  | x')
            - _log_q(prop, x, grad, step_size)        # log q(x' | x )
        )
        # NaN (a proposal into a -inf region) compares False -> reject.
        accept = np.log(rng.uniform(size=n_chains)) < log_ratio
        x[accept] = prop[accept]
        lp[accept] = lp_prop[accept]
        grad[accept] = grad_prop[accept]

        if it >= n_warmup:
            samples[:, it - n_warmup, :] = x
            n_accept += accept

    return SamplerResult(
        samples=samples,
        accept_rate=n_accept / n_samples,
        extras={"step_size": step_size, "n_grad_evals": n_grad_evals},
    )
