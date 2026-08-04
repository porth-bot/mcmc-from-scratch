"""Stochastic gradient Langevin dynamics (SGLD): MALA with the accept step
deleted and the gradient replaced by a minibatch estimate.

MALA (``mcmc/mala.py``) proposes

    x' = x + (eps^2 / 2) grad log pi(x) + eps z,      z ~ N(0, I),

and then repairs the O(eps) Euler-Maruyama bias with a Metropolis accept. SGLD
(Welling & Teh 2011) keeps the proposal and throws the accept away, for a
reason that is entirely about cost: the accept ratio needs ``log pi(x')``,
which for a posterior over N data points is a full pass over the data. Drop it
and every iteration costs one minibatch. The price is that the chain no longer
targets pi exactly, and this module's job is to measure that price rather than
wave at it.

Two separate errors, and it matters that they are separate:

1. **Discretization bias**, present even with the exact full-batch gradient.
   It is the O(eps) error MALA's accept step removes and SGLD does not. On a
   Gaussian target it is available in closed form, which is what makes this
   module testable rather than merely plausible. For pi = N(0, s^2) the update
   is linear-plus-Gaussian, so its stationary law is exactly Gaussian with

       Var_ULA = s^2 / (1 - eps^2 / (4 s^2)),                            (1)

   derived by solving V = a^2 V + eps^2 with a = 1 - eps^2/(2 s^2). The
   sampler always *over*-disperses, by a factor 1 + eps^2/(4 s^2) + O(eps^4),
   and it diverges outright once eps >= 2s. The mean is unbiased at every eps.

2. **Gradient noise** from the minibatch. Welling & Teh's argument for
   ignoring it is a scaling one: the injected noise contributes variance
   ``eps^2`` per step, while the minibatch gradient enters through the drift
   and so contributes ``(eps^2/2)^2 Var[ghat] = O(eps^4)``. Their ratio is
   ``0.25 eps^2 Var[ghat]``, so as the step shrinks the gradient noise is
   drowned out by noise the sampler wanted anyway.

   ``test_sgld.py`` measures that ratio rather than quoting the argument, and
   the measurement is worth stating plainly: the two noises are equal at
   ``eps = 2 / sqrt(Var[ghat])``, which on this repo's BNN posterior with a
   batch of 20 out of 200 is ``eps ~ 0.0045``. That is *below* the step sizes
   a run on that posterior actually uses. So the asymptotic claim is true and
   the regime it describes is not the regime one samples in; at practical
   steps here the minibatch noise dominates. Bigger batches move the crossover
   up as ``sqrt(batch size)``.

The Robbins-Monro schedule ``eps_t = a (b + t)^{-gamma}``, ``gamma`` in
(0.5, 1], is what makes the first error vanish asymptotically: ``sum eps_t =
inf`` keeps the chain mixing, ``sum eps_t^2 < inf`` kills the discretization
bias. The honest caveat is that this is an asymptotic statement about a chain
whose step size is going to zero, so mixing slows down at the same time the
bias does; a decaying schedule does not give you an exact sampler at finite
compute, it gives you a knob with a documented trade-off.

Where this sits relative to the rest of the repo: SGLD is the same object as
the annealed Langevin sampler in ``diffusion-from-scratch``, which runs
unadjusted Langevin against a *learned* score. That repo's result -- an
unadjusted sampler converges to the stationary law of the score it is handed,
not to the true one -- is the same statement as (1) with a different source of
error in the gradient.

Prototype scope: the sampler and the closed-form checks are here. The
measurement that matters, SGLD's sampling bias against exact HMC on the BNN
posterior already in this repo, is not -- see ``mcmc/bnn.py`` for the
minibatch gradient hook it will use.

References
----------
Welling & Teh (2011), Bayesian learning via stochastic gradient Langevin
dynamics. (SGLD, the polynomial schedule, the noise-domination argument.)
Roberts & Tweedie (1996), Exponential convergence of Langevin distributions
and their discrete approximations. (ULA's bias and its transient behaviour.)
Teh, Thiery & Vollmer (2016), Consistency and fluctuations for stochastic
gradient Langevin dynamics. (What the decaying schedule does and does not buy.)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .base import SamplerResult


def polynomial_schedule(
    a: float, b: float = 1.0, gamma: float = 0.55
) -> Callable[[int], float]:
    """Welling & Teh's step schedule ``eps_t = a (b + t)^{-gamma}``.

    ``gamma`` must lie in (0.5, 1] for the Robbins-Monro conditions to hold:
    at gamma <= 0.5 the squared steps do not sum, so the discretization bias
    does not vanish; above 1 the steps sum, so the chain stops moving before
    it has explored anything.

    Note this returns the step in *this module's* convention, where the
    injected noise has standard deviation ``eps`` and the drift is
    ``eps^2 / 2`` -- matching ``mcmc/mala.py``. Welling & Teh parameterize by
    the noise *variance*, so their ``eps_WT`` is this ``eps`` squared and their
    exponents look different by a factor of two.

    Examples
    --------
    >>> sched = polynomial_schedule(a=0.1, b=10.0, gamma=1.0)
    >>> [round(sched(t), 5) for t in (0, 10, 90)]
    [0.01, 0.005, 0.001]
    """
    if not (0.5 < gamma <= 1.0):
        raise ValueError("gamma must be in (0.5, 1] for Robbins-Monro")
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    return lambda t: a * (b + t) ** (-gamma)


def ula_gaussian_variance(step_size: float, target_var: float) -> float:
    """The exact stationary variance (1) of unadjusted Langevin on N(0, s^2).

    Ground truth for the discretization bias, with no Monte Carlo in it: the
    update ``x' = a x + eps z`` with ``a = 1 - eps^2/(2 s^2)`` is a Gaussian
    AR(1), so its stationary variance is ``eps^2 / (1 - a^2)``, which
    simplifies to ``s^2 / (1 - eps^2 / (4 s^2))``.

    Raises if ``|a| >= 1`` (``eps >= 2 s``), where the recursion has no
    stationary law at all and the chain diverges -- worth an exception rather
    than a negative variance, because that is a real SGLD failure mode and a
    silently negative number would hide it.

    Examples
    --------
    Always an over-estimate, and quadratic in the step:

    >>> round(ula_gaussian_variance(0.2, 1.0), 8)
    1.01010101
    >>> round(ula_gaussian_variance(0.1, 1.0), 8)
    1.00250627
    >>> round(ula_gaussian_variance(1e-6, 1.0), 12)
    1.0
    """
    ratio = step_size**2 / (4.0 * target_var)
    if ratio >= 1.0:
        raise ValueError(
            f"step_size={step_size} exceeds 2*sd={2 * np.sqrt(target_var)}: "
            "unadjusted Langevin has no stationary law here, it diverges"
        )
    return float(target_var / (1.0 - ratio))


def sgld(
    target: Any,
    x0: np.ndarray,
    n_samples: int,
    step_size: float,
    rng: np.random.Generator,
    n_warmup: int = 0,
    batch_size: Optional[int] = None,
    schedule: Optional[Callable[[int], float]] = None,
) -> SamplerResult:
    """Run batched SGLD chains (unadjusted -- there is no accept step).

    Parameters
    ----------
    target : object with batched ``grad_logpdf(x)``. ``logpdf`` is never
        called: not needing it is the entire point of the method.
        If ``batch_size`` is given, the target must also provide
        ``grad_logpdf_minibatch(x, batch_size, rng)`` returning an unbiased
        estimate of ``grad_logpdf(x)``.
    x0 : ndarray, shape (n_chains, dim)
    step_size : float
        The step ``eps``: noise standard deviation ``eps``, drift
        ``eps^2 / 2`` times the gradient (the ``mcmc/mala.py`` convention).
        Ignored when ``schedule`` is given.
    schedule : callable, optional
        ``t -> eps_t`` over the whole run, warmup included, e.g.
        ``polynomial_schedule``. A decaying schedule is what makes the
        discretization bias vanish asymptotically.
    batch_size : int, optional
        Minibatch size for the stochastic gradient. ``None`` (default) uses
        the full-batch gradient, which makes this plain unadjusted Langevin
        (ULA) and isolates error source 1 from error source 2.

    Returns
    -------
    SamplerResult. ``accept_rate`` is identically 1.0 -- nothing is ever
    rejected, which is the honest way to report "there is no accept step"
    within the shared result type. ``extras`` carries ``step_size`` (the
    final one under a schedule), ``n_grad_evals``, ``batch_size`` and
    ``adjusted=False``.

    Examples
    --------
    On a standard normal the sampler over-disperses by exactly the factor (1),
    and it does so reproducibly enough to check against the closed form rather
    than against the target it is supposed to be sampling:

    >>> import numpy as np
    >>> from mcmc.targets import Gaussian
    >>> rng = np.random.default_rng(0)
    >>> target = Gaussian(mean=[0.0], cov=[[1.0]])
    >>> res = sgld(target, np.zeros((256, 1)), n_samples=4000, step_size=0.8,
    ...            rng=rng, n_warmup=500)
    >>> float(res.accept_rate.mean())
    1.0
    >>> predicted = ula_gaussian_variance(0.8, 1.0)
    >>> round(predicted, 4)
    1.1905
    >>> bool(abs(res.pooled().var() - predicted) < 0.02)
    True
    >>> bool(abs(res.pooled().var() - 1.0) > 0.15)      # and it is NOT 1.0
    True
    """
    x = np.array(x0, dtype=float, copy=True)
    n_chains, dim = x.shape
    if batch_size is not None and not hasattr(target, "grad_logpdf_minibatch"):
        raise TypeError(
            "batch_size given but the target has no grad_logpdf_minibatch; "
            "pass batch_size=None to run full-batch (ULA)"
        )

    samples = np.empty((n_chains, n_samples, dim))
    n_grad_evals = 0
    eps = float(step_size)

    for it in range(n_warmup + n_samples):
        if schedule is not None:
            eps = float(schedule(it))
        if batch_size is None:
            grad = np.asarray(target.grad_logpdf(x), dtype=float)
        else:
            grad = np.asarray(
                target.grad_logpdf_minibatch(x, batch_size, rng), dtype=float
            )
        n_grad_evals += n_chains

        noise = rng.standard_normal((n_chains, dim))
        x = x + 0.5 * eps**2 * grad + eps * noise

        if it >= n_warmup:
            samples[:, it - n_warmup, :] = x

    return SamplerResult(
        samples=samples,
        accept_rate=np.ones(n_chains),
        extras={
            "step_size": eps,
            "n_grad_evals": n_grad_evals,
            "batch_size": batch_size,
            "adjusted": False,
        },
    )
