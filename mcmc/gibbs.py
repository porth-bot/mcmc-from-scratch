"""Gibbs sampling: cycle through full-conditional updates.

Gibbs is Metropolis-Hastings whose proposal for block i is the *exact* full
conditional pi(x_i | x_{-i}). Substituting that proposal into the Hastings
ratio makes it identically 1 (derivations.md, Sec. 3), so every proposal is
accepted. The price: you must be able to *sample* each full conditional in
closed form, which in practice means conjugate model structure. There is no
step size to tune, but nothing protects you from slow mixing when components
are strongly coupled -- coordinatewise moves fight the correlation (the
correlated-Gaussian experiment quantifies this).

The driver is model-agnostic: a model is a list of update functions, each
resampling one block of a state dict from its full conditional. Conditionals
live with their models (see ``make_gaussian_gibbs_updates`` below and the
eight-schools model in ``mcmc.models``), because full conditionals are a
property of the model, not of the algorithm.
"""

import numpy as np

from .base import SamplerResult


def gibbs(update_fns, init_state, n_samples, rng, n_warmup=0, store=None):
    """Run a systematic-scan Gibbs sampler over a batched state dict.

    Parameters
    ----------
    update_fns : sequence of callables ``f(state, rng) -> state``
        Each resamples one block from its full conditional given the rest.
        Applied in fixed order each iteration (systematic scan). Each kernel
        leaves pi invariant, so their composition does too.
    init_state : dict[str, ndarray]
        Batched over chains in the leading axis, e.g. ``{"theta": (m, J), "tau2": (m,)}``.
    store : list[str] or None
        Keys to record (default: all).

    Returns
    -------
    SamplerResult where ``samples`` stacks the stored (flattened) keys along
    the last axis, in the order given; ``extras["fields"]`` maps each key to
    its column slice, ``extras["unpack"]`` recovers a dict of arrays.
    """
    state = {k: np.array(v, dtype=float, copy=True) for k, v in init_state.items()}
    store = list(store) if store is not None else list(state.keys())
    n_chains = next(iter(state.values())).shape[0]

    # column layout for flattening the state dict into a samples matrix
    fields, start = {}, 0
    for k in store:
        width = int(np.prod(state[k].shape[1:])) if state[k].ndim > 1 else 1
        fields[k] = slice(start, start + width)
        start += width

    samples = np.empty((n_chains, n_samples, start))
    for it in range(n_warmup + n_samples):
        for f in update_fns:
            state = f(state, rng)
        if it >= n_warmup:
            for k in store:
                samples[:, it - n_warmup, fields[k]] = state[k].reshape(n_chains, -1)

    def unpack(s=samples):
        return {k: s[..., sl] for k, sl in fields.items()}

    return SamplerResult(
        samples=samples,
        accept_rate=np.ones(n_chains),
        extras={"fields": fields, "unpack": unpack},
    )


def make_gaussian_gibbs_updates(mean, cov):
    """Coordinatewise full-conditional updates for N(mean, cov).

    With precision P = Sigma^{-1}, the full conditional of coordinate i is

        x_i | x_{-i} ~ N( mu_i - (1/P_ii) * sum_{j != i} P_ij (x_j - mu_j),  1/P_ii )

    (complete the square in log pi as a quadratic in x_i; derivations.md
    Sec. 3.1). Sampling a Gaussian this way is deliberately redundant -- it
    exists to expose Gibbs mixing behavior where exact answers are known.
    State layout: ``{"x": (n_chains, dim)}``.
    """
    mean = np.asarray(mean, dtype=float)
    P = np.linalg.inv(np.atleast_2d(np.asarray(cov, dtype=float)))
    dim = mean.shape[0]
    cond_sd = 1.0 / np.sqrt(np.diag(P))

    def make_update(i):
        others = [j for j in range(dim) if j != i]

        def update(state, rng):
            x = state["x"]
            delta = x[:, others] - mean[others]
            cond_mean = mean[i] - (delta @ P[others, i]) / P[i, i]
            x[:, i] = cond_mean + cond_sd[i] * rng.standard_normal(x.shape[0])
            return state

        return update

    return [make_update(i) for i in range(dim)]
