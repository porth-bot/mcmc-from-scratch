"""Shared conventions and result container for all samplers.

Target protocol
---------------
Samplers accept a *target object* with:

- ``logpdf(x)``: unnormalized log-density, ``x`` of shape ``(n_chains, dim)``
  -> ``(n_chains,)``. Normalizing constants may be dropped: MCMC only ever
  uses log-density *differences*.
- ``grad_logpdf(x)``: gradient of ``logpdf`` w.r.t. ``x``, same shape as ``x``
  (only required by gradient-based samplers, i.e. HMC).

All chains advance in lockstep as one batched NumPy computation, so running
4 or 32 chains costs nearly the same wall-clock per iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SamplerResult:
    """Output of a sampler run.

    Attributes
    ----------
    samples : ndarray, shape (n_chains, n_samples, dim)
        Post-warmup draws.
    accept_rate : ndarray, shape (n_chains,)
        Fraction of accepted proposals per chain, post-warmup.
        (For Gibbs this is identically 1: full-conditional proposals are
        always accepted -- see ``theory/derivations.md``.)
    extras : dict
        Sampler-specific diagnostics, e.g. HMC stores ``delta_H`` (energy
        errors) and ``step_size`` (post-adaptation).
    """

    samples: np.ndarray
    accept_rate: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def n_chains(self) -> int:
        return self.samples.shape[0]

    @property
    def n_samples(self) -> int:
        return self.samples.shape[1]

    @property
    def dim(self) -> int:
        return self.samples.shape[2]

    def pooled(self) -> np.ndarray:
        """All chains concatenated: shape (n_chains * n_samples, dim)."""
        return self.samples.reshape(-1, self.samples.shape[-1])
