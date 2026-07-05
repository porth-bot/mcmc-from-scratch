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
probability to ``target_accept``.
"""

import numpy as np

from .base import SamplerResult

DIVERGENCE_DELTA_H = 25.0  # exp(-25) ~ 1e-11: trajectory left the typical set


def leapfrog(grad_logpdf, x, p, step_size, n_steps):
    """Integrate Hamilton's equations with the leapfrog (Stormer-Verlet) scheme.

    dx/dt = p,   dp/dt = -grad U(x) = +grad log pi(x)

        p_{1/2} = p_0 + (eps/2) grad log pi(x_0)        (half kick)
        x_{k+1} = x_k + eps p_{k+1/2}                   (drift)
        p_{k+3/2} = p_{k+1/2} + eps grad log pi(x_{k+1})(kick)
        ... final half kick to resynchronize p with x.

    Symmetric composition => second-order accurate and reversible:
    running n_steps from (x', -p') returns exactly to (x, -p) up to
    float roundoff (verified in tests/test_hmc.py).
    """
    # A trajectory that escapes the typical set can produce inf gradients and
    # then NaN positions. That IS the divergence signal: the NaN propagates to
    # a -inf acceptance ratio and the proposal is rejected (hmc() below), so
    # only the arithmetic warning is silenced here, not the failure.
    with np.errstate(over="ignore", invalid="ignore"):
        x = x.copy()
        p = p + 0.5 * step_size * grad_logpdf(x)
        for k in range(n_steps):
            x = x + step_size * p
            if k < n_steps - 1:
                p = p + step_size * grad_logpdf(x)
        p = p + 0.5 * step_size * grad_logpdf(x)
    return x, p


def hmc(
    target,
    x0,
    n_samples,
    step_size,
    n_leapfrog,
    rng,
    n_warmup=0,
    adapt_step_size=False,
    target_accept=0.8,
    jitter=0.2,
):
    """Run batched HMC chains with identity mass matrix.

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
    target_accept : float
        Dual-averaging target for the mean acceptance probability. 0.8 is a
        good default; the cost-optimal value for well-behaved targets is
        ~0.65 (Neal 2011, Sec. 5.4.4), while harder geometry rewards higher.

    Returns
    -------
    SamplerResult; ``extras`` holds per-iteration energy errors ``delta_H``
    (n_chains, n_samples), the final ``step_size``, ``n_divergent``, and
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

    # dual-averaging state (Hoffman & Gelman 2014, Alg. 5)
    eps = float(step_size)
    mu, gamma, t0, kappa = np.log(10.0 * eps), 0.05, 10.0, 0.75
    h_bar, log_eps_bar = 0.0, 0.0

    L_low = max(1, int(np.ceil((1.0 - jitter) * n_leapfrog)))

    for it in range(n_warmup + n_samples):
        p0 = rng.standard_normal((n_chains, dim))
        L = int(rng.integers(L_low, n_leapfrog + 1))
        x_prop, p_prop = leapfrog(target.grad_logpdf, x, p0, eps, L)
        n_grad_evals += (L + 1) * n_chains
        lp_prop = np.asarray(target.logpdf(x_prop), dtype=float)

        # -Delta H = [log pi(x') - K(p')] - [log pi(x) - K(p)]
        # Diverged trajectories carry inf/NaN through here by design; they
        # are mapped to -inf below and rejected.
        with np.errstate(over="ignore", invalid="ignore"):
            neg_dH = (lp_prop - 0.5 * np.sum(p_prop**2, axis=1)) - (
                lp - 0.5 * np.sum(p0**2, axis=1)
            )
        neg_dH = np.where(np.isnan(neg_dH), -np.inf, neg_dH)
        accept = np.log(rng.uniform(size=n_chains)) < neg_dH
        x[accept] = x_prop[accept]
        lp[accept] = lp_prop[accept]

        if it < n_warmup:
            if adapt_step_size:
                t = it + 1.0
                alpha = float(np.mean(np.exp(np.minimum(0.0, neg_dH))))
                h_bar = (1 - 1 / (t + t0)) * h_bar + (target_accept - alpha) / (t + t0)
                log_eps = mu - np.sqrt(t) / gamma * h_bar
                w = t ** (-kappa)
                log_eps_bar = w * log_eps + (1 - w) * log_eps_bar
                eps = float(np.exp(log_eps))
                if it == n_warmup - 1:
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
            "n_divergent": n_divergent,
            "n_grad_evals": n_grad_evals,
        },
    )
