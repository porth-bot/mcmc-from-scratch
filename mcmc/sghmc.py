"""Stochastic gradient HMC: momentum, friction, and why the friction is not
optional.

``mcmc/sgld.py`` deleted MALA's accept step and measured what that costs. The
same deletion applied to HMC does *not* work the same way, and the reason is
the point of this module.

HMC's leapfrog integrator is volume preserving and nearly energy conserving,
which is what lets a single accept/reject at the end of a long trajectory fix
the discretization error. Replace the gradient with a minibatch estimate and
that structure is gone: every step injects an independent kick into the
momentum, the Hamiltonian random-walks upward, and there is no accept step left
to notice. The chain does not merely acquire an O(eps) bias the way SGLD does --
on a Gaussian target it has **no stationary distribution at all** and its
variance grows without bound (:func:`sghmc_gaussian_cov` raises there, and
``tests/test_sghmc.py`` watches the variance actually diverge).

Chen, Fox & Guestrin (2014) fix it by adding friction. The dynamics become
underdamped (second-order) Langevin,

    dtheta = v dt,
    dv     = grad log pi(theta) dt - gamma v dt + sqrt(2 gamma) dW,          (1)

whose stationary law is exactly ``pi(theta) x N(v; 0, I)`` for every gamma > 0:
the friction removes energy at exactly the rate the thermal noise adds it. The
gradient noise is then one more energy source, and the fix is to inject
correspondingly less thermal noise. That is the whole method.

The discretization
------------------
Symplectic Euler on (1), with step h, and the gradient replaced by an unbiased
estimate ``ghat = grad log pi + n``, ``n ~ N(0, V)``:

    v_{k+1} = v_k + h ghat(theta_k) - h gamma v_k + sqrt(2 gamma h - h^2 Vhat) xi,
    theta_{k+1} = theta_k + h v_{k+1}.                                       (2)

The momentum update collects noise of variance ``h^2 V`` from the gradient and
``2 gamma h - h^2 Vhat`` from the injected term. If the noise level is known
(``Vhat = V``) the total is ``2 gamma h``, exactly what (1) asks for, and the
only remaining error is the discretization -- which is *not* removed by an
accept step, because there is none. Euler-Maruyama is weak order 1 in general;
on the Gaussian target below the error it leaves in the variance measures
second order (relation (4)), and the mean stays unbiased at every step.

Two conditions fall straight out of that line, and both are enforced:

- **The friction must dominate the gradient noise:** ``2 gamma >= h Vhat``,
  or the injected variance is negative and the scheme is not defined. This is
  Chen et al.'s ``C >= B_hat`` in this parameterization. It is a real
  constraint, not a formality: at a fixed gamma it caps the step size at
  ``2 gamma / Vhat``.
- **gamma = 0 is the naive method**, i.e. stochastic gradients dropped into
  Hamiltonian dynamics with nothing to remove the energy they add. It is
  available here (pass ``friction=0``) so that the failure can be *measured*
  rather than described.

Why the Gaussian case is worth writing down exactly
---------------------------------------------------
For ``pi = N(0, s^2)`` the gradient is linear, so (2) is a two-dimensional
Gaussian AR(1) in ``z = (theta, v)`` and its stationary covariance is the exact
solution of a discrete Lyapunov equation -- closed form, no Monte Carlo, no
asymptotics. :func:`sghmc_gaussian_cov` solves it, and it is the ground truth
every test in this module compares against. It plays the role
``sgld.ula_gaussian_variance`` plays for SGLD, and it answers strictly more:
the marginal variance of theta, the variance of the momentum (which should be
1 and is not), and their correlation (which should be 0 and is not, because
symplectic Euler updates theta with the *new* velocity).

What is here and what is not
----------------------------
The sampler, the closed form, and an estimator for the minibatch gradient noise
(:func:`estimate_grad_noise`) so that the friction condition can be checked on a
real posterior rather than assumed. Not here: a decaying step schedule (the
Robbins-Monro argument is identical to SGLD's and is written up there), and
mass-matrix preconditioning (``experiments/mass_matrix.py`` measures what that
buys for HMC on this repo's posteriors).

References
----------
Chen, Fox & Guestrin (2014), Stochastic gradient Hamiltonian Monte Carlo.
(The friction correction, and the divergence of the naive method.)
Neal (2011), MCMC using Hamiltonian dynamics. (Why the accept step is what
makes HMC exact, which is what is being given up here.)
Leimkuhler & Matthews (2015), Molecular Dynamics. (Discretizations of (1) and
their stationary-law errors; the scheme here is the simplest of them.)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import SamplerResult


def _gaussian_transition(step_size, friction, target_var):
    """The matrix ``A`` of (2) on ``pi = N(0, s^2)``, in ``z = (theta, v)``.

    From (2) with ``grad log pi = -theta / s^2``:

        v'     = (1 - h gamma) v - (h / s^2) theta + w,
        theta' = theta + h v' = (1 - h^2/s^2) theta + h(1 - h gamma) v + h w,

    so the same scalar noise ``w`` enters theta with weight ``h`` and v with
    weight 1 -- which is why the stationary covariance has a nonzero
    theta-v correlation no matter how small h is.
    """
    h, g, s2 = float(step_size), float(friction), float(target_var)
    return np.array([[1.0 - h * h / s2, h * (1.0 - h * g)],
                     [-h / s2, 1.0 - h * g]])


def sghmc_gaussian_cov(
    step_size: float,
    friction: float,
    target_var: float = 1.0,
    grad_noise_var: float = 0.0,
    est_noise_var: Optional[float] = None,
) -> np.ndarray:
    """Exact stationary covariance of ``(theta, v)`` under (2) on ``N(0, s^2)``.

    Ground truth with no Monte Carlo in it. The update is
    ``z' = A z + b w`` with ``w ~ N(0, q)``, ``b = (h, 1)``, and

        q = 2 gamma h - h^2 Vhat + h^2 V,

    the injected noise plus the gradient noise actually present. The stationary
    covariance solves the discrete Lyapunov equation ``Sigma = A Sigma A' + Q``
    with ``Q = q b b'``, which in vectorized form is the 4x4 linear system
    ``(I - A kron A) vec(Sigma) = vec(Q)`` -- solved directly rather than
    iterated, so the answer is exact to floating point.

    Parameters
    ----------
    step_size, friction : float
        ``h`` and ``gamma`` of (2).
    target_var : float
        ``s^2``. Every coordinate of an isotropic Gaussian decouples under (2),
        so the scalar case is the whole story.
    grad_noise_var : float
        ``V``, the variance of the gradient estimate actually used. 0 is the
        full-batch case.
    est_noise_var : float, optional
        ``Vhat``, the level the sampler *believes* and corrects for. Defaults
        to ``grad_noise_var`` (a perfect estimate). Setting it to 0 with
        ``grad_noise_var > 0`` is the uncorrected sampler, which is how the
        cost of ignoring the minibatch noise is measured.

    Raises
    ------
    ValueError
        If the injected variance ``2 gamma h - h^2 Vhat`` is negative (the
        friction condition fails), or if ``A`` has spectral radius >= 1, where
        there is no stationary law -- the naive ``friction=0`` sampler with a
        noisy gradient lands here, and that is the finding, not an inconvenience.

    Notes
    -----
    Three exact relations fall out of the solution, and they are what the
    numbers below mean. Write ``S = Var[theta]``, ``P = Var[v]``,
    ``dV = V - Vhat``:

        Cov[theta, v] = (h / 2) P,                                          (3)
        S = s^2 + (h^2 / 4) P + s^2 h dV / (2 gamma).                       (4)

    (3) is one line: stationarity of ``Var[theta]`` under
    ``theta' = theta + h v'`` forces ``2h Cov[theta, v'] + h^2 P = 0``, and
    ``Cov[theta', v'] = Cov[theta, v'] + h P``. So the position-momentum
    correlation the target says is zero is ``h/2`` times the momentum variance,
    for any friction and any noise -- an artifact of the *ordering* of the
    update, not of the noise.

    (4) then separates the two errors by their order in h. The discretization
    term is O(h^2). The term from an unestimated gradient noise is **O(h) and
    inversely proportional to the friction** -- so at small steps, ignoring the
    minibatch noise is the larger error by a factor of ``2 s^2 dV / (gamma h)``,
    and more friction buys the correction back. ``tests/test_sghmc.py`` checks
    both against a hand-solved formula for P that shares no code with this
    function.

    Examples
    --------
    With no gradient noise the marginal variance of theta is close to the
    target's, and the error is second order in the step:

    >>> import numpy as np
    >>> float(np.round(sghmc_gaussian_cov(0.1, 1.0, 1.0)[0, 0], 6))
    1.002639
    >>> float(np.round(sghmc_gaussian_cov(0.05, 1.0, 1.0)[0, 0], 6))
    1.000641

    The momentum is not standard normal either, and theta and v are correlated
    even though the target says they are independent -- relation (3):

    >>> cov = sghmc_gaussian_cov(0.2, 1.0, 1.0)
    >>> float(np.round(cov[1, 1], 6)), float(np.round(cov[0, 1], 6))
    (1.123596, 0.11236)

    Ignoring the minibatch noise inflates the variance by a computable amount,
    and correcting for it removes exactly that -- the corrected run is
    bit-identical to the noise-free one, because the correction is exact and not
    approximate:

    >>> float(np.round(sghmc_gaussian_cov(0.1, 1.0, 1.0, grad_noise_var=4.0,
    ...                                   est_noise_var=0.0)[0, 0], 6))
    1.203166
    >>> float(np.round(sghmc_gaussian_cov(0.1, 1.0, 1.0, grad_noise_var=4.0)[0, 0], 6))
    1.002639

    Without friction there is nothing to remove the energy the noisy gradient
    adds. The transition matrix has determinant 1 at ``gamma = 0`` -- symplectic
    Euler preserves phase-space volume, which is exactly the property HMC relies
    on -- so its eigenvalues sit on the unit circle and no stationary law
    exists. The variance grows without bound instead, roughly linearly in the
    number of steps (measured in ``tests/test_sghmc.py``):

    >>> sghmc_gaussian_cov(0.1, 0.0, 1.0, grad_noise_var=4.0, est_noise_var=0.0)
    Traceback (most recent call last):
        ...
    ValueError: no stationary law: the transition has spectral radius 1.000000

    Asking for a correction the friction cannot pay for is refused before any
    sampling happens, with the step size that would be admissible:

    >>> sghmc_gaussian_cov(0.1, 0.05, 1.0, grad_noise_var=4.0)
    Traceback (most recent call last):
        ...
    ValueError: friction 0.05 is too small for the estimated gradient noise 4.0 at step 0.1: the correction asks for negative variance. The condition is 2*friction >= step_size*est_noise_var, i.e. step_size <= 0.025
    """
    h, g = float(step_size), float(friction)
    v_true = float(grad_noise_var)
    v_hat = v_true if est_noise_var is None else float(est_noise_var)
    if h <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if g < 0:
        raise ValueError(f"friction must be non-negative, got {friction}")

    injected = 2.0 * g * h - h * h * v_hat
    if injected < 0:
        raise ValueError(
            f"friction {g} is too small for the estimated gradient noise "
            f"{v_hat} at step {h}: the correction asks for negative variance. "
            f"The condition is 2*friction >= step_size*est_noise_var, i.e. "
            f"step_size <= {2 * g / v_hat if v_hat else float('inf'):.6g}"
        )
    q = injected + h * h * v_true

    A = _gaussian_transition(h, g, target_var)
    radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    if radius >= 1.0 - 1e-15:
        raise ValueError(
            f"no stationary law: the transition has spectral radius {radius:.6f}"
        )

    b = np.array([h, 1.0])
    Q = q * np.outer(b, b)
    lhs = np.eye(4) - np.kron(A, A)
    return np.linalg.solve(lhs, Q.reshape(4)).reshape(2, 2)


def estimate_grad_noise(
    target: Any,
    x: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    n_probe: int = 64,
) -> np.ndarray:
    """Per-coordinate variance of the minibatch gradient at ``x``.

    The friction condition ``2 gamma >= h Vhat`` is a statement about a quantity
    nobody knows in advance, so it has to be measured. This draws ``n_probe``
    independent minibatch gradients at the same point and returns their sample
    variance per coordinate, averaged over the chains in ``x``.

    It is a *local* estimate -- the noise varies over the state space, and on a
    posterior it is typically largest where the fit is worst -- so it is a
    diagnostic for choosing gamma, not a quantity the sampler tracks online.
    ``mcmc/sgld.py`` measures the same object for the crossover between the two
    noise sources there; this is that measurement in the form SGHMC needs.

    Returns an array of shape ``(dim,)``.
    """
    if not hasattr(target, "grad_logpdf_minibatch"):
        raise TypeError("target has no grad_logpdf_minibatch to estimate noise from")
    x = np.atleast_2d(np.asarray(x, dtype=float))
    draws = np.stack(
        [np.asarray(target.grad_logpdf_minibatch(x, batch_size, rng), dtype=float)
         for _ in range(n_probe)]
    )                                        # (n_probe, n_chains, dim)
    return draws.var(axis=0, ddof=1).mean(axis=0)


def sghmc(
    target: Any,
    x0: np.ndarray,
    n_samples: int,
    step_size: float,
    friction: float,
    rng: np.random.Generator,
    n_warmup: int = 0,
    batch_size: Optional[int] = None,
    est_noise_var: float = 0.0,
    v0: Optional[np.ndarray] = None,
) -> SamplerResult:
    """Run batched SGHMC chains: scheme (2), unadjusted, one gradient per step.

    Parameters
    ----------
    target : object with batched ``grad_logpdf(x)``; also
        ``grad_logpdf_minibatch(x, batch_size, rng)`` when ``batch_size`` is
        given. ``logpdf`` is never called -- as in SGLD, not needing the
        full-data density is the reason the method exists.
    x0 : ndarray, shape (n_chains, dim)
    step_size, friction : float
        ``h`` and ``gamma``. Note that ``h`` here multiplies the velocity
        directly, unlike ``mcmc/sgld.py`` where the drift carries ``eps^2/2``;
        the two are not the same knob and are not comparable digit for digit.
        Matching them by *gradient evaluations* is the fair comparison, and
        both samplers use exactly one per step per chain.
    est_noise_var : float
        ``Vhat``: the gradient-noise variance to correct for, as one scalar
        applied to every coordinate. The default 0 corrects for nothing, which
        is right for a full-batch run and wrong for a minibatch one -- see
        :func:`estimate_grad_noise` for how to fill it in, and
        :func:`sghmc_gaussian_cov` for what leaving it at 0 costs.
    v0 : ndarray, optional
        Initial velocities. Default draws from ``N(0, I)``, the stationary
        momentum distribution of (1).

    Returns
    -------
    SamplerResult. ``accept_rate`` is identically 1.0 (there is no accept step).
    ``extras`` carries ``step_size``, ``friction``, ``est_noise_var``,
    ``injected_sd``, ``n_grad_evals``, ``batch_size``, ``adjusted=False`` and
    the final velocities as ``v``.

    Examples
    --------
    The sampler reproduces its own closed form, which is the check that the
    implementation and the derivation are the same object:

    >>> import numpy as np
    >>> from mcmc.targets import Gaussian
    >>> rng = np.random.default_rng(0)
    >>> res = sghmc(Gaussian(mean=[0.0], cov=[[1.0]]), np.zeros((256, 1)),
    ...             n_samples=4000, step_size=0.3, friction=1.0, rng=rng,
    ...             n_warmup=500)
    >>> predicted = sghmc_gaussian_cov(0.3, 1.0, 1.0)[0, 0]
    >>> float(np.round(predicted, 4))
    1.0272
    >>> bool(abs(res.pooled().var() - predicted) < 0.02)
    True
    """
    x = np.array(x0, dtype=float, copy=True)
    n_chains, dim = x.shape
    h, g = float(step_size), float(friction)
    if h <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if g < 0:
        raise ValueError(f"friction must be non-negative, got {friction}")
    if batch_size is not None and not hasattr(target, "grad_logpdf_minibatch"):
        raise TypeError(
            "batch_size given but the target has no grad_logpdf_minibatch; "
            "pass batch_size=None to run full-batch"
        )

    injected = 2.0 * g * h - h * h * float(est_noise_var)
    if injected < 0:
        raise ValueError(
            f"friction {g} is too small for est_noise_var {est_noise_var} at "
            f"step {h}: the correction asks for negative variance. Raise the "
            f"friction to at least {h * float(est_noise_var) / 2:.6g}, or "
            f"lower the step."
        )
    injected_sd = float(np.sqrt(injected))

    v = (rng.standard_normal((n_chains, dim)) if v0 is None
         else np.array(v0, dtype=float, copy=True))
    if v.shape != x.shape:
        raise ValueError(f"v0 has shape {v.shape}, expected {x.shape}")

    samples = np.empty((n_chains, n_samples, dim))
    n_grad_evals = 0
    for it in range(n_warmup + n_samples):
        if batch_size is None:
            grad = np.asarray(target.grad_logpdf(x), dtype=float)
        else:
            grad = np.asarray(
                target.grad_logpdf_minibatch(x, batch_size, rng), dtype=float
            )
        n_grad_evals += n_chains

        v = v + h * grad - h * g * v
        if injected_sd > 0:
            v = v + injected_sd * rng.standard_normal((n_chains, dim))
        x = x + h * v

        if it >= n_warmup:
            samples[:, it - n_warmup, :] = x

    return SamplerResult(
        samples=samples,
        accept_rate=np.ones(n_chains),
        extras={
            "step_size": h,
            "friction": g,
            "est_noise_var": float(est_noise_var),
            "injected_sd": injected_sd,
            "n_grad_evals": n_grad_evals,
            "batch_size": batch_size,
            "adjusted": False,
            "v": v,
        },
    )
