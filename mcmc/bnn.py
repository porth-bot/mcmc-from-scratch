"""Bayesian neural network for 1D regression, sampled by HMC.

This is MCMC on a *real* posterior rather than a hand-picked density: the
unknown is the full weight vector of a small MLP, and the target is its
Bayesian posterior. It is the first model in the repo whose gradient is a
backprop pass rather than a one-line derivative -- and it is exactly the
setting HMC was built for, since the gradient of the log-posterior is the
same quantity training would compute.

Model
-----
A single-hidden-layer tanh network with scalar input and scalar output::

    f(x; theta) = w2 . tanh(W1 x + b1) + b2

with H hidden units. Packed parameter vector (dimension ``3H + 1``)::

    theta = [ W1 (H,) | b1 (H,) | w2 (H,) | b2 (1,) ]

Likelihood is homoscedastic Gaussian, prior is an isotropic Gaussian over
every weight (one shared scale ``prior_std`` -- the simplest honest choice;
it plays the role weight decay plays in training):

    y_i | theta ~ N(f(x_i; theta), noise_std^2)
    theta       ~ N(0, prior_std^2 I)

Log-posterior (up to an additive constant), with residual r_i = y_i - f_i:

    log p(theta | data) = -1/(2 noise_std^2) sum_i r_i^2
                          -1/(2 prior_std^2) ||theta||^2

Gradient (hand-derived, checked against finite differences in
``tests/test_bnn.py``). With g_i = r_i / noise_std^2 the residual sensitivity
and z = tanh(W1 x + b1) the hidden activations,

    d/df_i             : g_i
    dL/db2             = sum_i g_i
    dL/dw2[h]          = sum_i g_i z[h, i]
    delta[h, i]        = w2[h] g_i (1 - z[h, i]^2)      (through the tanh)
    dL/db1[h]          = sum_i delta[h, i]
    dL/dW1[h]          = sum_i delta[h, i] x_i
    minus theta / prior_std^2 added to every block (the Gaussian prior).

Everything is batched over chains: ``theta`` has shape ``(n_chains, dim)`` and
all activations carry a leading chain axis, so 16 chains cost one vectorized
pass. This is standard reverse-mode backprop written out by hand for a
two-layer net; no autodiff is used.

A caution repeated in the experiment writeup: the posterior is invariant to
permuting hidden units and to sign flips (tanh is odd), so it is massively
multimodal in *weight* space. Convergence must therefore be judged in
*function* space (predictions), never by R-hat on raw weights -- see Day 5's
experiment. What HMC gives here is a posterior over functions whose
predictive spread widens where the data leave gaps.
"""

from __future__ import annotations

import numpy as np


class BayesianNNRegression:
    """Posterior over the weights of a 1-hidden-layer tanh MLP (1D regression).

    Parameters
    ----------
    X, y : array_like, shape (n_data,)
        Training inputs and targets (scalar in, scalar out).
    n_hidden : int
        Hidden-unit count H; parameter dimension is ``3H + 1``.
    noise_std : float
        Observation-noise standard deviation in the Gaussian likelihood.
    prior_std : float
        Standard deviation of the isotropic Gaussian weight prior.

    Implements the sampler target protocol (``logpdf``/``grad_logpdf`` on
    ``(n_chains, dim)`` arrays); see ``mcmc/base.py``.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_hidden: int = 32,
        noise_std: float = 0.1,
        prior_std: float = 1.0,
    ):
        self.X = np.asarray(X, dtype=float).ravel()
        self.y = np.asarray(y, dtype=float).ravel()
        if self.X.shape != self.y.shape:
            raise ValueError("X and y must have the same length")
        self.H = int(n_hidden)
        self.noise_var = float(noise_std) ** 2
        self.prior_var = float(prior_std) ** 2
        self.dim = 3 * self.H + 1

    # -- parameter (un)packing -------------------------------------------
    def _unpack(
        self, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(n_chains, dim) -> W1, b1, w2 each (n_chains, H); b2 (n_chains,)."""
        theta = np.atleast_2d(theta)
        H = self.H
        W1 = theta[:, 0:H]
        b1 = theta[:, H : 2 * H]
        w2 = theta[:, 2 * H : 3 * H]
        b2 = theta[:, 3 * H]
        return W1, b1, w2, b2

    # -- forward pass -----------------------------------------------------
    def forward(self, theta: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        """Network output for every chain at every input.

        Returns an array of shape ``(n_chains, n_points)``. ``X`` defaults to
        the training inputs; pass a grid for posterior-predictive evaluation.
        """
        W1, b1, w2, b2 = self._unpack(theta)
        x = self.X if X is None else np.asarray(X, dtype=float).ravel()
        # pre-activation a[c, h, i] = W1[c, h] * x[i] + b1[c, h]
        a = W1[:, :, None] * x[None, None, :] + b1[:, :, None]
        z = np.tanh(a)  # (C, H, N)
        return np.einsum("ch,chi->ci", w2, z) + b2[:, None]

    # -- target protocol --------------------------------------------------
    def logpdf(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        f = self.forward(theta)  # (C, N)
        resid = self.y[None, :] - f
        return (
            -0.5 * np.sum(resid**2, axis=1) / self.noise_var
            - 0.5 * np.sum(theta**2, axis=1) / self.prior_var
        )

    def grad_logpdf(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        W1, b1, w2, b2 = self._unpack(theta)
        x = self.X
        a = W1[:, :, None] * x[None, None, :] + b1[:, :, None]  # (C, H, N)
        z = np.tanh(a)
        f = np.einsum("ch,chi->ci", w2, z) + b2[:, None]  # (C, N)

        g = (self.y[None, :] - f) / self.noise_var  # dL/df_i, (C, N)
        # output layer
        grad_b2 = np.sum(g, axis=1)  # (C,)
        grad_w2 = np.einsum("ci,chi->ch", g, z)  # (C, H)
        # backprop through tanh into the hidden layer
        delta = (w2[:, :, None] * g[:, None, :]) * (1.0 - z**2)  # (C, H, N)
        grad_b1 = np.sum(delta, axis=2)  # (C, H)
        grad_W1 = np.einsum("chi,i->ch", delta, x)  # (C, H)

        grad = np.concatenate(
            [grad_W1, grad_b1, grad_w2, grad_b2[:, None]], axis=1
        )
        return grad - theta / self.prior_var  # add the Gaussian-prior gradient

    # -- convenience for experiments -------------------------------------
    def posterior_predictive(
        self,
        samples: np.ndarray,
        X_grid: np.ndarray,
        include_noise: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predictive mean and std over ``X_grid`` from posterior weight draws.

        ``samples`` is an ``(n_chains, n_samples, dim)`` array (a
        ``SamplerResult.samples``); chains are pooled. With
        ``include_noise=True`` the observation-noise variance is added, giving
        a predictive band for *new observations* rather than for the latent
        function.
        """
        flat = np.asarray(samples).reshape(-1, self.dim)
        preds = self.forward(flat, X_grid)  # (n_draws, n_grid)
        mean = preds.mean(axis=0)
        var = preds.var(axis=0)
        if include_noise:
            var = var + self.noise_var
        return mean, np.sqrt(var)


def train_map(
    model: BayesianNNRegression,
    x0: np.ndarray,
    n_steps: int = 3000,
    lr: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """MAP point estimate(s) by Adam ascent on the log-posterior.

    The single point estimate and every member of a deep ensemble are the same
    computation: maximize ``model.logpdf`` (Gaussian likelihood + Gaussian
    prior, i.e. an L2-regularized least-squares fit) from a starting weight
    vector. The Gaussian prior is exactly the weight decay a trainer would use,
    so this is an honest "train the net" baseline that shares the BNN's model
    and objective -- the only thing HMC adds is sampling the posterior instead
    of climbing to its mode.

    Adam (Kingma & Ba 2015) is written out rather than pulled from a library:
    bias-corrected first/second moment estimates ``m``, ``v`` with the standard
    ``(0.9, 0.999, 1e-8)`` constants. Because ``grad_logpdf`` is batched over
    the leading axis, an entire ensemble trains in one vectorized run -- pass
    ``x0`` of shape ``(n_members, dim)`` and get back the trained members.

    Parameters
    ----------
    model : BayesianNNRegression
    x0 : ndarray (n_members, dim)
        Initial weights, one row per ensemble member. Different rows (different
        random inits) are what give a deep ensemble its spread.
    n_steps, lr : int, float
        Adam iterations and step size.
    rng : unused
        Accepted for a uniform call signature; MAP training is deterministic
        given ``x0``.

    Returns
    -------
    theta : ndarray (n_members, dim)
        Trained weights. Guaranteed (tested) to reach a higher log-posterior
        than the initialization.
    """
    theta = np.atleast_2d(np.array(x0, dtype=float, copy=True))
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, n_steps + 1):
        g = model.grad_logpdf(theta)  # gradient of the log-posterior (ascend)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g**2
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        theta = theta + lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta


def make_gapped_sine(
    rng: np.random.Generator,
    n: int = 40,
    noise_std: float = 0.1,
    gap: tuple[float, float] = (-0.5, 0.5),
) -> tuple[np.ndarray, np.ndarray]:
    """1D toy: y = sin(3x) sampled on [-2, 2] with a hole cut out of the middle.

    The gap is the point of the demo -- a Bayesian posterior should report
    growing predictive uncertainty across the region it never saw. Returns
    ``(X, y)``.
    """
    X = rng.uniform(-2.0, 2.0, size=2 * n)
    X = X[(X < gap[0]) | (X > gap[1])][:n]
    while X.shape[0] < n:  # top up if the rejection sampling came up short
        extra = rng.uniform(-2.0, 2.0, size=n)
        extra = extra[(extra < gap[0]) | (extra > gap[1])]
        X = np.concatenate([X, extra])[:n]
    X.sort()
    y = np.sin(3.0 * X) + noise_std * rng.standard_normal(X.shape[0])
    return X, y
