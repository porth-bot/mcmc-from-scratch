"""Bayesian models used in the experiments.

Two roles:

- ``ConjugateLinearRegression`` has a *closed-form* Gaussian posterior, so
  sampler output can be compared to exact answers on a real inference
  problem (not just a hand-picked density).
- ``EightSchools*`` is Rubin's (1981) SAT coaching study, the canonical
  hierarchical model: 8 schools, effect estimates y_j with known standard
  errors sigma_j. Partial pooling happens through the population level
  (mu, tau). No closed form exists; correctness is established by agreement
  between two independent inference routes (conjugate Gibbs on the centered
  parameterization vs HMC on the non-centered one) plus R-hat/ESS.
"""

from __future__ import annotations

import numpy as np

from .gibbs import UpdateFn
from .targets import Gaussian


class ConjugateLinearRegression:
    """Bayesian linear regression with known noise, Gaussian prior.

        y = X beta + eps,  eps ~ N(0, noise_var I),  beta ~ N(0, prior_var I)

    The log posterior (up to a constant) is

        log p(beta | y) = -||y - X beta||^2 / (2 noise_var)
                          - ||beta||^2 / (2 prior_var)

    which is quadratic in beta, so the posterior is Gaussian:

        Sigma_n = (X^T X / noise_var + I / prior_var)^{-1}
        mu_n    = Sigma_n X^T y / noise_var

    Samplers get only the unnormalized log posterior and its gradient
    (anything else would be circular); ``exact_posterior()`` is reserved
    for the comparison.
    """

    def __init__(
        self, X: np.ndarray, y: np.ndarray, noise_var: float, prior_var: float
    ):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.noise_var = float(noise_var)
        self.prior_var = float(prior_var)
        self.dim = self.X.shape[1]

    def logpdf(self, beta: np.ndarray) -> np.ndarray:
        beta = np.atleast_2d(beta)
        resid = self.y - beta @ self.X.T  # (n_chains, n_data)
        return (
            -0.5 * np.sum(resid**2, axis=1) / self.noise_var
            - 0.5 * np.sum(beta**2, axis=1) / self.prior_var
        )

    def grad_logpdf(self, beta: np.ndarray) -> np.ndarray:
        beta = np.atleast_2d(beta)
        resid = self.y - beta @ self.X.T
        return (resid @ self.X) / self.noise_var - beta / self.prior_var

    def exact_posterior(self) -> Gaussian:
        precision = self.X.T @ self.X / self.noise_var + np.eye(self.dim) / self.prior_var
        cov = np.linalg.inv(precision)
        mean = cov @ self.X.T @ self.y / self.noise_var
        return Gaussian(mean, cov)


# Rubin (1981), "Estimation in parallel randomized experiments": estimated
# SAT coaching effects and their standard errors for eight schools.
EIGHT_SCHOOLS_Y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
EIGHT_SCHOOLS_SIGMA = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])


class EightSchoolsNonCentered:
    """Non-centered parameterization for HMC.

    Model (centered form):
        y_j | theta_j ~ N(theta_j, sigma_j^2)     [sigma_j known]
        theta_j | mu, tau ~ N(mu, tau^2)
        p(mu) propto 1,   tau^2 ~ InvGamma(a, b)

    Centered posteriors have funnel geometry: when tau is small the theta_j
    are pinned to mu, and the scale of the theta-conditional varies with
    log tau exactly as in Neal's funnel. The non-centered change of variables

        theta_j = mu + tau * eta_j,   eta_j ~ N(0, 1),   t = log tau

    makes the latent coordinates (mu, t, eta) closer to unit scale.

    Unconstrained state z = (mu, t, eta_1..eta_J), dim = J + 2. Densities
    transform with a Jacobian: with u = tau^2 = e^{2t}, |du/dt| = 2 e^{2t},

        log p(t) = log p_InvGamma(e^{2t}; a, b) + log(2 e^{2t})
                 = -2 a t - b e^{-2t} + const.

    Log posterior (dropping constants), with r_j = y_j - mu - e^t eta_j:

        L(z) = sum_j [ -r_j^2 / (2 sigma_j^2) - eta_j^2 / 2 ] - 2 a t - b e^{-2t}

    Gradient (hand-derived; checked against finite differences in tests):

        dL/dmu   = sum_j r_j / sigma_j^2
        dL/dt    = e^t sum_j (r_j / sigma_j^2) eta_j - 2a + 2b e^{-2t}
        dL/deta_j = e^t r_j / sigma_j^2 - eta_j
    """

    def __init__(
        self,
        y: np.ndarray = EIGHT_SCHOOLS_Y,
        sigma: np.ndarray = EIGHT_SCHOOLS_SIGMA,
        a: float = 1.0,
        b: float = 1.0,
    ):
        self.y = np.asarray(y, dtype=float)
        self.sigma2 = np.asarray(sigma, dtype=float) ** 2
        self.a, self.b = float(a), float(b)
        self.n_schools = len(self.y)
        self.dim = self.n_schools + 2

    def _split(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = np.atleast_2d(z)
        return z[:, 0], z[:, 1], z[:, 2:]

    # Leapfrog trajectories that leave the typical set can push t far enough
    # that e^t / e^{-2t} overflow float64. That is not a bug to paper over:
    # the resulting -inf/NaN log-density makes the Metropolis step reject the
    # proposal (exactly what should happen to a diverged trajectory), so we
    # only silence the warning, not the mechanism.

    def logpdf(self, z: np.ndarray) -> np.ndarray:
        mu, t, eta = self._split(z)
        with np.errstate(over="ignore", invalid="ignore"):
            r = self.y - mu[:, None] - np.exp(t)[:, None] * eta
            return (
                -0.5 * np.sum(r**2 / self.sigma2, axis=1)
                - 0.5 * np.sum(eta**2, axis=1)
                - 2.0 * self.a * t
                - self.b * np.exp(-2.0 * t)
            )

    def grad_logpdf(self, z: np.ndarray) -> np.ndarray:
        mu, t, eta = self._split(z)
        with np.errstate(over="ignore", invalid="ignore"):
            e_t = np.exp(t)
            r = self.y - mu[:, None] - e_t[:, None] * eta
            w = r / self.sigma2  # r_j / sigma_j^2
            g = np.empty_like(np.atleast_2d(z))
            g[:, 0] = np.sum(w, axis=1)
            g[:, 1] = (
                e_t * np.sum(w * eta, axis=1)
                - 2.0 * self.a
                + 2.0 * self.b * np.exp(-2.0 * t)
            )
            g[:, 2:] = e_t[:, None] * w - eta
        return g

    def transform(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """Map unconstrained draws to interpretable parameters.

        z : (..., J+2) -> dict with mu (...), tau (...), theta (..., J).
        """
        mu, t, eta = z[..., 0], z[..., 1], z[..., 2:]
        tau = np.exp(t)
        return {"mu": mu, "tau": tau, "theta": mu[..., None] + tau[..., None] * eta}


def make_eight_schools_gibbs_updates(
    y: np.ndarray = EIGHT_SCHOOLS_Y,
    sigma: np.ndarray = EIGHT_SCHOOLS_SIGMA,
    a: float = 1.0,
    b: float = 1.0,
) -> list[UpdateFn]:
    """Conjugate full conditionals for the *centered* eight-schools model.

    All three blocks are conjugate (derivations in theory/derivations.md,
    Sec. 5.2):

        theta_j | mu, tau2, y ~ N( (y_j/sigma_j^2 + mu/tau2) / P_j,  1/P_j ),
                                   P_j = 1/sigma_j^2 + 1/tau2
        mu | theta, tau2      ~ N( mean(theta), tau2 / J )
        tau2 | theta, mu      ~ InvGamma( a + J/2,  b + sum_j (theta_j - mu)^2 / 2 )

    State: {"theta": (m, J), "mu": (m,), "tau2": (m,)}. InvGamma is sampled
    as the reciprocal of a Gamma(shape, scale=1/rate) draw.
    """
    y = np.asarray(y, dtype=float)
    sigma2 = np.asarray(sigma, dtype=float) ** 2
    J = len(y)

    def update_theta(
        state: dict[str, np.ndarray], rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        prec = 1.0 / sigma2 + 1.0 / state["tau2"][:, None]
        mean = (y / sigma2 + state["mu"][:, None] / state["tau2"][:, None]) / prec
        state["theta"] = mean + rng.standard_normal(mean.shape) / np.sqrt(prec)
        return state

    def update_mu(
        state: dict[str, np.ndarray], rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        m = state["theta"].shape[0]
        state["mu"] = state["theta"].mean(axis=1) + np.sqrt(
            state["tau2"] / J
        ) * rng.standard_normal(m)
        return state

    def update_tau2(
        state: dict[str, np.ndarray], rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        rate = b + 0.5 * np.sum((state["theta"] - state["mu"][:, None]) ** 2, axis=1)
        state["tau2"] = 1.0 / rng.gamma(a + J / 2.0, scale=1.0 / rate)
        return state

    return [update_theta, update_mu, update_tau2]
