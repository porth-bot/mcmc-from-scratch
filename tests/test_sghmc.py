"""SGHMC: the friction, measured against a closed form derived by hand.

The method's claim is a negative one -- that stochastic gradients cannot simply
be dropped into Hamiltonian dynamics, and that friction is what repairs them --
so the tests have to be able to tell a repaired sampler from a broken one that
still produces plausible-looking draws. Three layers do that:

1. **An independent oracle for the closed form.** ``sghmc_gaussian_cov`` solves
   a discrete Lyapunov equation numerically. The formulas in ``_oracle`` below
   are the same stationary covariance solved out *by hand* from the two balance
   equations, and share no code with it. Agreement to floating point is
   evidence about the derivation, not just about the linear solve.
2. **The sampler against that closed form**, at several steps and frictions and
   on an anisotropic target, where each coordinate must land on its own
   prediction.
3. **The failure modes, run rather than described.** Without friction the
   variance grows without bound; with friction but no noise correction it
   settles at the inflated value the closed form names, not at the target's.

The noise is injected by ``NoisyGaussian`` rather than by a real minibatch, so
``V`` is a number the test chooses and the prediction is exact. The BNN
posterior appears once, at the end, to check the estimator that has to supply
``V`` when nobody knows it.
"""

import numpy as np
import pytest

from mcmc.bnn import BayesianNNRegression, make_gapped_sine
from mcmc.sghmc import (
    estimate_grad_noise,
    sghmc,
    sghmc_gaussian_cov,
)
from mcmc.targets import Gaussian


class NoisyGaussian:
    """``N(0, s^2 I)`` whose "minibatch" gradient is the exact one plus N(0, V).

    Nothing is subsampled -- the point is a gradient estimator whose noise
    variance is *known exactly*, so the stationary covariance is a number and
    not an estimate. It is unbiased by construction, which is the only property
    the method assumes of a real minibatch gradient.
    """

    def __init__(self, var=1.0, noise_var=1.0, dim=1):
        self.var, self.noise_var, self.dim = float(var), float(noise_var), int(dim)

    def logpdf(self, x):
        return -0.5 * np.sum(np.asarray(x) ** 2, axis=-1) / self.var

    def grad_logpdf(self, x):
        return -np.asarray(x, dtype=float) / self.var

    def grad_logpdf_minibatch(self, x, batch_size, rng):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return self.grad_logpdf(x) + np.sqrt(self.noise_var) * rng.standard_normal(
            x.shape
        )


def _oracle(h, gamma, s2, grad_noise_var=0.0, est_noise_var=None):
    """The stationary variances solved out by hand. See ``sghmc`` relation (3).

    Stationarity of ``Var[theta]`` under ``theta' = theta + h v'`` gives
    ``Cov[theta, v] = (h/2) P``; substituting that into the momentum balance

        P = (1 - h gamma)^2 P + h^2 S / s^4
            - 2 (1 - h gamma) (h / s^2) Cov[theta, v] + q,
        q = 2 gamma h + h^2 (V - Vhat),

    and eliminating ``S`` with (4) leaves one linear equation in P:

        P = (2 gamma + h/s^2 + h dV (1 + h / (2 gamma s^2)))
            / (2 gamma - h gamma^2 + h/s^2 - h^2 gamma/s^2 - h^3/(4 s^4)).

    Returns ``(S, P, C)``: the position variance, the momentum variance and
    their covariance.
    """
    dv = float(grad_noise_var) - (
        float(grad_noise_var) if est_noise_var is None else float(est_noise_var)
    )
    num = 2 * gamma + h / s2 + h * dv * (1.0 + h / (2 * gamma * s2))
    den = (2 * gamma - h * gamma ** 2 + h / s2
           - h ** 2 * gamma / s2 - h ** 3 / (4 * s2 ** 2))
    P = num / den
    S = s2 + h ** 2 * P / 4 + s2 * h * dv / (2 * gamma)
    return S, P, h * P / 2


# ---------------------------------------------------------------------------
# 1. the closed form against the hand-solved one
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "h,gamma,s2,v_true,v_hat",
    [(0.1, 1.0, 1.0, 0.0, 0.0),
     (0.4, 1.5, 1.0, 0.0, 0.0),
     (0.2, 0.5, 2.0, 0.0, 0.0),
     (0.05, 3.0, 0.25, 1.0, 1.0),
     (0.1, 2.0, 1.0, 4.0, 0.0),
     (0.2, 2.0, 3.0, 3.0, 1.0)],
)
def test_lyapunov_solution_matches_the_hand_derivation(h, gamma, s2, v_true, v_hat):
    cov = sghmc_gaussian_cov(h, gamma, s2, grad_noise_var=v_true, est_noise_var=v_hat)
    S, P, C = _oracle(h, gamma, s2, grad_noise_var=v_true, est_noise_var=v_hat)
    assert cov[0, 0] == pytest.approx(S, rel=1e-12)
    assert cov[1, 1] == pytest.approx(P, rel=1e-12)
    assert cov[0, 1] == pytest.approx(C, rel=1e-12)
    assert cov[0, 1] == pytest.approx(cov[1, 0], rel=1e-12)


def test_the_two_errors_separate_by_their_order_in_the_step():
    """Discretization is O(h^2); an uncorrected gradient noise is O(h)/gamma.

    Halving the step quarters the first and halves the second, which is the
    practical content of relation (4): shrinking the step does not rescue an
    uncorrected sampler nearly as fast as it rescues a correct one.

    Both are leading-order statements, so what is asserted is the *approach* to
    the order and not the order at a convenient step: the successive ratios have
    to fall monotonically toward the limit and reach it at the small end. At
    h = 0.4 the discretization ratio reads 4.68, which a test that picked its
    steps carelessly would have called a violation of second order.
    """
    steps = (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125)

    def ratios(errs):
        return [a / b for a, b in zip(errs, errs[1:])]

    disc = [sghmc_gaussian_cov(h, 1.0, 1.0)[0, 0] - 1.0 for h in steps]
    r = ratios(disc)
    assert r == sorted(r, reverse=True)
    assert r[-1] == pytest.approx(4.0, rel=0.02)

    noisy = [sghmc_gaussian_cov(h, 2.0, 1.0, grad_noise_var=4.0,
                                est_noise_var=0.0)[0, 0] - 1.0 for h in steps]
    rn = ratios(noisy)
    assert rn == sorted(rn, reverse=True)
    assert rn[-1] == pytest.approx(2.0, rel=0.02)

    # The gap between the two widens as the step shrinks, which is the point:
    # at h = 0.0125 the uncorrected term is 300x the discretization one.
    assert noisy[-1] / disc[-1] > 100
    assert noisy[-1] / disc[-1] > 4 * (noisy[1] / disc[1])


def test_more_friction_pays_for_the_missing_correction():
    """The O(h) term carries 1/gamma, so raising the friction shrinks it."""
    errs = [sghmc_gaussian_cov(0.05, g, 1.0, grad_noise_var=4.0,
                               est_noise_var=0.0)[0, 0] - 1.0
            for g in (0.5, 1.0, 2.0, 4.0)]
    ratios = [a / b for a, b in zip(errs, errs[1:])]
    assert all(r == pytest.approx(2.0, rel=0.05) for r in ratios), ratios


def test_no_stationary_law_without_friction():
    """gamma = 0 is symplectic Euler: det A = 1, so nothing contracts."""
    with pytest.raises(ValueError, match="no stationary law"):
        sghmc_gaussian_cov(0.1, 0.0, 1.0, grad_noise_var=4.0, est_noise_var=0.0)
    with pytest.raises(ValueError, match="no stationary law"):
        sghmc_gaussian_cov(0.1, 0.0, 1.0)          # even with an exact gradient


def test_friction_condition_is_refused_not_approximated():
    with pytest.raises(ValueError, match="too small for the estimated"):
        sghmc_gaussian_cov(0.1, 0.05, 1.0, grad_noise_var=4.0)
    with pytest.raises(ValueError, match="too small for est_noise_var"):
        sghmc(Gaussian(mean=[0.0], cov=[[1.0]]), np.zeros((2, 1)), n_samples=5,
              step_size=0.1, friction=0.05, rng=np.random.default_rng(0),
              est_noise_var=4.0)


# ---------------------------------------------------------------------------
# 2. the sampler against the closed form
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("h,gamma", [(0.1, 1.0), (0.3, 1.0), (0.3, 3.0), (0.5, 2.0)])
def test_sampler_reproduces_its_own_closed_form(h, gamma):
    """Not "is it roughly N(0,1)" -- the prediction is a specific wrong number
    and the sampler has to hit *that*."""
    rng = np.random.default_rng(11)
    res = sghmc(Gaussian(mean=[0.0], cov=[[1.0]]), np.zeros((512, 1)),
                n_samples=6000, step_size=h, friction=gamma, rng=rng, n_warmup=1000)
    predicted = sghmc_gaussian_cov(h, gamma, 1.0)[0, 0]
    observed = float(res.pooled().var())
    assert observed == pytest.approx(predicted, rel=0.02)


def test_momentum_and_correlation_match_too():
    """Relation (3): the position-momentum correlation is h/2 times Var[v],
    which the target says should be zero. Checked on the sampler's own state."""
    h, gamma = 0.4, 1.0
    rng = np.random.default_rng(3)
    res = sghmc(Gaussian(mean=[0.0], cov=[[1.0]]), np.zeros((4000, 1)),
                n_samples=400, step_size=h, friction=gamma, rng=rng, n_warmup=2000)
    predicted = sghmc_gaussian_cov(h, gamma, 1.0)

    # The last draw of every chain, so the sample is across chains and the
    # position/momentum pair is from one instant rather than a time average.
    theta = res.samples[:, -1, 0]
    v = res.extras["v"][:, 0]
    assert float(v.var()) == pytest.approx(predicted[1, 1], rel=0.06)
    assert float(np.cov(theta, v)[0, 1]) == pytest.approx(predicted[0, 1], rel=0.10)


def test_each_coordinate_lands_on_its_own_prediction():
    """A diagonal Gaussian decouples into independent scalar chains, so the
    same scalar formula has to hold per coordinate at two different s^2 --
    which is the check that the step is not being applied in the wrong units
    somewhere."""
    h, gamma = 0.2, 1.5
    target = Gaussian(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 4.0]])
    rng = np.random.default_rng(5)
    res = sghmc(target, np.zeros((512, 2)), n_samples=8000, step_size=h,
                friction=gamma, rng=rng, n_warmup=2000)
    draws = res.pooled()
    for j, s2 in enumerate((1.0, 4.0)):
        predicted = sghmc_gaussian_cov(h, gamma, s2)[0, 0]
        assert float(draws[:, j].var()) == pytest.approx(predicted, rel=0.03), j


def test_mean_is_unbiased():
    rng = np.random.default_rng(7)
    res = sghmc(Gaussian(mean=[2.0], cov=[[1.0]]), np.full((256, 1), 2.0),
                n_samples=4000, step_size=0.3, friction=1.0, rng=rng, n_warmup=500)
    assert float(res.pooled().mean()) == pytest.approx(2.0, abs=0.02)


def test_run_is_reproducible_and_velocities_come_back():
    kw = dict(n_samples=50, step_size=0.2, friction=1.0, n_warmup=10)
    a = sghmc(Gaussian(mean=[0.0], cov=[[1.0]]), np.zeros((8, 1)),
              rng=np.random.default_rng(2), **kw)
    b = sghmc(Gaussian(mean=[0.0], cov=[[1.0]]), np.zeros((8, 1)),
              rng=np.random.default_rng(2), **kw)
    assert np.array_equal(a.samples, b.samples)
    assert a.extras["v"].shape == (8, 1)
    assert np.all(a.accept_rate == 1.0)
    assert a.extras["adjusted"] is False
    assert a.extras["n_grad_evals"] == 8 * 60


# ---------------------------------------------------------------------------
# 3. the failure modes, run
# ---------------------------------------------------------------------------
def test_naive_stochastic_gradient_hmc_diverges():
    """gamma = 0 with a noisy gradient: the variance grows roughly linearly in
    the number of steps, so doubling the run doubles it. This is the failure the
    friction exists to prevent, and it is measured rather than asserted."""
    target = NoisyGaussian(var=1.0, noise_var=1.0)
    var = []
    for n in (2000, 4000, 8000, 16000):
        rng = np.random.default_rng(0)
        res = sghmc(target, np.zeros((256, 1)), n_samples=1, step_size=0.05,
                    friction=0.0, rng=rng, n_warmup=n, batch_size=1)
        var.append(float(res.pooled().var()))
    assert all(b / a > 1.6 for a, b in zip(var, var[1:])), var
    assert var[-1] > 15.0           # against a target variance of 1


def test_friction_alone_is_not_enough_and_the_shortfall_is_predicted():
    """Friction with no noise correction gives a stationary law, but the wrong
    one -- and the closed form says which wrong one. Both arms are run at the
    same step and friction, so the only difference is ``est_noise_var``."""
    h, gamma, V = 0.1, 2.0, 4.0
    target = NoisyGaussian(var=1.0, noise_var=V)

    rng = np.random.default_rng(1)
    uncorrected = sghmc(target, np.zeros((512, 1)), n_samples=8000, step_size=h,
                        friction=gamma, rng=rng, n_warmup=2000, batch_size=1,
                        est_noise_var=0.0)
    predicted_bad = sghmc_gaussian_cov(h, gamma, 1.0, grad_noise_var=V,
                                       est_noise_var=0.0)[0, 0]
    assert float(uncorrected.pooled().var()) == pytest.approx(predicted_bad, rel=0.03)

    rng = np.random.default_rng(1)
    corrected = sghmc(target, np.zeros((512, 1)), n_samples=8000, step_size=h,
                      friction=gamma, rng=rng, n_warmup=2000, batch_size=1,
                      est_noise_var=V)
    predicted_good = sghmc_gaussian_cov(h, gamma, 1.0, grad_noise_var=V)[0, 0]
    assert float(corrected.pooled().var()) == pytest.approx(predicted_good, rel=0.03)

    # The correction is worth the whole gap: 10.3% too wide becomes 0.28%.
    assert predicted_bad - 1.0 > 30 * (predicted_good - 1.0)


def test_full_batch_run_needs_no_correction_and_matches_the_noise_free_form():
    """With the exact gradient there is nothing to correct for, so the default
    ``est_noise_var=0`` is the right setting and the sampler lands on the
    noise-free prediction."""
    rng = np.random.default_rng(4)
    res = sghmc(NoisyGaussian(var=1.0, noise_var=9.0), np.zeros((512, 1)),
                n_samples=6000, step_size=0.2, friction=1.0, rng=rng,
                n_warmup=1500, batch_size=None)
    predicted = sghmc_gaussian_cov(0.2, 1.0, 1.0)[0, 0]
    assert float(res.pooled().var()) == pytest.approx(predicted, rel=0.02)


# ---------------------------------------------------------------------------
# 4. the noise estimator that has to supply V on a real posterior
# ---------------------------------------------------------------------------
def test_noise_estimator_recovers_a_known_variance():
    target = NoisyGaussian(var=1.0, noise_var=2.5, dim=1)
    est = estimate_grad_noise(target, np.zeros((16, 1)), batch_size=1,
                              rng=np.random.default_rng(0), n_probe=400)
    assert est.shape == (1,)
    assert float(est[0]) == pytest.approx(2.5, rel=0.15)


def test_minibatch_noise_falls_with_batch_size_on_the_bnn():
    """On the repo's own posterior, where V is not a number anyone knows. The
    portable statement is the ordering, not the level: a larger minibatch has
    less gradient noise, and at the full data set the estimator is exact."""
    X, y = make_gapped_sine(n=120, rng=np.random.default_rng(0))
    model = BayesianNNRegression(X, y, n_hidden=4)
    theta = np.zeros((4, model.dim))
    rng = np.random.default_rng(0)
    totals = [float(estimate_grad_noise(model, theta, b, rng, n_probe=64).sum())
              for b in (10, 40, 120)]
    assert totals[0] > totals[1] > totals[2]
    assert totals[2] == pytest.approx(0.0, abs=1e-18)   # no subsampling left


def test_noise_estimator_refuses_a_target_without_the_hook():
    with pytest.raises(TypeError, match="grad_logpdf_minibatch"):
        estimate_grad_noise(Gaussian(mean=[0.0], cov=[[1.0]]), np.zeros((2, 1)),
                            batch_size=1, rng=np.random.default_rng(0))
