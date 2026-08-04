"""SGLD: the two errors it commits, each measured against something exact.

Unadjusted samplers are the easy ones to get wrong-but-plausible, because the
output always *looks* like samples. Every test here compares against a closed
form -- the exact stationary variance of unadjusted Langevin on a Gaussian, or
the full-batch gradient it is estimating -- rather than against "does it look
roughly like the target".
"""

import numpy as np
import pytest

from mcmc.bnn import BayesianNNRegression, make_gapped_sine
from mcmc.mala import mala
from mcmc.sgld import polynomial_schedule, sgld, ula_gaussian_variance
from mcmc.targets import Gaussian


def _std_normal(dim=1):
    return Gaussian(mean=np.zeros(dim), cov=np.eye(dim))


# -- error 1: discretization bias, against the closed form -------------------


def test_stationary_variance_matches_the_closed_form_at_every_step():
    """The load-bearing test. For pi = N(0, s^2) the SGLD update is a Gaussian
    AR(1), so its stationary variance is exactly s^2 / (1 - eps^2/(4 s^2)) --
    no approximation, no asymptotics. A sampler that agreed with pi here would
    be the broken one.

    Stepped in units of the target's own sd, at eps/s = 0.1 to 1.4, where the
    over-dispersion runs from 0.25% to 96%: agreement across that range is not
    one lucky point, and the two target scales confirm the formula depends on
    eps only through eps/s.

    The warmup is set from the chain's relaxation time, 2 s^2 / eps^2 steps.
    That is not a tuning knob -- at eps/s = 0.1 the chain decorrelates 400x
    more slowly than at eps/s = 1.4, and an unadjusted sampler still in its
    transient reports a variance that is too SMALL, which would mask the very
    bias this test is for.
    """
    for s2 in (1.0, 4.0):
        target = Gaussian(mean=[0.0], cov=[[s2]])
        for c in (0.1, 0.3, 0.8, 1.4):
            eps = c * np.sqrt(s2)
            n_warmup = int(max(500, 20 * 2.0 * s2 / eps**2))
            rng = np.random.default_rng(int(abs(hash((s2, c))) % 2**32))
            res = sgld(target, np.zeros((512, 1)), n_samples=6000,
                       step_size=eps, rng=rng, n_warmup=n_warmup)
            measured = float(res.pooled().var())
            predicted = ula_gaussian_variance(eps, s2)
            assert abs(measured - predicted) < 0.05 * predicted, (
                s2, c, measured, predicted)


def test_the_bias_is_an_over_dispersion_and_the_mean_is_unbiased():
    """The direction of the error is not incidental: unadjusted Langevin
    always over-disperses (the drift under-corrects the injected noise), and
    it does so while leaving the mean exactly right. Reporting "SGLD is
    biased" without the direction would hide that its error bars are too wide,
    not too narrow -- the opposite of the RFF failure in the GP repo.
    """
    target = Gaussian(mean=[2.0], cov=[[1.0]])
    rng = np.random.default_rng(1)
    res = sgld(target, np.full((512, 1), 2.0), n_samples=4000, step_size=1.0,
               rng=rng, n_warmup=2000)
    draws = res.pooled()
    assert abs(draws.mean() - 2.0) < 0.02
    assert draws.var() > 1.0
    assert abs(draws.var() - ula_gaussian_variance(1.0, 1.0)) < 0.05


def test_the_accept_step_is_what_mala_buys_with_its_extra_density_evals():
    """Same proposal, same step, same target: MALA's accept step removes the
    bias that SGLD keeps. This is the comparison that says what the accept
    step is FOR, in one number.
    """
    target = Gaussian(mean=[0.0], cov=[[1.0]])
    kwargs = dict(n_samples=4000, step_size=1.0, n_warmup=2000)

    adjusted = mala(target, np.zeros((256, 1)), rng=np.random.default_rng(2),
                    **kwargs)
    unadjusted = sgld(target, np.zeros((256, 1)), rng=np.random.default_rng(2),
                      **kwargs)

    assert abs(adjusted.pooled().var() - 1.0) < 0.03
    assert unadjusted.pooled().var() > 1.03            # 6.7% over by (1)
    assert unadjusted.accept_rate.mean() == 1.0        # nothing is rejected
    assert unadjusted.extras["adjusted"] is False


def test_divergence_above_two_standard_deviations_is_reported_not_hidden():
    """eps >= 2s makes |a| >= 1 and the AR(1) has no stationary law -- a real
    SGLD failure mode. The closed form raises instead of returning a negative
    variance, and the sampler itself genuinely blows up there."""
    with pytest.raises(ValueError):
        ula_gaussian_variance(2.0, 1.0)
    with pytest.raises(ValueError):
        ula_gaussian_variance(3.0, 1.0)

    target = Gaussian(mean=[0.0], cov=[[1.0]])
    res = sgld(target, np.zeros((8, 1)), n_samples=200, step_size=2.5,
               rng=np.random.default_rng(3), n_warmup=200)
    assert res.pooled().var() > 100.0                  # not a sample of N(0,1)


def test_decaying_schedule_shrinks_the_bias():
    """The Robbins-Monro promise, measured: run the same number of iterations
    at a fixed step and under a decaying one that ends near the same place,
    and the decaying chain's stationary variance is closer to the truth.

    What this does NOT show, and the module docstring says so, is that the
    decaying chain is a better sampler at fixed compute -- its steps shrink,
    so it mixes more slowly at exactly the same time.
    """
    target = Gaussian(mean=[0.0], cov=[[1.0]])
    n = 6000

    fixed = sgld(target, np.zeros((512, 1)), n_samples=n, step_size=1.2,
                 rng=np.random.default_rng(4), n_warmup=1000)
    decayed = sgld(target, np.zeros((512, 1)), n_samples=n, step_size=1.2,
                   rng=np.random.default_rng(4), n_warmup=1000,
                   schedule=polynomial_schedule(a=1.2, b=1.0, gamma=0.55))

    assert abs(decayed.pooled().var() - 1.0) < abs(fixed.pooled().var() - 1.0)
    assert decayed.extras["step_size"] < 1.2           # it really did decay


def test_polynomial_schedule_enforces_the_robbins_monro_window():
    with pytest.raises(ValueError):
        polynomial_schedule(a=0.1, gamma=0.5)          # squares still diverge
    with pytest.raises(ValueError):
        polynomial_schedule(a=0.1, gamma=1.5)          # steps sum: chain stalls
    with pytest.raises(ValueError):
        polynomial_schedule(a=-0.1, gamma=0.8)
    sched = polynomial_schedule(a=0.5, b=4.0, gamma=0.75)
    assert sched(0) > sched(100) > sched(10000) > 0.0


# -- error 2: minibatch gradient noise ---------------------------------------


def _bnn(n_data=200, n_hidden=4):
    X, y = make_gapped_sine(n=n_data, rng=np.random.default_rng(0))
    return BayesianNNRegression(X, y, n_hidden=n_hidden, noise_std=0.2,
                                prior_std=1.0)


def test_minibatch_gradient_is_unbiased_for_the_full_gradient():
    """The N/n rescaling is the whole claim. Average enough minibatch
    gradients and they converge on grad_logpdf; the residual falls like
    1/sqrt(n_draws), which is what distinguishes 'unbiased' from 'close'."""
    model = _bnn()
    rng = np.random.default_rng(7)
    theta = 0.3 * rng.standard_normal((1, model.dim))
    exact = model.grad_logpdf(theta)

    errors = []
    for n_draws in (50, 800):
        acc = np.zeros_like(exact)
        r = np.random.default_rng(11)
        for _ in range(n_draws):
            acc += model.grad_logpdf_minibatch(theta, 20, r)
        errors.append(float(np.abs(acc / n_draws - exact).max()))
    assert errors[1] < errors[0] / 2.0, errors
    assert errors[1] < 0.05 * np.abs(exact).max()


def test_full_batch_minibatch_is_exactly_the_full_gradient():
    """Sampling without replacement at batch_size == n_data leaves nothing to
    estimate: the estimator must collapse onto grad_logpdf bit for bit, which
    also pins the N/n scale factor."""
    model = _bnn(n_data=60)
    rng = np.random.default_rng(8)
    theta = 0.3 * rng.standard_normal((3, model.dim))
    assert np.allclose(
        model.grad_logpdf_minibatch(theta, 60, rng), model.grad_logpdf(theta)
    )


def _grad_var(model, theta, batch_size, n_draws=400):
    """Per-coordinate variance of the minibatch gradient, averaged over
    coordinates: the Var[ghat] in the noise-domination argument."""
    ghats = np.concatenate(
        [model.grad_logpdf_minibatch(theta, batch_size, np.random.default_rng(s))
         for s in range(n_draws)]
    )
    return float(np.mean(np.var(ghats, axis=0)))


def test_minibatch_variance_follows_the_without_replacement_formula():
    """Var[ghat] should scale as (1/n)(1 - n/N), not 1/n: sampling without
    replacement carries the finite-population correction, and it is what
    forces the variance to hit exactly zero at n = N. Checked at four batch
    sizes against the ratios the formula predicts.
    """
    model = _bnn(n_data=200)
    theta = 0.3 * np.random.default_rng(9).standard_normal((1, model.dim))

    sizes = (5, 20, 50, 100)
    measured = [_grad_var(model, theta, n) for n in sizes]
    predicted = [(1.0 / n) * (1.0 - n / 200) for n in sizes]

    for (m_a, m_b), (p_a, p_b) in zip(zip(measured, measured[1:]),
                                      zip(predicted, predicted[1:])):
        assert abs((m_a / m_b) / (p_a / p_b) - 1.0) < 0.15, (measured, predicted)


def test_gradient_noise_is_drowned_out_only_below_a_measured_step_size():
    """Welling & Teh's justification for ignoring the minibatch noise, checked
    rather than quoted -- and the check does not fully flatter it.

    Injected noise enters the update with variance eps^2; the gradient noise
    enters through the drift, so with variance (eps^2/2)^2 Var[ghat] =
    O(eps^4). Their ratio is 0.25 eps^2 Var[ghat], which does fall like eps^2,
    exactly a factor of 100 per decade -- the asymptotic claim holds.

    But 'asymptotic' is doing real work. The crossover, where the two noises
    are equal, is at eps = 2 / sqrt(Var[ghat]), and on this BNN posterior with
    batch 20 of 200 that is eps ~ 0.0045 -- BELOW the step sizes a run on this
    posterior actually uses (the smoke test below uses 0.02). So at practical
    steps here the minibatch noise dominates rather than being negligible.
    Larger batches push the crossover up, as sqrt(batch size).
    """
    model = _bnn(n_data=200)
    theta = 0.3 * np.random.default_rng(9).standard_normal((1, model.dim))
    grad_var = _grad_var(model, theta, 20)
    assert grad_var > 0.0

    ratios = [0.25 * eps**2 * grad_var for eps in (0.1, 0.01, 0.001)]
    for a, b in zip(ratios, ratios[1:]):
        assert abs(a / b - 100.0) < 1e-6, ratios       # exactly eps^2 scaling

    crossover = 2.0 / np.sqrt(grad_var)
    assert 0.003 < crossover < 0.007, crossover        # measured ~0.0045
    assert crossover < 0.02                            # i.e. below a usable step
    # Ten times smaller a step and the injected noise is 100x the gradient's.
    assert 0.25 * (crossover / 10) ** 2 * grad_var < 0.011


def test_sgld_runs_on_the_bnn_posterior_with_minibatches():
    """End to end on the repo's real posterior: minibatch SGLD moves, stays
    finite, and lands in the same region as full-batch SGLD from the same
    start. This is a smoke test on purpose -- the actual bias measurement
    against exact HMC is a separate piece of work, not something to assert
    here on a short run.
    """
    model = _bnn(n_data=120, n_hidden=4)
    x0 = 0.1 * np.random.default_rng(13).standard_normal((4, model.dim))

    minibatched = sgld(model, x0, n_samples=400, step_size=0.02,
                       rng=np.random.default_rng(14), n_warmup=200,
                       batch_size=16)
    full = sgld(model, x0, n_samples=400, step_size=0.02,
                rng=np.random.default_rng(14), n_warmup=200)

    assert np.all(np.isfinite(minibatched.samples))
    assert minibatched.extras["batch_size"] == 16
    assert minibatched.extras["n_grad_evals"] == 4 * 600
    # Same order of magnitude in log-density, not the same draws.
    lp_mini = model.logpdf(minibatched.pooled()[::50])
    lp_full = model.logpdf(full.pooled()[::50])
    assert abs(lp_mini.mean() - lp_full.mean()) < 0.5 * abs(lp_full.mean())


def test_minibatching_requires_a_target_that_supports_it():
    """A target without the hook must fail loudly rather than silently running
    full-batch and reporting itself as stochastic-gradient."""
    with pytest.raises(TypeError):
        sgld(_std_normal(), np.zeros((2, 1)), n_samples=5, step_size=0.1,
             rng=np.random.default_rng(0), batch_size=4)
    with pytest.raises(ValueError):
        _bnn(n_data=30).grad_logpdf_minibatch(np.zeros((1, _bnn(30).dim)), 31,
                                              np.random.default_rng(0))


# -- plumbing ----------------------------------------------------------------


def test_result_shape_and_no_logpdf_call():
    """SGLD must never call logpdf -- not needing the full-data density is the
    only reason the method exists. A target that raises on logpdf still works.
    """
    class GradOnly:
        dim = 2

        def logpdf(self, x):
            raise AssertionError("SGLD called logpdf")

        def grad_logpdf(self, x):
            return -np.atleast_2d(x)

    res = sgld(GradOnly(), np.zeros((3, 2)), n_samples=50, step_size=0.3,
               rng=np.random.default_rng(0), n_warmup=10)
    assert res.samples.shape == (3, 50, 2)
    assert res.extras["n_grad_evals"] == 3 * 60
    assert np.allclose(res.accept_rate, 1.0)
