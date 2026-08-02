"""The metric is an efficiency knob, never a change of target.

Section 5 of the README reports a *negative* result: on the BNN weight
posterior a diagonal mass matrix buys no efficiency, even though the adapted
scales span ~7x, because permutation and sign symmetries make a marginal
variance something other than the local scale. A negative result is only worth
printing if the two things it rests on are true, so pin both here on a small
model:

1. the adaptation actually fires -- the metric it returns is not the identity
   in disguise, so "no gain" is a statement about the metric rather than about
   nothing having happened;
2. either metric samples the *same* posterior. Changing the kinetic energy
   changes the path through phase space, not the stationary distribution
   (theory/derivations.md Sec. 4.8), so the posterior predictive must agree
   within Monte Carlo error. If it did not, the comparison in section 5 would
   be between two different targets and the ESS numbers would mean nothing.
"""

import numpy as np

from mcmc.bnn import BayesianNNRegression, make_gapped_sine
from mcmc.hmc import hmc

NOISE_STD = 0.1


def _model(seed=0):
    rng = np.random.default_rng(seed)
    X, y = make_gapped_sine(rng, n=24, noise_std=NOISE_STD, gap=(-0.5, 0.5))
    return BayesianNNRegression(X, y, n_hidden=4, noise_std=NOISE_STD,
                                prior_std=1.0)


def _sample(model, adapt_mass, seed=1):
    rng = np.random.default_rng(seed)
    x0 = 0.1 * rng.standard_normal((4, model.dim))
    return hmc(model, x0, n_samples=1500, step_size=0.02, n_leapfrog=20,
               rng=rng, n_warmup=800, adapt_step_size=True,
               target_accept=0.9, adapt_mass=adapt_mass)


def test_identity_metric_is_reported_as_such():
    res = _sample(_model(), adapt_mass=False)
    assert np.allclose(res.extras["inv_mass"], 1.0)


def test_adaptation_returns_a_metric_with_real_spread():
    """Not the identity: the weight coordinates genuinely differ in scale."""
    res = _sample(_model(), adapt_mass=True)
    inv_mass = res.extras["inv_mass"]
    assert inv_mass.shape == (res.samples.shape[2],)
    assert np.all(inv_mass > 0)
    assert inv_mass.max() / inv_mass.min() > 1.5


def test_both_metrics_sample_the_same_posterior_predictive():
    """The kinetic energy changes the trajectory, not the stationary law."""
    model = _model()
    grid = np.linspace(-2.0, 2.0, 40)

    m_id, s_id = model.posterior_predictive(
        _sample(model, adapt_mass=False).samples, grid, include_noise=True)
    m_ad, s_ad = model.posterior_predictive(
        _sample(model, adapt_mass=True).samples, grid, include_noise=True)

    # Tolerance is set by the observation noise (0.1), not by the difference
    # being small: two independent chains on the same posterior should agree on
    # the predictive mean well inside one noise sd.
    assert np.max(np.abs(m_id - m_ad)) < 0.5 * NOISE_STD
    assert np.max(np.abs(s_id - s_ad)) < 0.5 * NOISE_STD
