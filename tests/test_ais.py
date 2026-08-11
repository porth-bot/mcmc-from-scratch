"""Annealed importance sampling, against normalizers known in closed form.

The point of AIS is a number nothing else in the repo can produce, which means
there is no sampler here to cross-check it against -- so every test below pins
it to a Z that is known exactly, or to a structural identity the derivation
(theory/derivations.md Sec. 7) guarantees regardless of tuning.
"""

import numpy as np
import pytest

from mcmc.ais import AISResult, annealed_importance_sampling, geometric_betas
from mcmc.targets import Gaussian, GaussianMixture


def _run(log_fT, n_temps=100, n_particles=500, seed=0, step_size=1.5,
         n_transition=1, broad=None):
    broad = broad or Gaussian(mean=[0.0, 0.0], cov=[[9.0, 0.0], [0.0, 9.0]])
    return annealed_importance_sampling(
        log_f0=broad.logpdf, sample_f0=broad.sample, log_fT=log_fT,
        betas=geometric_betas(n_temps), n_particles=n_particles,
        rng=np.random.default_rng(seed), step_size=step_size,
        n_transition=n_transition,
    )


# -- the ladder ---------------------------------------------------------------


def test_geometric_betas_spans_zero_to_one():
    b = geometric_betas(20)
    assert b[0] == 0.0 and b[-1] == 1.0 and len(b) == 21
    assert np.all(np.diff(b) > 0)


def test_a_power_ladder_bunches_temperatures_near_zero():
    """The knob it exists for: more rungs where the density moves fastest."""
    flat, bunched = geometric_betas(10), geometric_betas(10, power=3.0)
    assert bunched[1] < flat[1]
    assert np.all(bunched <= flat + 1e-12)


def test_a_malformed_ladder_is_refused():
    g = Gaussian(mean=[0.0], cov=[[1.0]])
    for bad in ([0.0, 0.5], [0.5, 1.0], [0.0, 0.5, 0.4, 1.0], [1.0]):
        with pytest.raises(ValueError):
            annealed_importance_sampling(
                g.logpdf, g.sample, g.logpdf, np.array(bad, dtype=float),
                n_particles=4, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="at least one temperature"):
        geometric_betas(0)


# -- the estimate, against a known Z ------------------------------------------


@pytest.mark.parametrize("log_z", [-4.0, 0.0, 3.0, 11.0])
def test_recovers_a_known_normalizer(log_z):
    """The target is a normalized Gaussian times exp(log_z), so log Z is that
    constant exactly, at any value. An estimator with a scale error would pass
    at one offset and fail at the others."""
    target = Gaussian(mean=[1.0, -1.0], cov=[[1.0, 0.5], [0.5, 1.0]])
    res = _run(lambda x: target.logpdf(x) + log_z)
    assert abs(res.log_z - log_z) < 0.15


def test_a_nonzero_log_z0_shifts_the_answer_by_exactly_that():
    """Z_0 enters as an additive constant on log Z and nothing else -- the
    estimator's job is the *ratio*."""
    target = Gaussian(mean=[0.5, 0.5], cov=[[1.0, 0.0], [0.0, 1.0]])
    broad = Gaussian(mean=[0.0, 0.0], cov=[[9.0, 0.0], [0.0, 9.0]])
    kw = dict(log_f0=broad.logpdf, sample_f0=broad.sample,
              log_fT=lambda x: target.logpdf(x) + 2.0,
              betas=geometric_betas(60), n_particles=300, step_size=1.5)
    a = annealed_importance_sampling(rng=np.random.default_rng(1), **kw)
    b = annealed_importance_sampling(rng=np.random.default_rng(1),
                                     log_z0=7.5, **kw)
    assert b.log_z == pytest.approx(a.log_z + 7.5)


def test_recovers_the_normalizer_of_a_multimodal_target():
    """A two-component mixture, which the transitions have to cross for the
    weights to be right. Modes at +-2.5 with unit variance, so the barrier is
    crossable by a random walk -- this is a check on the estimator, not a claim
    that AIS handles well-separated modes."""
    mix = GaussianMixture(weights=[0.35, 0.65],
                          means=[[-2.5, 0.0], [2.5, 0.0]],
                          covs=np.eye(2))
    res = _run(lambda x: mix.logpdf(x) + 1.25, n_temps=200, n_particles=800,
               step_size=1.5)
    assert abs(res.log_z - 1.25) < 0.2


def test_identical_endpoints_give_exactly_log_z0_and_full_ess():
    """f_0 = f_T: every weight increment is (beta_j - beta_{j-1}) * 0, so the
    weights are identically 1 and the answer is Z_0, exactly -- no Monte Carlo
    error at all. Any off-by-one in which state a rung is evaluated at would
    still pass this, but any error in the *increment* would not."""
    g = Gaussian(mean=[0.3, -0.2], cov=[[2.0, 0.3], [0.3, 1.0]])
    res = annealed_importance_sampling(
        g.logpdf, g.sample, g.logpdf, geometric_betas(25), n_particles=64,
        rng=np.random.default_rng(3), log_z0=1.75)
    assert np.all(res.log_weights == 0.0)
    assert res.log_z == pytest.approx(1.75)
    assert res.ess == pytest.approx(res.n_particles)


# -- the structural properties of the estimator -------------------------------


def test_the_weight_mean_is_unbiased_for_z_even_with_one_temperature():
    """T = 1 is plain importance sampling -- no annealing, no transitions that
    matter -- and the derivation says the weights are *still* unbiased for
    Z_T/Z_0. Averaging exp(log w) over many particles must converge to Z, even
    though log of that mean does not converge to log Z at the same rate."""
    target = Gaussian(mean=[0.4, 0.4], cov=[[1.0, 0.0], [0.0, 1.0]])
    res = _run(lambda x: target.logpdf(x) + 2.0, n_temps=1, n_particles=200000)
    z_hat = np.mean(np.exp(res.log_weights))
    assert z_hat == pytest.approx(np.exp(2.0), rel=0.05)


def test_annealing_buys_variance_not_correctness():
    """The same target through a long ladder and a short one. Both are
    unbiased in Z; the long one has the higher effective sample size, which is
    the entire benefit AIS claims."""
    target = Gaussian(mean=[1.5, 1.5], cov=[[0.3, 0.0], [0.0, 0.3]])
    short = _run(lambda x: target.logpdf(x), n_temps=2, n_particles=2000)
    long = _run(lambda x: target.logpdf(x), n_temps=200, n_particles=2000)
    assert long.ess > 4 * short.ess


def test_log_z_is_below_the_mean_of_the_log_weights_always():
    """Jensen as an identity on whatever weights came out: log of the mean is
    at least the mean of the log, with equality only for equal weights. This is
    the inequality the downward bias in log Z comes from, and it holds run by
    run, with no Monte Carlo caveat."""
    rng = np.random.default_rng(0)
    for spread in (0.0, 0.5, 4.0):
        res = AISResult(log_weights=rng.normal(0.0, spread, size=200),
                        log_z0=0.0, accept_rates=np.ones(200),
                        betas=np.array([0.0, 1.0]))
        assert res.log_z >= np.mean(res.log_weights) - 1e-12
        if spread == 0.0:
            assert res.log_z == pytest.approx(np.mean(res.log_weights))


def test_log_z_is_biased_low_when_the_weights_are_degenerate():
    """And the same inequality as a statement about the *estimator*: over
    independent runs of a badly mismatched importance sampler, the mean log-Z
    estimate sits far below the truth. This is what makes an under-annealed run
    understate the evidence rather than scatter around it.

    The threshold is the run-to-run standard error of these same estimates, not
    a number chosen to pass: the claim is that the bias is many times larger
    than the noise, which at this mismatch it is (about 13 sigma).
    """
    target = Gaussian(mean=[4.0, 4.0], cov=[[0.05, 0.0], [0.0, 0.05]])
    est = np.array([_run(lambda x: target.logpdf(x), n_temps=1, n_particles=60,
                         seed=s).log_z for s in range(200)])
    se = est.std(ddof=1) / np.sqrt(len(est))
    assert est.mean() < -4.0 * se              # truth is 0
    assert est.mean() < -1.0                   # and it is nats, not rounding


def test_the_log_z_bias_shrinks_with_the_particle_count():
    """It is an O(1/N) bias, so more particles at the same (useless) ladder
    still helps -- which is the only reason the jackknife correction is
    meaningful."""
    target = Gaussian(mean=[4.0, 4.0], cov=[[0.05, 0.0], [0.0, 0.05]])
    few = np.mean([_run(lambda x: target.logpdf(x), n_temps=1, n_particles=60,
                        seed=s).log_z for s in range(200)])
    many = np.mean([_run(lambda x: target.logpdf(x), n_temps=1,
                         n_particles=600, seed=s).log_z for s in range(200)])
    assert few < many < 0.0


def test_ess_is_between_one_and_n_and_falls_as_the_weights_spread():
    target = Gaussian(mean=[3.0, 3.0], cov=[[0.1, 0.0], [0.0, 0.1]])
    for n_temps in (1, 10, 300):
        res = _run(lambda x: target.logpdf(x), n_temps=n_temps, n_particles=400)
        assert 1.0 <= res.ess <= res.n_particles + 1e-9
    assert (_run(lambda x: target.logpdf(x), n_temps=300, n_particles=400).ess
            > _run(lambda x: target.logpdf(x), n_temps=1, n_particles=400).ess)


def test_ess_of_equal_weights_is_exactly_n():
    res = AISResult(log_weights=np.full(37, -12.5), log_z0=0.0,
                    accept_rates=np.ones(37), betas=np.array([0.0, 1.0]))
    assert res.ess == pytest.approx(37.0)
    assert res.log_z == pytest.approx(-12.5)


def test_ess_survives_weights_that_would_overflow_in_linear_space():
    """Log weights of a few hundred nats are ordinary in AIS; computing the ESS
    as (sum w)^2 / sum w^2 without the log-sum-exp would return nan."""
    res = AISResult(log_weights=np.array([800.0, 799.0, 300.0]), log_z0=0.0,
                    accept_rates=np.ones(3), betas=np.array([0.0, 1.0]))
    assert np.isfinite(res.ess) and 1.0 <= res.ess <= 3.0
    assert res.log_z == pytest.approx(
        800.0 + np.log(np.mean(np.exp(np.array([0.0, -1.0, -500.0])))))


def test_the_jackknife_moves_the_estimate_up_and_needs_two_particles():
    """The bias it corrects is downward, so on spread-out weights the
    correction is positive."""
    rng = np.random.default_rng(0)
    res = AISResult(log_weights=rng.normal(0.0, 2.0, size=64), log_z0=0.0,
                    accept_rates=np.ones(64), betas=np.array([0.0, 1.0]))
    corrected, correction = res.log_z_jackknife()
    assert correction > 0.0
    assert corrected == pytest.approx(res.log_z + correction)

    with pytest.raises(ValueError, match="two particles"):
        AISResult(log_weights=np.array([0.0]), log_z0=0.0,
                  accept_rates=np.ones(1),
                  betas=np.array([0.0, 1.0])).log_z_jackknife()


def test_the_jackknife_leaves_equal_weights_alone():
    """No spread, no bias, nothing to correct."""
    res = AISResult(log_weights=np.zeros(20), log_z0=3.0,
                    accept_rates=np.ones(20), betas=np.array([0.0, 1.0]))
    corrected, correction = res.log_z_jackknife()
    assert correction == pytest.approx(0.0, abs=1e-12)
    assert corrected == pytest.approx(3.0)


def test_extra_transition_steps_improve_the_ess_at_fixed_ladder():
    """The other knob. More mixing per rung means the particles are closer to
    p_j when the next increment is charged."""
    target = Gaussian(mean=[1.5, 1.5], cov=[[0.25, 0.0], [0.0, 0.25]])
    one = _run(lambda x: target.logpdf(x), n_temps=20, n_particles=600,
               n_transition=1)
    five = _run(lambda x: target.logpdf(x), n_temps=20, n_particles=600,
                n_transition=5)
    assert five.ess > one.ess


def test_returned_particles_and_diagnostics_have_the_right_shapes():
    target = Gaussian(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]])
    res = _run(target.logpdf, n_temps=15, n_particles=32)
    assert res.log_weights.shape == (32,)
    assert res.extras["particles"].shape == (32, 2)
    assert res.accept_rates.shape == (32,)
    assert np.all((res.accept_rates >= 0.0) & (res.accept_rates <= 1.0))
    assert res.betas[0] == 0.0 and res.betas[-1] == 1.0


def test_a_proposal_that_returns_the_wrong_count_is_caught():
    g = Gaussian(mean=[0.0], cov=[[1.0]])
    with pytest.raises(ValueError, match="expected 10"):
        annealed_importance_sampling(
            g.logpdf, lambda n, rng: g.sample(n - 1, rng), g.logpdf,
            geometric_betas(5), n_particles=10, rng=np.random.default_rng(0))
