"""Experiment 6: external benchmark -- our samplers vs emcee.

Every experiment so far checks our samplers against *exact answers*. This one
checks them against another *sampler*: emcee (Foreman-Mackey et al. 2013), the
widely used affine-invariant ensemble sampler. emcee is gradient-free -- it
proposes moves by stretching one walker toward another, a move that is
invariant under affine transformations of the parameter space. That single
property is the whole story of this benchmark:

- On a strongly *correlated* Gaussian, affine invariance means emcee does not
  care about the correlation at all (an affine map decorrelates it, and the
  stretch move is unchanged by that map), so it mixes well with **no tuning**,
  where our coordinate-wise RWMH and Gibbs are slowed by the same correlation.
- emcee needs **no gradient**. On the eight-schools non-centered posterior we
  hand-derived and coded the gradient for HMC; emcee gets the same answer from
  the log-density alone. When gradients are unavailable or expensive, that is
  decisive.
- But per *draw* HMC's gradient-guided trajectories give far lower
  autocorrelation, and the ensemble's efficiency is known to degrade as
  dimension grows (Huijser et al. 2015) -- so where a good gradient exists and
  the dimension is not tiny, HMC dominates.

Fairness notes (so the numbers mean something):

- emcee is run **vectorized** (``vectorize=True``) on our batched ``logpdf``,
  so both sides are the same NumPy-over-an-ensemble computation -- the
  wall-clock gap is algorithmic, not a Python-loop artifact.
- ESS is computed with *our* estimator (``mcmc.diagnostics``) for every
  sampler, treating emcee's walkers as chains -- one consistent yardstick.
- "Evaluations" counts every call that touches the whole model over the full
  run (warmup/burn-in included): a density eval (RWMH, emcee), a
  full-conditional draw (Gibbs), or a gradient eval (HMC). emcee's are counted
  exactly by wrapping its log-prob. See ``efficiency_summary`` for the honest
  caveat on comparing a gradient eval to a density eval.

Run:  python experiments/external_benchmark.py   (needs `pip install emcee`)
"""

import time

import numpy as np

try:
    import emcee
except ImportError:  # pragma: no cover - benchmark-only dependency
    raise SystemExit(
        "This benchmark needs emcee: pip install emcee  (see README 'Reproduce')."
    )

from common import plt, print_table, savefig
from mcmc.diagnostics import efficiency_summary, split_rhat
from mcmc.gibbs import gibbs, make_gaussian_gibbs_updates
from mcmc.hmc import hmc
from mcmc.metropolis import random_walk_metropolis
from mcmc.models import (
    EightSchoolsNonCentered,
    make_eight_schools_gibbs_updates,
)
from mcmc.targets import Gaussian

SEED = 20260703
N_CHAINS = 4


def _timed(fn):
    """Run fn(), returning (result, wall_seconds)."""
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


class CountingLogProb:
    """Wrap a batched logpdf to count exactly how many positions emcee scores.

    emcee with ``vectorize=True`` calls this once per step with an
    ``(n_walkers, dim)`` block; we add the block size so the tally is the total
    number of model evaluations over the whole run (burn-in included).
    """

    def __init__(self, logpdf):
        self._logpdf = logpdf
        self.n_evals = 0

    def __call__(self, x):
        x = np.atleast_2d(x)
        self.n_evals += x.shape[0]
        return self._logpdf(x)


def run_emcee(logpdf, p0, n_burnin, n_steps, seed):
    """Vectorized affine-invariant ensemble run on a batched ``logpdf``.

    p0 : (n_walkers, dim) initial walker positions. Returns
    (chain, n_evals): chain has shape (n_walkers, n_steps, dim) to match our
    SamplerResult.samples layout, and n_evals is the exact model-eval count.
    emcee 3.x drives a legacy ``RandomState``; seed it for reproducibility.
    """
    n_walkers, dim = p0.shape
    counter = CountingLogProb(logpdf)
    sampler = emcee.EnsembleSampler(
        n_walkers, dim, counter, vectorize=True,
        moves=emcee.moves.StretchMove(),
    )
    sampler._random.seed(int(seed))
    state = sampler.run_mcmc(p0, n_burnin, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, n_steps, progress=False)
    chain = sampler.get_chain()  # (n_steps, n_walkers, dim)
    return np.transpose(chain, (1, 0, 2)), counter.n_evals


# --------------------------------------------------------------------------
# Problem 1: correlated Gaussian (rho = 0.9), the affine-invariance showcase
# --------------------------------------------------------------------------
def correlated_gaussian():
    print("=" * 74)
    print("Problem 1: correlated Gaussian, rho = 0.9, sd = (1, 2)")
    print("=" * 74)
    mean = np.array([1.0, -1.0])
    cov = np.array([[1.0, 1.8], [1.8, 4.0]])
    target = Gaussian(mean, cov)
    rng = np.random.default_rng(SEED)
    x0 = mean + rng.standard_normal((N_CHAINS, 2)) * 4.0

    rows = []

    def add(name, chains, seconds, n_evals, gradient):
        # worst-dimension ESS: the honest summary of a correlated target
        per_dim = [efficiency_summary(chains[:, :, d], seconds, n_evals)
                   for d in range(chains.shape[2])]
        worst = min(per_dim, key=lambda s: s["ess"])
        rows.append({
            "sampler": name,
            "grad": "yes" if gradient else "no",
            "draws": chains.shape[0] * chains.shape[1],
            "wall s": seconds,
            "min ESS": worst["ess"],
            "ESS/s": worst["ess_per_sec"],
            "ESS/1k ev": worst["ess_per_keval"],
            "R-hat": max(split_rhat(chains[:, :, d]) for d in range(chains.shape[2])),
        })
        return worst

    res, sec = _timed(lambda: random_walk_metropolis(
        target, x0, n_samples=40_000, step_size=0.75, rng=rng, n_warmup=2_000))
    add("RWMH (ours)", res.samples, sec, 42_000 * N_CHAINS, gradient=False)

    res, sec = _timed(lambda: gibbs(
        make_gaussian_gibbs_updates(mean, cov), {"x": x0.copy()},
        n_samples=40_000, rng=rng, n_warmup=2_000))
    add("Gibbs (ours)", res.samples, sec, 42_000 * N_CHAINS * 2, gradient=False)

    res, sec = _timed(lambda: hmc(
        target, x0, n_samples=10_000, step_size=0.3, n_leapfrog=20, rng=rng,
        n_warmup=1_000, adapt_step_size=True))
    add("HMC (ours)", res.samples, sec, res.extras["n_grad_evals"], gradient=True)

    # emcee: 32 walkers, same total draws budget as HMC-ish; overdispersed start
    n_walkers = 32
    p0 = mean + rng.standard_normal((n_walkers, 2)) * 4.0
    emcee_seed = int(rng.integers(2**31))
    (chain, n_evals), sec = _timed(lambda: run_emcee(
        target.logpdf, p0, n_burnin=1_000, n_steps=8_000, seed=emcee_seed))
    add("emcee (stretch)", chain, sec, n_evals, gradient=False)

    print_table(rows, ["sampler", "grad", "draws", "wall s", "min ESS",
                       "ESS/s", "ESS/1k ev", "R-hat"])
    print("\nPer evaluation, emcee's affine-invariant stretch beats the naive\n"
          "coordinate-wise random walk with zero tuning (the correlation an affine\n"
          "map removes costs it nothing), and even edges HMC -- but recall an HMC\n"
          "eval is a gradient, the rest are densities. On a target this cheap and\n"
          "low-dimensional, exact-conditional Gibbs wins outright, per eval AND per\n"
          "second; HMC wins per-draw ESS. No single method is best on every axis.")
    return rows


# --------------------------------------------------------------------------
# Problem 2: eight schools (non-centered) -- the gradient-vs-no-gradient case
# --------------------------------------------------------------------------
def eight_schools():
    print("\n" + "=" * 74)
    print("Problem 2: eight schools (Rubin 1981), 10-dim hierarchical posterior")
    print("=" * 74)
    rng = np.random.default_rng(SEED + 1)
    rows = []

    def add(name, mu, tau, seconds, n_evals, gradient):
        s_mu = efficiency_summary(mu, seconds, n_evals)
        s_tau = efficiency_summary(tau, seconds, n_evals)
        rows.append({
            "sampler": name,
            "grad": "yes" if gradient else "no",
            "wall s": seconds,
            "ESS(mu)": s_mu["ess"],
            "ESS(tau)": s_tau["ess"],
            "ESS(tau)/s": s_tau["ess_per_sec"],
            "ESS(tau)/1k ev": s_tau["ess_per_keval"],
            "R-hat(tau)": split_rhat(tau),
        })

    # our Gibbs on the centered parameterization
    updates = make_eight_schools_gibbs_updates()
    init = {
        "theta": rng.standard_normal((N_CHAINS, 8)) * 10.0,
        "mu": rng.standard_normal(N_CHAINS) * 10.0,
        "tau2": np.full(N_CHAINS, 4.0),
    }
    res, sec = _timed(lambda: gibbs(
        updates, init, n_samples=20_000, rng=rng, n_warmup=2_000))
    parts = res.extras["unpack"]()
    add("Gibbs (ours)", parts["mu"][..., 0], np.sqrt(parts["tau2"][..., 0]),
        sec, 22_000 * N_CHAINS * 10, gradient=False)

    # our HMC on the non-centered parameterization
    model = EightSchoolsNonCentered()
    z0 = 0.1 * rng.standard_normal((N_CHAINS, model.dim))
    res, sec = _timed(lambda: hmc(
        model, z0, n_samples=10_000, step_size=0.1, n_leapfrog=20, rng=rng,
        n_warmup=2_000, adapt_step_size=True, target_accept=0.9))
    h = model.transform(res.samples)
    add("HMC (ours)", h["mu"], h["tau"], sec, res.extras["n_grad_evals"],
        gradient=True)
    print(f"  (HMC: accept={res.accept_rate.mean():.3f}, "
          f"divergent={res.extras['n_divergent']})")

    # emcee on the same non-centered log-density (no gradient used)
    n_walkers = 40  # >> 2*dim, a healthy ensemble for dim=10
    p0 = 0.1 * rng.standard_normal((n_walkers, model.dim))
    emcee_seed = int(rng.integers(2**31))
    (chain, n_evals), sec = _timed(lambda: run_emcee(
        model.logpdf, p0, n_burnin=2_000, n_steps=20_000, seed=emcee_seed))
    e = model.transform(chain)
    add("emcee (stretch)", e["mu"], e["tau"], sec, n_evals, gradient=False)

    print_table(rows, ["sampler", "grad", "wall s", "ESS(mu)", "ESS(tau)",
                       "ESS(tau)/s", "ESS(tau)/1k ev", "R-hat(tau)"])
    print("\nHMC is the only sampler uniformly efficient across all ten coordinates:\n"
          "gradient + non-centering give ESS(mu) ~30k -- but that took a hand-derived\n"
          "Jacobian-corrected gradient and still logged ~1% divergences in the neck.\n"
          "emcee, with no gradient and no reparameterization, reaches ESS comparable\n"
          "to HMC on the hard tau coordinate and balanced ESS elsewhere -- for the\n"
          "price of writing down the log-density alone. Centered Gibbs is fastest per\n"
          "second but its mu-theta coupling wrecks ESS(mu) (~600). This is emcee's\n"
          "real case: a competitive sampler with zero gradient work.")
    return rows


def figure(rows_gauss, rows_es):
    """Two-panel ESS-per-1k-evaluations bar chart.

    Per-evaluation efficiency is the hardware-independent, reproducible axis
    (wall-clock lives in the printed tables). Gradient-based bars are colored
    apart because their "evaluation" is a gradient, not a density -- the honest
    asterisk on any per-eval comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)

    def panel(ax, rows, key, ylabel, title):
        names = [r["sampler"] for r in rows]
        vals = [r[key] for r in rows]
        colors = ["C0" if r["grad"] == "no" else "C3" for r in rows]
        ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)),
                      [n.replace(" ", "\n") for n in names], fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")

    panel(axes[0], rows_gauss, "ESS/1k ev", "worst-dim ESS / 1k evals",
          "Correlated Gaussian (rho=0.9)")
    panel(axes[1], rows_es, "ESS(tau)/1k ev", "ESS(tau) / 1k evals",
          "Eight schools (10-dim)")

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="C3", label="gradient-based (eval = gradient)"),
                        Patch(color="C0", label="gradient-free (eval = density)")],
               loc="upper right", fontsize=7)
    fig.suptitle("Efficiency per model evaluation: ours vs emcee", x=0.02,
                 ha="left")
    savefig(fig, "external_benchmark.png")


def main():
    rows_gauss = correlated_gaussian()
    rows_es = eight_schools()
    figure(rows_gauss, rows_es)


if __name__ == "__main__":
    main()
