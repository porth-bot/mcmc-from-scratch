"""Experiment 1: validate all three samplers against exactly-solvable posteriors.

Part A -- correlated 2D Gaussian (rho = 0.9): every sampler must reproduce the
known mean and covariance; the interesting comparison is *efficiency* (ESS and
ESS per density/gradient evaluation) on a target whose correlation hurts both
coordinatewise Gibbs moves and isotropic random walks.

Part B -- conjugate Bayesian linear regression: samplers see only the
unnormalized log posterior; the exact Gaussian posterior provides the answer
key. This is the same check as Part A but on a posterior arising from data.

Run:  python experiments/validate_exact.py
"""

import numpy as np

from common import plt, print_table, savefig
from mcmc.diagnostics import autocorrelation, ess, integrated_autocorr_time, split_rhat
from mcmc.gibbs import gibbs, make_gaussian_gibbs_updates
from mcmc.hmc import hmc
from mcmc.metropolis import random_walk_metropolis
from mcmc.models import ConjugateLinearRegression
from mcmc.targets import Gaussian

SEED = 20260703
N_CHAINS = 4


def part_a_gaussian():
    print("=" * 72)
    print("Part A: correlated Gaussian, rho = 0.9, sd = (1, 2)")
    print("=" * 72)
    mean = np.array([1.0, -1.0])
    cov = np.array([[1.0, 1.8], [1.8, 4.0]])
    target = Gaussian(mean, cov)
    rng = np.random.default_rng(SEED)
    x0 = mean + rng.standard_normal((N_CHAINS, 2)) * 4.0  # overdispersed

    runs = {}
    res = random_walk_metropolis(
        target, x0, n_samples=40_000, step_size=0.75, rng=rng, n_warmup=2_000
    )
    runs["RWMH"] = (res, res.n_samples * N_CHAINS)  # one density eval per step

    res = gibbs(
        make_gaussian_gibbs_updates(mean, cov),
        {"x": x0.copy()},
        n_samples=40_000,
        rng=rng,
        n_warmup=2_000,
    )
    runs["Gibbs"] = (res, res.n_samples * N_CHAINS * 2)  # one conditional per coord

    res = hmc(
        target, x0, n_samples=10_000, step_size=0.3, n_leapfrog=20, rng=rng,
        n_warmup=1_000, adapt_step_size=True,
    )
    runs["HMC"] = (res, res.extras["n_grad_evals"])

    rows = []
    for name, (res, n_evals) in runs.items():
        pooled = res.pooled()
        mean_err = np.abs(pooled.mean(axis=0) - mean).max()
        cov_err = np.linalg.norm(np.cov(pooled.T) - cov) / np.linalg.norm(cov)
        ess0 = ess(res.samples[:, :, 0])
        rows.append(
            {
                "sampler": name,
                "draws": res.n_samples * N_CHAINS,
                "accept": float(res.accept_rate.mean()),
                "max |mean err|": float(mean_err),
                "rel cov err": float(cov_err),
                "tau(x0)": integrated_autocorr_time(res.samples[:, :, 0]),
                "ESS(x0)": float(ess0),
                "ESS/1k evals": float(1000.0 * ess0 / n_evals),
                "R-hat(x0)": split_rhat(res.samples[:, :, 0]),
            }
        )
    print_table(rows, list(rows[0].keys()))

    # trace plot figure: early mixing from overdispersed starts
    fig, axes = plt.subplots(3, 1, figsize=(7, 5), sharex=True, constrained_layout=True)
    for ax, (name, (res, _)) in zip(axes, runs.items()):
        for c in range(N_CHAINS):
            ax.plot(res.samples[c, :600, 0], lw=0.6, alpha=0.8)
        ax.axhline(mean[0], color="k", lw=0.8, ls="--")
        ax.set_ylabel("$x_0$")
        ax.set_title(f"{name}: 4 chains, first 600 post-warmup draws", loc="left")
    axes[-1].set_xlabel("iteration")
    savefig(fig, "gaussian_traces.png")

    # autocorrelation figure
    fig, ax = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)
    for name, (res, _) in runs.items():
        rho = autocorrelation(res.samples[:, :, 0], max_lag=120)
        ax.plot(rho, label=name, lw=1.4)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("lag")
    ax.set_ylabel(r"autocorrelation of $x_0$")
    ax.set_title(r"Correlated Gaussian ($\rho=0.9$): mixing speed", loc="left")
    ax.legend()
    savefig(fig, "gaussian_autocorr.png")
    return rows


def part_b_linreg():
    print()
    print("=" * 72)
    print("Part B: conjugate Bayesian linear regression (d=3, n=50)")
    print("=" * 72)
    rng = np.random.default_rng(SEED + 1)
    X = rng.standard_normal((50, 3))
    beta_true = np.array([1.5, -2.0, 0.5])
    y = X @ beta_true + 0.7 * rng.standard_normal(50)
    model = ConjugateLinearRegression(X, y, noise_var=0.49, prior_var=10.0)
    post = model.exact_posterior()

    x0 = rng.standard_normal((N_CHAINS, 3)) * 2.0
    res_r = random_walk_metropolis(
        model, x0, n_samples=40_000, step_size=0.09, rng=rng, n_warmup=2_000
    )
    res_h = hmc(
        model, x0, n_samples=10_000, step_size=0.05, n_leapfrog=15, rng=rng,
        n_warmup=1_000, adapt_step_size=True,
    )

    rows = []
    for name, res in [("RWMH", res_r), ("HMC", res_h)]:
        pooled = res.pooled()
        rows.append(
            {
                "sampler": name,
                "accept": float(res.accept_rate.mean()),
                "max |mean err|": float(np.abs(pooled.mean(axis=0) - post.mean).max()),
                "rel cov err": float(
                    np.linalg.norm(np.cov(pooled.T) - post.cov) / np.linalg.norm(post.cov)
                ),
                "min ESS": float(min(ess(res.samples[:, :, i]) for i in range(3))),
                "max R-hat": float(max(split_rhat(res.samples[:, :, i]) for i in range(3))),
            }
        )
    print_table(rows, list(rows[0].keys()))
    print(f"exact posterior mean: {np.round(post.mean, 4)}")

    # exact 1/2-sigma credible ellipses vs HMC draws in the (b0, b1) plane
    fig, ax = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
    pooled = res_h.pooled()
    ax.plot(pooled[::20, 0], pooled[::20, 1], ".", ms=2, alpha=0.25, label="HMC draws")
    sub = post.cov[:2, :2]
    evals, evecs = np.linalg.eigh(sub)
    t = np.linspace(0, 2 * np.pi, 200)
    circ = np.stack([np.cos(t), np.sin(t)])
    for k, ls in [(1, "-"), (2, "--")]:
        e = post.mean[:2, None] + evecs @ (k * np.sqrt(evals)[:, None] * circ)
        ax.plot(e[0], e[1], "k", ls=ls, lw=1.2, label=f"exact {k}$\\sigma$")
    ax.plot(*post.mean[:2], "r+", ms=10, mew=2, label="exact mean")
    ax.set_xlabel(r"$\beta_0$")
    ax.set_ylabel(r"$\beta_1$")
    ax.set_title("Sampled posterior vs closed form", loc="left")
    ax.legend(loc="upper right", fontsize=7)
    savefig(fig, "linreg_posterior.png")


if __name__ == "__main__":
    part_a_gaussian()
    part_b_linreg()
