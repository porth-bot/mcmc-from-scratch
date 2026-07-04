"""Experiment 3: real-data Bayesian inference -- Rubin's (1981) eight schools.

Eight schools ran SAT coaching programs; school j reports an estimated
treatment effect y_j with known standard error sigma_j. The hierarchical
model

    y_j | theta_j ~ N(theta_j, sigma_j^2)
    theta_j | mu, tau ~ N(mu, tau^2),   p(mu) propto 1,  tau^2 ~ InvGamma(1, 1)

lets the schools share strength: each theta_j is pulled toward the population
mean mu by an amount governed by tau (partial pooling). There is no closed
form, so correctness rests on two *independent* inference routes agreeing:

- Gibbs on the centered parameterization (all three full conditionals are
  conjugate -- derived in mcmc/models.py),
- HMC on the non-centered parameterization (hand-derived gradients),

plus R-hat/ESS diagnostics on both.

Run:  python experiments/eight_schools.py
"""

import numpy as np

from common import plt, print_table, savefig
from mcmc.diagnostics import ess, split_rhat
from mcmc.gibbs import gibbs
from mcmc.hmc import hmc
from mcmc.models import (
    EIGHT_SCHOOLS_SIGMA,
    EIGHT_SCHOOLS_Y,
    EightSchoolsNonCentered,
    make_eight_schools_gibbs_updates,
)

SEED = 20260703
N_CHAINS = 4


def run_gibbs(rng):
    updates = make_eight_schools_gibbs_updates()
    init = {
        "theta": rng.standard_normal((N_CHAINS, 8)) * 10.0,
        "mu": rng.standard_normal(N_CHAINS) * 10.0,
        "tau2": np.full(N_CHAINS, 4.0),
    }
    res = gibbs(updates, init, n_samples=40_000, rng=rng, n_warmup=4_000)
    parts = res.extras["unpack"]()
    return {
        "mu": parts["mu"][..., 0],
        "tau": np.sqrt(parts["tau2"][..., 0]),
        "theta": parts["theta"],
    }


def run_hmc(rng):
    model = EightSchoolsNonCentered()
    z0 = 0.1 * rng.standard_normal((N_CHAINS, model.dim))
    res = hmc(
        model, z0, n_samples=20_000, step_size=0.1, n_leapfrog=20, rng=rng,
        n_warmup=2_000, adapt_step_size=True, target_accept=0.9,
    )
    print(
        f"HMC: accept={res.accept_rate.mean():.3f}, "
        f"step={res.extras['step_size']:.3f}, divergent={res.extras['n_divergent']}"
    )
    return EightSchoolsNonCentered().transform(res.samples)


def table(params, label):
    names = ["mu", "tau"] + [f"theta[{j}]" for j in range(8)]
    series = [params["mu"], params["tau"]] + [params["theta"][:, :, j] for j in range(8)]
    rows = []
    for name, s in zip(names, series):
        rows.append(
            {
                "param": name,
                "mean": float(s.mean()),
                "sd": float(s.std(ddof=1)),
                "ESS": float(ess(s)),
                "R-hat": split_rhat(s),
            }
        )
    print(f"\n{label}")
    print_table(rows, list(rows[0].keys()))
    return rows


def main():
    print("=" * 66)
    print("Eight schools (Rubin 1981): Gibbs (centered) vs HMC (non-centered)")
    print("=" * 66)
    rng = np.random.default_rng(SEED)
    g = run_gibbs(rng)
    h = run_hmc(rng)
    rows_g = table(g, "Gibbs, 4 x 40k draws")
    rows_h = table(h, "HMC, 4 x 20k draws")

    worst = max(
        abs(a["mean"] - b["mean"]) for a, b in zip(rows_g, rows_h)
    )
    print(f"\nlargest |mean difference| across all 10 parameters: {worst:.3f}")

    # agreement figure: posterior of mu and tau from the two routes
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    for ax, key, xlab in [(axes[0], "mu", r"$\mu$"), (axes[1], "tau", r"$\tau$")]:
        for label, params in [("Gibbs (centered)", g), ("HMC (non-centered)", h)]:
            ax.hist(params[key].ravel(), bins=120, density=True, histtype="step",
                    lw=1.3, label=label)
        ax.set_xlabel(xlab)
        ax.set_ylabel("posterior density")
    axes[1].set_xlim(0, 20)
    axes[0].legend(fontsize=7)
    fig.suptitle("Two independent inference routes agree", y=1.04)
    savefig(fig, "eight_schools_agreement.png")

    # shrinkage figure: raw estimates vs partially-pooled posteriors
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    jj = np.arange(8)
    post_mean = h["theta"].reshape(-1, 8).mean(axis=0)
    post_sd = h["theta"].reshape(-1, 8).std(axis=0)
    ax.errorbar(jj - 0.12, EIGHT_SCHOOLS_Y, yerr=EIGHT_SCHOOLS_SIGMA, fmt="o", ms=4,
                capsize=3, label=r"raw $y_j \pm \sigma_j$")
    ax.errorbar(jj + 0.12, post_mean, yerr=post_sd, fmt="s", ms=4, capsize=3,
                label=r"posterior $\theta_j$ (mean $\pm$ sd)")
    mu_mean = float(h["mu"].mean())
    ax.axhline(mu_mean, color="k", lw=0.8, ls="--", label=rf"$E[\mu|y] = {mu_mean:.1f}$")
    ax.set_xticks(jj, [chr(ord("A") + j) for j in jj])
    ax.set_xlabel("school")
    ax.set_ylabel("treatment effect")
    ax.set_title("Partial pooling shrinks noisy estimates toward the population mean",
                 loc="left")
    ax.legend(fontsize=7, loc="upper right")
    savefig(fig, "eight_schools_shrinkage.png")


if __name__ == "__main__":
    main()
