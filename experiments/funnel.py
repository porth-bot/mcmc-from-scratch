"""Experiment 2: Neal's funnel -- where random walks fail and geometry matters.

The funnel (v ~ N(0, 9), x_i | v ~ N(0, e^v), 10 dimensions) has a conditional
scale that varies by orders of magnitude with v. A random-walk proposal must
pick ONE scale: big steps are always rejected in the neck (v << 0), small
steps take forever to cross the mouth (v >> 0). The chain doesn't just mix
slowly -- it systematically under-visits the neck, silently biasing E[v] and
sd(v). HMC's gradient guidance helps but unit-metric HMC with a single step
size still struggles in the neck (divergences are counted and reported, not
hidden).

Ground truth: the v-marginal is exactly N(0, 3^2). An exact i.i.d. sampler
from the generative process provides the reference scatter.

The third act is the fix: the *non-centered* change of variables
x_i = e^{v/2} z_i turns the funnel into an independent Gaussian in (v, z)
(the Jacobian of the transform exactly cancels the varying conditional
scale), where HMC mixes essentially perfectly. Fixing the geometry beats
tuning the sampler -- the same trick the eight-schools model uses.

Run:  python experiments/funnel.py
"""

import numpy as np

from common import plt, print_table, savefig
from mcmc.diagnostics import ess, integrated_autocorr_time, split_rhat
from mcmc.hmc import hmc
from mcmc.metropolis import random_walk_metropolis
from mcmc.targets import Gaussian, NealsFunnel

SEED = 20260703
N_CHAINS = 4
DIM = 10


def main():
    target = NealsFunnel(dim=DIM, sigma_v=3.0)
    rng = np.random.default_rng(SEED)
    x0 = rng.standard_normal((N_CHAINS, DIM))

    print("=" * 78)
    print("Neal's funnel, 10D: v ~ N(0,9), x_i | v ~ N(0, e^v)")
    print("=" * 78)

    res_r = random_walk_metropolis(
        target, x0, n_samples=100_000, step_size=0.6, rng=rng, n_warmup=5_000
    )
    res_h = hmc(
        target, x0, n_samples=25_000, step_size=0.05, n_leapfrog=40, rng=rng,
        n_warmup=2_000, adapt_step_size=True, target_accept=0.9,
    )

    # Non-centered reparameterization: (v, z) with x = e^{v/2} z is exactly
    # N(0, diag(9, 1, ..., 1)). Sample there with HMC, then map back.
    noncentered = Gaussian(np.zeros(DIM), np.diag([9.0] + [1.0] * (DIM - 1)))
    res_n = hmc(
        noncentered, x0, n_samples=25_000, step_size=0.5, n_leapfrog=20, rng=rng,
        n_warmup=2_000, adapt_step_size=True,
    )
    w = res_n.samples
    samples_n = np.concatenate(
        [w[:, :, :1], np.exp(0.5 * w[:, :, :1]) * w[:, :, 1:]], axis=2
    )
    res_n.samples = samples_n  # back in (v, x) coordinates; v itself unchanged

    exact = target.sample(100_000, rng)

    rows = []
    for name, res in [("RWMH", res_r), ("HMC", res_h), ("HMC reparam", res_n)]:
        v = res.samples[:, :, 0]
        rows.append(
            {
                "sampler": name,
                "draws": v.size,
                "accept": float(res.accept_rate.mean()),
                "E[v] (true 0)": float(v.mean()),
                "sd[v] (true 3)": float(v.std(ddof=1)),
                "tau(v)": integrated_autocorr_time(v),
                "ESS(v)": float(ess(v)),
                "R-hat(v)": split_rhat(v),
                "divergent": res.extras.get("n_divergent", 0),
            }
        )
    print_table(rows, list(rows[0].keys()))
    print(f"HMC adapted step size: {res_h.extras['step_size']:.4f}")
    print(f"exact-sampler reference: sd[v] = {exact[:, 0].std(ddof=1):.3f}")

    # money figure: (x_1, v) scatter, exact vs RWMH vs HMC
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharex=True, sharey=True,
                             constrained_layout=True)
    panels = [
        ("exact (i.i.d.)", exact[::10]),
        ("RWMH", res_r.pooled()[::40]),
        ("HMC (centered)", res_h.pooled()[::10]),
        ("HMC (non-centered)", res_n.pooled()[::10]),
    ]
    for ax, (name, pts) in zip(axes, panels):
        ax.plot(pts[:, 1], pts[:, 0], ".", ms=1.5, alpha=0.3)
        ax.set_xlim(-25, 25)
        ax.set_ylim(-10, 10)
        ax.set_title(name, loc="left")
        ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$v$")
    fig.suptitle("Who reaches the neck of the funnel?", y=1.04)
    savefig(fig, "funnel_scatter.png")

    # v-marginal against the exact N(0,9) density
    fig, ax = plt.subplots(figsize=(5.5, 3.4), constrained_layout=True)
    grid = np.linspace(-12, 12, 400)
    ax.plot(grid, np.exp(-grid**2 / 18) / np.sqrt(18 * np.pi), "k", lw=1.5,
            label="true $N(0, 3^2)$")
    for name, res in [("RWMH", res_r), ("HMC centered", res_h), ("HMC non-centered", res_n)]:
        ax.hist(res.samples[:, :, 0].ravel(), bins=100, density=True,
                histtype="step", lw=1.3, label=name)
    ax.set_xlabel("$v$")
    ax.set_ylabel("density")
    ax.set_title("Marginal of $v$: the neck is under-sampled by RWMH", loc="left")
    ax.legend()
    savefig(fig, "funnel_v_marginal.png")

    # trace of v: qualitative mixing
    fig, axes = plt.subplots(2, 1, figsize=(7, 3.8), sharex=True, constrained_layout=True)
    for ax, (name, res) in zip(axes, [("RWMH", res_r), ("HMC", res_h)]):
        ax.plot(res.samples[0, :20_000, 0], lw=0.4)
        ax.set_ylabel("$v$")
        ax.set_ylim(-11, 11)
        ax.set_title(f"{name}: single chain trace of $v$", loc="left")
    axes[-1].set_xlabel("iteration")
    savefig(fig, "funnel_traces.png")


if __name__ == "__main__":
    main()
