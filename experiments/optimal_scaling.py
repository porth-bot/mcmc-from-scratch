"""Experiment 4: the 0.234 rule -- optimal step size for random-walk Metropolis.

Roberts, Gelman & Gilks (1997) analyzed RWMH on a product target
pi(x) = prod_i f(x_i) in the high-dimensional limit. Scaling the proposal as
sigma = l / sqrt(d), the rescaled first coordinate converges to a Langevin
diffusion whose speed is maximized at a specific l -- and at that optimum the
average acceptance probability tends to **0.234**. Too small a step: high
acceptance but tiny moves. Too large: big proposed moves, almost all rejected.
Efficiency (ESS per evaluation) peaks in between, near a = 0.234.

This script reproduces the story empirically on a d = 10 standard Gaussian:
sweep the step size, and show that ESS-per-sample is maximized where the
acceptance rate is ~0.2-0.3, not near 0 or 1. The exact optimum drifts a little
at finite d (the 0.234 result is a d -> infinity limit), which we say honestly.

Run:  python experiments/optimal_scaling.py
"""

import numpy as np

from common import plt, print_table, savefig
from mcmc.diagnostics import ess
from mcmc.metropolis import random_walk_metropolis
from mcmc.targets import Gaussian

SEED = 20260704
DIM = 10
N_CHAINS = 6
N_SAMPLES = 40_000


def sweep():
    target = Gaussian(np.zeros(DIM), np.eye(DIM))
    rng = np.random.default_rng(SEED)
    x0 = rng.standard_normal((N_CHAINS, DIM)) * 2.0

    # step sizes bracketing the optimum, quoted as l = sigma * sqrt(d) so the
    # numbers are comparable to the theory's O(1) scaling parameter.
    ls = np.array([0.3, 0.6, 1.0, 1.5, 2.0, 2.4, 3.0, 4.0, 6.0, 9.0])
    rows = []
    for l in ls:
        step = l / np.sqrt(DIM)
        res = random_walk_metropolis(
            target, x0.copy(), n_samples=N_SAMPLES, step_size=step, rng=rng,
            n_warmup=4_000,
        )
        accept = float(res.accept_rate.mean())
        # ESS per coordinate, averaged over dimensions, normalized per draw
        ess_per_dim = np.array([ess(res.samples[:, :, i]) for i in range(DIM)])
        ess_frac = float(ess_per_dim.mean() / (res.n_samples * N_CHAINS))
        rows.append({"l = sigma*sqrt(d)": float(l), "step size": step,
                     "accept": accept, "ESS/draw": ess_frac})
    return rows


def main():
    print("=" * 60)
    print(f"Optimal scaling of RWMH on a d={DIM} standard Gaussian")
    print("=" * 60)
    rows = sweep()
    print_table(rows, list(rows[0].keys()))

    best = max(rows, key=lambda r: r["ESS/draw"])
    print(f"peak efficiency at l={best['l = sigma*sqrt(d)']:.1f} "
          f"(accept {best['accept']:.3f}); theory predicts accept -> 0.234 "
          f"as d -> infinity.")

    accepts = [r["accept"] for r in rows]
    effs = [r["ESS/draw"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    ax.plot(accepts, effs, "o-", color="C0")
    for r in rows:
        ax.annotate(f"{r['l = sigma*sqrt(d)']:.1f}",
                    (r["accept"], r["ESS/draw"]), fontsize=6,
                    textcoords="offset points", xytext=(3, 4), color="gray")
    ax.axvline(0.234, color="C3", ls="--", lw=1.2, label="0.234 (theory)")
    ax.set_xlabel("mean acceptance rate")
    ax.set_ylabel("ESS per draw (efficiency)")
    ax.set_title("RWMH efficiency peaks near a = 0.234", loc="left")
    ax.legend()
    savefig(fig, "optimal_scaling.png")


if __name__ == "__main__":
    main()
