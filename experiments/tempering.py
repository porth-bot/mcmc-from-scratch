"""Parallel tempering on a well-separated bimodal target.

Shows the failure a single chain cannot escape and how replica exchange fixes
it: two Gaussians 12 units apart (weights 0.35 / 0.65). Started entirely in
the left mode, a plain random walk never crosses the barrier; parallel
tempering, ferrying mobility down a temperature ladder, recovers both modes
and their weights.

Run:  python experiments/tempering.py
"""

import numpy as np
from common import plt, savefig

from mcmc.metropolis import random_walk_metropolis
from mcmc.targets import GaussianMixture
from mcmc.tempering import geometric_ladder, parallel_tempering

TARGET = GaussianMixture(
    weights=[0.35, 0.65], means=[[-6.0, 0.0], [6.0, 0.0]], covs=[np.eye(2), np.eye(2)]
)


def run():
    rng = np.random.default_rng(0)
    K = 8
    betas = geometric_ladder(K, beta_min=0.01)
    x0 = np.tile([-6.0, 0.0], (K, 1)) + 0.3 * rng.standard_normal((K, 2))

    pt = parallel_tempering(
        TARGET, x0, n_samples=20_000, step_sizes=1.2 / np.sqrt(betas), betas=betas,
        rng=rng, n_warmup=5_000,
    )
    rw = random_walk_metropolis(
        TARGET, x0[:1], n_samples=20_000, step_size=1.2,
        rng=np.random.default_rng(1), n_warmup=5_000,
    )
    return betas, pt, rw


def main():
    betas, pt, rw = run()
    pt_x, rw_x = pt.samples[0], rw.samples[0]

    def summarize(name, x):
        print(f"{name:20s} mean {x.mean(axis=0).round(2)}  "
              f"left-mode frac {float((x[:, 0] < 0).mean()):.3f}")

    print(f"true mean {TARGET.mean().round(2)}  true left-mode frac 0.35")
    summarize("parallel tempering", pt_x)
    summarize("single random walk", rw_x)
    print("swap rates:", pt.extras["swap_rates"].round(2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True, sharex=True, sharey=True)
    for ax, title, x in [
        (axes[0], "Single random walk (trapped)", rw_x),
        (axes[1], "Parallel tempering (both modes)", pt_x),
    ]:
        ax.hexbin(x[:, 0], x[:, 1], gridsize=40, extent=(-9, 9, -4, 4), cmap="Blues", mincnt=1)
        for m in TARGET.means:
            ax.plot(m[0], m[1], "rx", ms=9, mew=2)
        ax.set_title(title)
        ax.set_xlabel("$x_0$")
    axes[0].set_ylabel("$x_1$")
    fig.suptitle(
        "Bimodal target, both samplers started in the left mode. "
        "The random walk never crosses; tempering recovers both modes "
        "(red × = true means).", fontsize=9,
    )
    savefig(fig, "tempering_bimodal.png")


if __name__ == "__main__":
    main()
