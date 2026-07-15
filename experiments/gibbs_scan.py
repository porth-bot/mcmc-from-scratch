"""Random-scan vs systematic-scan Gibbs on a correlated Gaussian.

Gibbs has a choice of sweep order that does not change *correctness* but does
change *mixing*:

- **Systematic** (deterministic) scan visits the blocks in a fixed order every
  sweep. Its kernel is the composition K_d ... K_1, which is not reversible
  (reversing the order gives a different -- also valid -- kernel), but leaves
  pi invariant.
- **Random** scan picks a block uniformly at random at each sub-step. Its
  kernel is the mixture (1/d) sum_i K_i, which *is* reversible when the K_i are.

Both are pi-invariant, so both are exact. The interesting question is which
mixes faster, and the textbook answer is "it depends on the target" (Roberts &
Sahu 1997). This script measures it on the one target where the systematic-scan
autocorrelation is known in closed form: the 2D Gaussian with correlation rho,
whose per-coordinate lag-1 autocorrelation under systematic scan is exactly
rho^2.

The comparison is made fair by counting *work*, not sweeps: one recorded sweep
does d block updates under either scan (random scan draws d indices with
replacement), so effective sample size per unit work is directly comparable.

What comes out (seed 0, 4 chains, 20k sweeps, warmup 1k):

    rho     systematic ESS     random ESS      systematic / random
    0.90        ~8500              ~4500              ~1.9
    0.95        ~4100              ~2100              ~1.9
    0.99        ~860               ~430               ~2.0

Systematic scan is about twice as efficient here, and the reason is concrete:
with replacement, a random sweep sometimes updates one coordinate twice and the
other zero times, leaving a coordinate *stale* for that sweep -- so the chain's
autocorrelation is a hair higher (lag-1 ~0.85 vs the systematic 0.81 at
rho=0.9) and, integrated, costs a factor of ~2. This is not a universal verdict
(random scan can win when a fixed order induces a bad drift, or when d is large
and a full sweep is wasteful), but on this canonical correlated target the
deterministic scan is the better default -- which is why it *is* the default in
``mcmc.gibbs``.

Run:  python experiments/gibbs_scan.py
"""

import numpy as np

from common import print_table, savefig
from mcmc.diagnostics import ess
from mcmc.gibbs import gibbs, make_gaussian_gibbs_updates


def measure(rho, n_samples=20_000, n_warmup=1_000, n_chains=4, seed=0):
    """ESS and lag-1 autocorrelation of coordinate 0 under each scan.

    Both scans share the same target, initial state, and seed so the only
    difference is the sweep order.
    """
    cov = np.array([[1.0, rho], [rho, 1.0]])
    updates = make_gaussian_gibbs_updates(np.zeros(2), cov)
    out = {"rho": rho}
    for scan in ("systematic", "random"):
        rng = np.random.default_rng(seed)
        res = gibbs(
            updates,
            {"x": np.zeros((n_chains, 2))},
            n_samples=n_samples,
            rng=rng,
            n_warmup=n_warmup,
            scan=scan,
        )
        x = res.samples[:, :, 0]  # (chains, samples) for coordinate 0
        out[f"{scan}_ess"] = ess(x)
        c = x - x.mean(axis=1, keepdims=True)
        out[f"{scan}_lag1"] = float(
            np.mean([(ci[:-1] @ ci[1:]) / (ci @ ci) for ci in c])
        )
    out["ratio"] = out["systematic_ess"] / out["random_ess"]
    return out


def figure(rows):
    import matplotlib.pyplot as plt

    rhos = [r["rho"] for r in rows]
    sys_ess = [r["systematic_ess"] for r in rows]
    rand_ess = [r["random_ess"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(rhos, sys_ess, "o-", label="systematic scan")
    axes[0].plot(rhos, rand_ess, "s-", label="random scan")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("correlation rho")
    axes[0].set_ylabel("ESS  (per 20k sweeps, 4 chains)")
    axes[0].set_title("mixing vs correlation")
    axes[0].legend()

    ratio = [r["ratio"] for r in rows]
    axes[1].plot(rhos, ratio, "o-", color="C2")
    axes[1].axhline(1.0, color="0.6", lw=0.8, ls="--")
    axes[1].set_xlabel("correlation rho")
    axes[1].set_ylabel("systematic ESS / random ESS")
    axes[1].set_title("systematic scan is ~2x more efficient here")
    axes[1].set_ylim(0, 2.4)
    fig.tight_layout()
    savefig(fig, "gibbs_scan.png")


def main():
    rows = [measure(rho) for rho in (0.9, 0.95, 0.99)]
    cols = ["rho", "systematic_ess", "random_ess", "ratio",
            "systematic_lag1", "random_lag1"]
    print_table(rows, cols)
    figure(rows)


if __name__ == "__main__":
    main()
