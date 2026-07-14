"""Thinning: the folklore, the closed form, and what it actually costs.

"MCMC draws are autocorrelated, so keep every k-th one and throw the rest
away." This is one of the most persistent pieces of MCMC folklore, and for
*accuracy* it is simply wrong: thinning always inflates the variance of your
estimates. Geyer (1992) said so plainly and it is still repeated.

The reason is not subtle. Autocorrelated draws are worth less than independent
ones -- that is what tau measures -- but they are not worth *nothing*, and
discarding them throws away the value they did have. The estimator you actually
care about is the sample mean, and the mean of the full chain has strictly lower
variance than the mean of any thinned subsample of it.

For AR(1) this is exact (derived in theory/derivations.md Sec. 6.3 and
implemented as ``diagnostics.thinning_variance_ratio``):

    R(rho, k) = Var(thinned mean) / Var(full mean)
              = k (1 + rho^k)(1 - rho) / [ (1 - rho^k)(1 + rho) ]   >= 1,

with equality only at k = 1. Two limits organize the whole picture:

    rho = 0      ->  R = k exactly.  Thinning independent draws wastes exactly
                     the fraction you discard. The pure-waste case.
    rho -> 1     ->  R -> 1.         Thinning a chain far stickier than the
                     thinning interval is nearly free -- the discarded draws
                     were near-duplicates. It still does not help.

So the cost of thinning is *largest exactly where people think it is most
justified* (a fast-mixing chain) and smallest where the chain is so sticky the
draws really were redundant. There is no regime where it improves accuracy.

This script measures all of that two ways:

  Part 1 -- AR(1), where the formula is exact. Brute-force the variance of the
    sample mean over 4000 independent replicate chains and compare to R.

  Part 2 -- a real sampler. Random-walk Metropolis on the correlated 2D
    Gaussian. An MH chain is *not* AR(1) -- rejections make it repeat states,
    so its autocorrelation is not a clean geometric rho^j -- and the honest
    question is whether the AR(1) formula still predicts the observed ESS loss.
    It does, to within a few percent, using the chain's measured lag-1
    correlation.

When *is* thinning legitimate? When the binding constraint is cost, not
accuracy: RAM or disk for a long high-dimensional run, or an expensive
per-draw post-processing step (say each retained draw seeds a downstream
simulation). Then you are buying a cheaper pipeline with a known, quantified
amount of precision -- and R is exactly the exchange rate. What you must not do
is thin believing it makes the answer *better*.

Run:  python experiments/thinning.py
"""

from __future__ import annotations

import numpy as np

from common import plt, print_table, savefig
from mcmc.diagnostics import (
    autocorrelation,
    ess,
    integrated_autocorr_time,
    thinning_variance_ratio,
)
from mcmc.metropolis import random_walk_metropolis
from mcmc.targets import Gaussian

K_VALUES = (1, 2, 5, 10, 20)


# ---------------------------------------------------------------------------
# Part 1: AR(1), where the formula is exact
# ---------------------------------------------------------------------------
def ar1(rho, m, n, rng):
    x = np.empty((m, n))
    x[:, 0] = rng.standard_normal(m)
    innov = np.sqrt(1 - rho ** 2) * rng.standard_normal((m, n))
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + innov[:, t]
    return x


def part1_ar1(rho=0.9, n_rep=4000, n=2000, seed=11):
    """Brute-force Var(mean) over replicate chains vs the closed form."""
    rng = np.random.default_rng(seed)
    x = ar1(rho, n_rep, n, rng)  # each row an independent chain
    var_full = float(np.var(x.mean(axis=1)))
    ess_full = ess(x)

    rows = []
    for k in K_VALUES:
        thinned = x[:, ::k]
        var_thin = float(np.var(thinned.mean(axis=1)))
        rows.append(
            {
                "k": k,
                "kept": thinned.shape[1],
                "ESS": round(ess(thinned)),
                "ESS/ESS_full": ess(thinned) / ess_full,
                "R predicted": thinning_variance_ratio(rho, k),
                "R measured": var_thin / var_full,
            }
        )
    print(f"\nAR(1), rho = {rho}: {n_rep} independent chains of {n} draws")
    print(f"  tau = (1+rho)/(1-rho) = {(1 + rho) / (1 - rho):.1f}, "
          f"full-chain ESS = {ess_full:.0f} of {n_rep * n} draws")
    print_table(
        rows, ["k", "kept", "ESS", "ESS/ESS_full", "R predicted", "R measured"]
    )
    return rows


# ---------------------------------------------------------------------------
# Part 2: a real MH chain, which is not AR(1)
# ---------------------------------------------------------------------------
def part2_rwmh(n_samples=200_000, n_chains=4, seed=3):
    """RWMH on the correlated Gaussian; does the AR(1) formula still predict?"""
    rng = np.random.default_rng(seed)
    target = Gaussian(mean=np.zeros(2), cov=np.array([[1.0, 0.9], [0.9, 1.0]]))
    x0 = rng.standard_normal((n_chains, 2)) * 2.0
    res = random_walk_metropolis(
        target, x0, n_samples=n_samples, step_size=0.6, rng=rng, n_warmup=2000
    )
    x = res.samples[:, :, 0]  # first coordinate

    # The chain's measured lag-1 correlation is what we feed the AR(1) formula.
    rho_hat = float(autocorrelation(x, max_lag=1)[1])
    ess_full = ess(x)
    tau = integrated_autocorr_time(x)

    print(f"\nRWMH on the correlated Gaussian ({n_chains} chains x {n_samples}):")
    print(f"  accept rate {res.accept_rate.mean():.3f}, tau = {tau:.1f}, "
          f"measured lag-1 rho = {rho_hat:.3f}, ESS = {ess_full:.0f}")
    print("  (an MH chain repeats states on rejection, so it is NOT AR(1) --")
    print("   the question is whether the AR(1) formula still predicts the loss)")

    rows = []
    for k in K_VALUES:
        thinned = x[:, ::k]
        ess_thin = ess(thinned)
        rows.append(
            {
                "k": k,
                "kept/chain": thinned.shape[1],
                "ESS": round(ess_thin),
                "R measured": ess_full / ess_thin,  # ESS ratio == variance ratio
                "R predicted": thinning_variance_ratio(rho_hat, k),
            }
        )
    print_table(rows, ["k", "kept/chain", "ESS", "R measured", "R predicted"])
    return rows, rho_hat


# ---------------------------------------------------------------------------
def figure(ar1_rows, rwmh_rows, rho_ar1, rho_rwmh):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    ks = np.arange(1, 21)
    for rho in (0.0, 0.5, 0.9, 0.99):
        ax.plot(
            ks,
            [thinning_variance_ratio(rho, int(k)) for k in ks],
            lw=1.3,
            label=f"rho = {rho}",
        )

    ax.plot(
        [r["k"] for r in ar1_rows],
        [r["R measured"] for r in ar1_rows],
        "ko", ms=6, label=f"AR(1) measured (rho={rho_ar1})",
    )
    ax.plot(
        [r["k"] for r in rwmh_rows],
        [r["R measured"] for r in rwmh_rows],
        "k^", ms=6, mfc="none", mew=1.4,
        label=f"RWMH measured (lag-1 rho={rho_rwmh:.2f})",
    )
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.text(11.5, 1.05, "no cost", fontsize=7, color="0.4")

    ax.set_xlabel("thinning interval k  (keep every k-th draw)")
    ax.set_ylabel("Var(thinned mean) / Var(full mean)")
    ax.set_title(
        "Thinning always costs.\n"
        "The cost is worst for a well-mixing chain (rho=0: exactly k)."
    )
    ax.set_ylim(0.8, 12)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    savefig(fig, "thinning.png")


def main():
    print("=" * 68)
    print("Thinning: what it costs (theory/derivations.md Sec. 6.3)")
    print("=" * 68)
    rho_ar1 = 0.9
    ar1_rows = part1_ar1(rho=rho_ar1)
    rwmh_rows, rho_rwmh = part2_rwmh()
    figure(ar1_rows, rwmh_rows, rho_ar1, rho_rwmh)


if __name__ == "__main__":
    main()
