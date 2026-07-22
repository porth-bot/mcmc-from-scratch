"""Vectorized chains: how flat is wall-clock per step as the chain count grows?

Every sampler in this repo advances *all* its chains in lockstep as one batched
NumPy computation (``mcmc/base.py``): a proposal is ``x + step * rng.standard_normal((n_chains, dim))``,
the accept test is a vectorized comparison, and HMC's leapfrog integrates the
whole ``(n_chains, dim)`` block at once. The design claim is that running 1 chain
or 64 chains costs *nearly the same wall-clock per iteration*, because the work
is one set of BLAS calls whose shapes only grow in the batch dimension -- so the
extra chains are close to free until they saturate memory bandwidth.

This makes that claim quantitative. On a fixed correlated-Gaussian target we time
HMC (fixed step size and leapfrog length, so the per-chain arithmetic is
identical) across ``n_chains`` in {1, 2, 4, 8, 16, 32, 64} and report:

- **wall-clock per iteration** -- flat in the vectorized regime, rising once the
  batch stops fitting the cache/bandwidth budget;
- **throughput** -- effective samples (summed ESS across chains) per second,
  which climbs nearly linearly with the chain count exactly because the per-
  iteration cost stays almost flat.

Timing is machine-dependent, so the committed figure is illustrative and the
correctness of the batching (identical statistics at any chain count) is what the
test pins. Run:  python experiments/vectorized_scaling.py
"""

from __future__ import annotations

import time

import numpy as np

from mcmc.diagnostics import ess
from mcmc.hmc import hmc
from mcmc.targets import Gaussian

from common import print_table, savefig

CHAIN_COUNTS = [1, 2, 4, 8, 16, 32, 64]
DIM = 10
N_SAMPLES = 400
N_WARMUP = 200
N_LEAPFROG = 20
STEP_SIZE = 0.25
REPEATS = 5


def correlated_gaussian(dim: int) -> Gaussian:
    """A moderately correlated Gaussian: unit variances, 0.6 pairwise correlation."""
    cov = 0.6 * np.ones((dim, dim)) + 0.4 * np.eye(dim)
    return Gaussian(mean=np.zeros(dim), cov=cov)


def run_once(target: Gaussian, n_chains: int, seed: int) -> "object":
    """One fixed-length HMC run at a given chain count (no adaptation, for timing)."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal((n_chains, target.dim))
    return hmc(
        target, x0, n_samples=N_SAMPLES, step_size=STEP_SIZE,
        n_leapfrog=N_LEAPFROG, rng=rng, n_warmup=N_WARMUP,
    )


def _time_per_iter(target: Gaussian, n_chains: int) -> tuple[float, object]:
    """Median wall-clock per iteration (ms) over REPEATS, plus one result to score."""
    times = []
    result = None
    for r in range(REPEATS):
        t0 = time.perf_counter()
        result = run_once(target, n_chains, seed=r)
        times.append(time.perf_counter() - t0)
    total_iters = N_SAMPLES + N_WARMUP
    per_iter_ms = 1e3 * float(np.median(times)) / total_iters
    return per_iter_ms, result


def measure() -> list[dict]:
    target = correlated_gaussian(DIM)
    # Warm up NumPy/BLAS so the first timed point is not penalized by lazy init.
    run_once(target, 8, seed=999)

    rows = []
    for n_chains in CHAIN_COUNTS:
        per_iter_ms, result = _time_per_iter(target, n_chains)
        # Effective samples across all chains for coordinate 0.
        x0 = result.samples[:, :, 0]                     # (n_chains, n_samples)
        total_ess = ess(x0)
        wall_s = per_iter_ms * (N_SAMPLES + N_WARMUP) / 1e3
        rows.append({
            "n_chains": n_chains,
            "ms_per_iter": per_iter_ms,
            "ms_per_iter_per_chain": per_iter_ms / n_chains,
            "ess_total": total_ess,
            "ess_per_sec": total_ess / wall_s,
        })
    return rows


def figure(rows: list[dict]) -> None:
    from common import plt  # matplotlib configured in common

    ns = [r["n_chains"] for r in rows]
    per_iter = [r["ms_per_iter"] for r in rows]
    ess_sec = [r["ess_per_sec"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), constrained_layout=True)

    ax = axes[0]
    ax.plot(ns, per_iter, "o-", color="C0")
    base = per_iter[0]
    ax.axhline(base, color="0.7", ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_xlabel("number of chains")
    ax.set_ylabel("wall-clock per iteration (ms)")
    ax.set_ylim(bottom=0)
    ax.set_title("Per-iteration cost stays near flat:\n64 chains for roughly the "
                 "price of 1", loc="left", fontsize=9)

    ax = axes[1]
    ax.plot(ns, ess_sec, "o-", color="C3", label="measured")
    ideal = [ess_sec[0] * n for n in ns]
    ax.plot(ns, ideal, ls="--", color="0.6", lw=1, label="linear in chains")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_xlabel("number of chains")
    ax.set_ylabel("effective samples / second")
    ax.set_title("Throughput climbs almost linearly\n(the batched design paying "
                 "off)", loc="left", fontsize=9)
    ax.legend(fontsize=8)

    fig.suptitle("Batched chains: extra chains are nearly free until bandwidth "
                 "saturates", y=1.06)
    savefig(fig, "vectorized_scaling.png")


def main() -> None:
    rows = measure()
    cols = ["n_chains", "ms_per_iter", "ms_per_iter_per_chain", "ess_total", "ess_per_sec"]
    print_table(rows, cols)
    speedup = rows[-1]["ms_per_iter_per_chain"] / rows[0]["ms_per_iter_per_chain"]
    print(f"\nper-chain cost at {CHAIN_COUNTS[-1]} chains vs 1 chain: "
          f"{speedup:.3f}x (smaller = better batching)")
    figure(rows)


if __name__ == "__main__":
    main()
