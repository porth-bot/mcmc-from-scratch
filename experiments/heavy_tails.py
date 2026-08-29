"""Heavy tails, taken apart: three failures, and the one diagnostic that sees
only the third.

``mcmc/targets.py`` has shipped a Student-t since the beginning, described as
"the standard heavy-tail mixing cautionary target", and no result in this repo
had ever used it. The reason to run it now is that "MCMC mixes badly in heavy
tails" runs three separate things together, and the target's exact sampler
makes them separable: an i.i.d. arm is a sampler with no autocorrelation at
all, so anything that goes wrong there is not the sampler's fault.

Three arms (exact i.i.d., random-walk Metropolis, HMC) on a 1-D Student-t at
six degrees of freedom, and two estimands chosen so that one has a CLT at every
dof and the other does not:

* **the mean**, whose CLT needs a finite variance (dof > 2);
* **P(|X| <= 1)**, a bounded functional, whose CLT holds always, and whose
  exact value comes from quadrature (``tails.interval_probability``).

What the arms are for. Anything the i.i.d. arm also suffers is a fact about the
estimand; anything only the samplers suffer is a fact about the sampler. That
split is the whole design, and it moves the headline: two of the three failures
below are not sampler failures, and the one diagnostic in wide use sees only
the one that is.

Predictions, derived in ``mcmc/tails.py`` and scored here rather than asserted:
the mean's error falls like n^(1/dof - 1) below dof = 2 and not at all at
dof = 1; the plug-in sd *grows* like n^(1/dof - 1/2) there.

Run:  python experiments/heavy_tails.py   (~90 s)
"""

import numpy as np
from common import plt, print_table, savefig

from mcmc.diagnostics import ess
from mcmc.hmc import hmc
from mcmc.metropolis import random_walk_metropolis
from mcmc.tails import (
    coverage,
    interval_probability,
    log_log_slope,
    mean_error_exponent,
    plug_in_interval,
    sample_sd_exponent,
)
from mcmc.targets import StudentT

SEED = 20260829
DOFS = (1.0, 1.25, 1.5, 2.5, 5.0, 30.0)
NS = (250, 1000, 4000, 16000)
N_REPLICATES = 400          # independent estimators, for rates and coverage
BOUND = 1.0                 # the bounded functional is P(|X| <= BOUND)

# One (dof, n) cell for the sampler arms; the rate sweep is i.i.d.-only,
# because 400 replicates x 4 lengths x 6 dofs of MCMC buys nothing the i.i.d.
# arm does not already establish about the *estimand*.
MCMC_N = 4000
MCMC_CHAINS = 256
RWM_STEP = 2.5
HMC_STEP, HMC_LEAPFROG = 0.5, 8


def target(dof):
    return StudentT([0.0], [[1.0]], dof)


def iid_draws(dof, n, replicates, rng):
    """(replicates, n) exact draws -- a sampler with zero autocorrelation."""
    return target(dof).sample(n * replicates, rng).reshape(replicates, n)


def rate_sweep(rng):
    """How fast each estimand converges, with perfect draws.

    Errors are summarized by their *median* across replicates, not their rms:
    for dof <= 2 the rms error of the sample mean is itself infinite, so the
    obvious summary of the error does not exist either. That is worth stating
    rather than working around silently.
    """
    rows = []
    for dof in DOFS:
        truth_p = interval_probability(dof, -BOUND, BOUND)
        mean_err, prob_err, sds = [], [], []
        for n in NS:
            x = iid_draws(dof, n, N_REPLICATES, rng)
            mean_err.append(np.median(np.abs(x.mean(axis=1))))
            p_hat = np.mean(np.abs(x) <= BOUND, axis=1)
            prob_err.append(np.median(np.abs(p_hat - truth_p)))
            sds.append(np.median(x.std(axis=1, ddof=1)))
        rows.append({
            "dof": dof,
            "mean rate": log_log_slope(NS, mean_err),
            "predicted": mean_error_exponent(dof),
            "P(|X|<=1) rate": log_log_slope(NS, prob_err),
            "sd rate": log_log_slope(NS, sds),
            "sd predicted": sample_sd_exponent(dof),
        })
    return rows


def coverage_sweep(rng):
    """Does the plug-in interval still cover when its variance is infinite?"""
    rows = []
    for dof in DOFS:
        row = {"dof": dof}
        for n in (NS[0], NS[-1]):
            x = iid_draws(dof, n, N_REPLICATES * 4, rng)
            centre, half = plug_in_interval(x)
            row[f"cover n={n}"] = coverage(centre, half, 0.0)
            row[f"width n={n}"] = float(np.median(half))
        rows.append(row)
    return rows


def sampler_arms(rng):
    """What the samplers add on top of the estimand's own difficulty."""
    rows = []
    for dof in DOFS:
        t = target(dof)
        truth_p = interval_probability(dof, -BOUND, BOUND)
        x0 = rng.standard_normal((MCMC_CHAINS, 1))

        arms = {"iid": iid_draws(dof, MCMC_N, MCMC_CHAINS, rng)}
        r = random_walk_metropolis(t, x0, n_samples=MCMC_N, step_size=RWM_STEP,
                                   rng=rng, n_warmup=1000)
        arms["rwm"] = r.samples[:, :, 0]
        h = hmc(t, x0, n_samples=MCMC_N, step_size=HMC_STEP,
                n_leapfrog=HMC_LEAPFROG, rng=rng, n_warmup=1000,
                adapt_step_size=True)
        arms["hmc"] = h.samples[:, :, 0]

        for name, s in arms.items():
            n_eff = ess(s)
            centre, half = plug_in_interval(s, effective_n=n_eff / MCMC_CHAINS)
            rows.append({
                "dof": dof,
                "arm": name,
                # ESS per draw: 1.0 is the i.i.d. ceiling.
                "ess/draw": n_eff / s.size,
                # How far into the tail the arm ever got, against the exact
                # draws' reach at the same budget. Reported as an integer: it
                # is a single extreme draw, and printing three decimals of it
                # once made two unrelated cells look like the same number.
                "max|x|": int(round(float(np.abs(s).max()))),
                "P(|X|<=1) err": float(abs(np.mean(np.abs(s) <= BOUND) - truth_p)),
                "cover": coverage(centre, half, 0.0),
            })
    return rows


def figure(rates, covers, arms):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    dofs = [r["dof"] for r in rates]

    ax = axes[0]
    ax.plot(dofs, [r["predicted"] for r in rates], "k--", lw=1.2,
            label="generalized CLT")
    ax.plot(dofs, [r["mean rate"] for r in rates], "o-", color="C3",
            label="measured, the mean")
    ax.plot(dofs, [r["P(|X|<=1) rate"] for r in rates], "s-", color="C0",
            label=r"measured, $P(|X|\leq 1)$")
    ax.axhline(-0.5, color="gray", ls=":", lw=1)
    ax.axvline(2.0, color="gray", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("degrees of freedom (log)")
    ax.set_ylabel("fitted exponent of error vs n")
    ax.set_title("Exact draws: the mean's rate decays\nbelow dof 2; a bounded "
                 "functional's does not", loc="left")
    ax.legend(fontsize=7.5, loc="lower right")

    # Coverage and width together, because the panel's point is that one holds
    # while the other does not -- a coverage-only panel would not show it.
    ax = axes[1]
    n_lo, n_hi = NS[0], NS[-1]
    ax.plot(dofs, [c[f"cover n={n_lo}"] for c in covers], "o-", color="C2",
            label=f"coverage, n = {n_lo:,}")
    ax.plot(dofs, [c[f"cover n={n_hi}"] for c in covers], "s-", color="C4",
            label=f"coverage, n = {n_hi:,}")
    ax.axhline(0.95, color="k", ls="--", lw=1, label="nominal 0.95")
    ax.set_xscale("log")
    ax.set_ylim(0.85, 1.02)
    ax.set_xlabel("degrees of freedom (log)")
    ax.set_ylabel(r"coverage of mean $\pm 1.96\,s/\sqrt{n}$")
    ax.set_title("The interval keeps covering, Cauchy included.\n"
                 "It is the width that stops shrinking", loc="left")

    ax2 = ax.twinx()
    shrink = [c[f"width n={n_lo}"] / c[f"width n={n_hi}"] for c in covers]
    ax2.plot(dofs, shrink, "d--", color="C1", label="width shrinkage")
    ax2.axhline((n_hi / n_lo) ** 0.5, color="C1", ls=":", lw=1,
                label=r"$\sqrt{n}$ ideal")
    ax2.set_yscale("log")
    ax2.set_ylabel(f"width(n={n_lo:,}) / width(n={n_hi:,})", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax2.spines["top"].set_visible(False)

    handles = ax.get_legend_handles_labels()
    extra = ax2.get_legend_handles_labels()
    ax.legend(handles[0] + extra[0], handles[1] + extra[1], fontsize=7,
              loc="center right")

    ax = axes[2]
    for name, colour, marker in (("iid", "C7", "^"), ("rwm", "C3", "o"),
                                 ("hmc", "C0", "s")):
        sel = [a for a in arms if a["arm"] == name]
        ax.plot([a["dof"] for a in sel], [a["ess/draw"] for a in sel],
                marker=marker, ls="-", color=colour, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degrees of freedom (log)")
    ax.set_ylabel("ESS per draw (log)")
    ax.set_title("What ESS does see: the samplers' own failure.\n"
                 "It reads ~1 for i.i.d. at every dof", loc="left")
    ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle("Heavy tails: the estimand's failure, the interval's, and the "
                 "sampler's", y=1.10)
    savefig(fig, "heavy_tails.png")


def main():
    rng = np.random.default_rng(SEED)

    print("\n=== Exact i.i.d. draws: how fast does each estimand converge? ===")
    print("(fitted exponent of median error vs n; rms is undefined at dof <= 2)")
    rates = rate_sweep(rng)
    print_table(rates, ["dof", "mean rate", "predicted", "P(|X|<=1) rate",
                        "sd rate", "sd predicted"])

    print("\n=== The plug-in interval, exact draws, nominal coverage 0.95 ===")
    covers = coverage_sweep(rng)
    print_table(covers, ["dof"] + [f"{k} n={n}" for n in (NS[0], NS[-1])
                                   for k in ("cover", "width")])

    print(f"\n=== The samplers, at n = {MCMC_N:,} x {MCMC_CHAINS} chains ===")
    print(f"(coverage here is over {MCMC_CHAINS} chains as replicates, so it "
          "carries about +/-0.014;")
    print(" the converged coverage numbers are the i.i.d. table above)")
    arms = sampler_arms(rng)
    print_table(arms, ["dof", "arm", "ess/draw", "max|x|", "P(|X|<=1) err",
                       "cover"])

    figure(rates, covers, arms)


if __name__ == "__main__":
    main()
