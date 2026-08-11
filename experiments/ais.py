"""Annealed importance sampling, scored against normalizers known exactly.

Nothing else in this repo estimates a normalizing constant, so nothing else can
check this one. Every target here is therefore a normalized density multiplied
by a constant we choose, which makes log Z that constant *exactly* -- ground
truth to four decimal places, at any dimension, for free.

Four questions, in the order they matter. Two of the four answers came out
against the story this script was written to tell, and they are the reason it
is worth running.

1. **What does annealing buy, at matched cost?** Total work is
   T * n_transition * N target evaluations, so a longer ladder means fewer
   particles. Sweeping T at a fixed budget is a different question from the
   usual "AIS beats IS", and at d = 4 it has a different answer: **plain
   importance sampling wins**, by 2.6x, and no ladder length, step size, or
   number of transitions tried here closes the gap. Spending a fixed budget on
   particles beats spending it on temperatures whenever IS still has effective
   particles left to spend.

2. **When does that reverse?** In dimension, which is where the folklore is
   right and for the reason it gives: IS weight variance compounds per
   coordinate. The crossover here is between d = 4 and d = 8, and the sweep
   reports where it is rather than assuming it exists.

3. **How large is the log Z bias, and does the jackknife remove it?**
   E[w] = Z is exact; log of that mean is not, and the gap is downward. Both
   are measured over independent replicates rather than argued from Jensen.

4. **Where does it lie to you?** Not where this script first looked. Two modes
   12 apart, a barrier no random walk crosses -- and AIS gets the answer right
   anyway, because its independence comes from p_0 and not from the chain. The
   real failure needs a p_0 that never proposes into one mode, and then the
   estimate is wrong by exactly log(missed weight) while the ESS diagnostic
   reads a perfect 1.000.

Run:  python experiments/ais.py     (~60 s)
"""

import numpy as np
from common import plt, print_table, savefig

from mcmc.ais import annealed_importance_sampling, geometric_betas
from mcmc.targets import Gaussian, GaussianMixture

LOG_Z = 2.5             # the constant every target is multiplied by
BUDGET = 40_000         # T * n_transition * N, held fixed in question 1
N_REPLICATES = 40
STEP = 1.2


def proposal(dim, scale=3.0):
    """The broad start: a zero-mean Gaussian, wider than every target here."""
    return Gaussian(mean=np.zeros(dim), cov=scale**2 * np.eye(dim))


def scaled(target):
    """The unnormalized target f = exp(LOG_Z) * p, so log Z = LOG_Z exactly."""
    return lambda x: target.logpdf(x) + LOG_Z


def run_ais(target, dim, n_temps, n_particles, seed, n_transition=1,
            scale=3.0, step=STEP, p0=None):
    p0 = p0 or proposal(dim, scale)
    return annealed_importance_sampling(
        log_f0=p0.logpdf, sample_f0=p0.sample, log_fT=scaled(target),
        betas=geometric_betas(n_temps), n_particles=n_particles,
        rng=np.random.default_rng(seed), step_size=step,
        n_transition=n_transition,
    )


# -- 1. the ladder, at matched cost -------------------------------------------


def ladder_sweep(dim, step=STEP, n_transition=1):
    """T from 1 (plain importance sampling) up, with N cut to match."""
    target = Gaussian(mean=np.full(dim, 1.5), cov=0.4 * np.eye(dim))
    rows = []
    for n_temps in (1, 2, 5, 10, 25, 50, 100, 250, 500):
        n_particles = max(BUDGET // (n_temps * n_transition), 20)
        runs = [run_ais(target, dim, n_temps, n_particles, seed=s, step=step,
                        n_transition=n_transition)
                for s in range(N_REPLICATES)]
        est = np.array([r.log_z for r in runs])
        rows.append({
            "T": n_temps, "N": n_particles,
            "bias": float(est.mean() - LOG_Z),
            "rmse": float(np.sqrt(np.mean((est - LOG_Z) ** 2))),
            "sd": float(est.std(ddof=1)),
            "ESS frac": float(np.mean([r.ess_fraction for r in runs])),
            "eff. particles": float(np.mean([r.ess for r in runs])),
        })
    print(f"\n1{'ab'[dim > 4]}. ladder length at a FIXED budget of "
          f"{BUDGET:,} target evaluations (d = {dim}, true log Z = {LOG_Z})")
    print("    T = 1 is plain importance sampling: no annealing, all particles.")
    print_table(rows, ["T", "N", "bias", "rmse", "sd", "ESS frac",
                       "eff. particles"])
    best = min(rows, key=lambda r: r["rmse"])
    verdict = ("annealing does not pay here" if best["T"] == 1
               else f"annealing pays: {rows[0]['rmse'] / best['rmse']:.1f}x "
                    "better than plain IS")
    print(f"    best RMSE at T = {best['T']} ({best['rmse']:.4f}) vs "
          f"{rows[0]['rmse']:.4f} at T = 1  ->  {verdict}")
    print("    the column that explains it is the last one: error tracks "
          "effective\n    particles, and a ladder buys ESS *fraction* by "
          "spending the particles it\n    is a fraction of.")
    return rows


def tuning_check(dim):
    """Is the T = 1 win at low d just an untuned AIS? Vary the knobs and see."""
    target = Gaussian(mean=np.full(dim, 1.5), cov=0.4 * np.eye(dim))
    rows = []
    for n_temps, k, step in ((1, 1, 1.2), (100, 1, 0.3), (100, 1, 0.6),
                             (100, 1, 1.2), (100, 1, 2.5), (20, 5, 1.2),
                             (100, 5, 1.2), (500, 1, 0.6)):
        n_particles = max(BUDGET // (n_temps * k), 20)
        est = np.array([
            run_ais(target, dim, n_temps, n_particles, seed=s, step=step,
                    n_transition=k).log_z for s in range(N_REPLICATES)
        ])
        rows.append({"T": n_temps, "steps/rung": k, "step size": step,
                     "N": n_particles,
                     "rmse": float(np.sqrt(np.mean((est - LOG_Z) ** 2)))})
    print(f"\n1c. the same budget spent every way I could think of (d = {dim})")
    print_table(rows, ["T", "steps/rung", "step size", "N", "rmse"])
    best_ais = min((r for r in rows if r["T"] > 1), key=lambda r: r["rmse"])
    print(f"    best annealed setting: {best_ais['rmse']:.4f}; plain IS: "
          f"{rows[0]['rmse']:.4f}")
    return rows


# -- 2. dimension --------------------------------------------------------------


def dimension_sweep():
    """The same mismatch per coordinate, in more coordinates."""
    rows = []
    for dim in (1, 2, 4, 8, 16):
        target = Gaussian(mean=np.full(dim, 0.5), cov=0.4 * np.eye(dim))
        entry = {"d": dim}
        for label, n_temps in (("IS (T=1)", 1), ("AIS (T=200)", 200)):
            n_particles = BUDGET // n_temps
            est = np.array([
                run_ais(target, dim, n_temps, n_particles, seed=s).log_z
                for s in range(N_REPLICATES)
            ])
            ess = run_ais(target, dim, n_temps, n_particles, seed=0).ess_fraction
            entry[f"{label} err"] = float(np.sqrt(np.mean((est - LOG_Z) ** 2)))
            entry[f"{label} ESS"] = float(ess)
        rows.append(entry)
    print("\n2. the same per-coordinate mismatch, in more coordinates "
          "(matched budget)")
    print_table(rows, ["d", "IS (T=1) err", "IS (T=1) ESS",
                       "AIS (T=200) err", "AIS (T=200) ESS"])
    return rows


# -- 3. the log Z bias, and the jackknife -------------------------------------


def bias_study():
    """Bias against N at a deliberately hopeless ladder (T = 1)."""
    dim = 4
    target = Gaussian(mean=np.full(dim, 2.0), cov=0.25 * np.eye(dim))
    rows = []
    for n_particles in (50, 100, 200, 400, 800, 1600):
        raw, jack, corr = [], [], []
        for s in range(N_REPLICATES * 3):
            res = run_ais(target, dim, 1, n_particles, seed=s)
            j, c = res.log_z_jackknife()
            raw.append(res.log_z)
            jack.append(j)
            corr.append(c)
        raw, jack = np.array(raw), np.array(jack)
        rows.append({
            "N": n_particles,
            "bias": float(raw.mean() - LOG_Z),
            "se": float(raw.std(ddof=1) / np.sqrt(len(raw))),
            "N*bias": float(n_particles * (raw.mean() - LOG_Z)),
            "jack bias": float(jack.mean() - LOG_Z),
            "correction": float(np.mean(corr)),
        })
    print(f"\n3. the log Z bias at T = 1, against N (d = {dim}, "
          f"{N_REPLICATES * 3} replicates per row)")
    print("   E[w] = Z is exact; log of the mean is not, and Jensen fixes the "
          "sign.\n   If the bias is O(1/N), N*bias is the column that stops "
          "moving.")
    print_table(rows, ["N", "bias", "se", "N*bias", "jack bias", "correction"])
    return rows


# -- 4. where it lies ----------------------------------------------------------


def separated_modes():
    """Two modes 12 apart, and three starting distributions.

    The barrier is the one Sec. 4 built parallel tempering for: no random walk
    at this step size crosses it. The question is whether that breaks AIS, and
    the answer is no -- AIS draws independently from p_0 every run, so as long
    as p_0 *proposes* into both modes the weights fix the balance without any
    chain ever crossing. What breaks it is a p_0 that misses a mode, and then
    the error is not noise: it is exactly log of the weight that was missed.
    """
    dim = 2
    weights = np.array([0.35, 0.65])
    target = GaussianMixture(weights=weights, means=[[-6.0, 0.0], [6.0, 0.0]],
                             covs=np.eye(dim))
    starts = (
        ("broad, covers both", Gaussian([0.0, 0.0], 9.0 * np.eye(dim)), None),
        ("narrow on left mode", Gaussian([-6.0, 0.0], np.eye(dim)),
         float(np.log(weights[0]))),
        ("narrow on right mode", Gaussian([6.0, 0.0], np.eye(dim)),
         float(np.log(weights[1]))),
    )
    rows = []
    for name, p0, predicted in starts:
        for n_temps in (1, 200):
            n_particles = max(BUDGET // n_temps, 40)
            runs = [run_ais(target, dim, n_temps, n_particles, seed=s, p0=p0)
                    for s in range(12)]
            est = np.array([r.log_z for r in runs])
            share = []
            for r in runs:
                x = r.extras["particles"]
                w = np.exp(r.log_weights - r.log_weights.max())
                share.append(float(np.sum(w[x[:, 0] > 0.0]) / np.sum(w)))
            rows.append({
                "start": name, "T": n_temps,
                "log Z": float(est.mean()),
                "err": float(est.mean() - LOG_Z),
                "predicted err": "--" if predicted is None else f"{predicted:.3f}",
                "ESS frac": float(np.mean([r.ess_fraction for r in runs])),
                "right-mode wt": float(np.mean(share)),
            })
    print("\n4. two modes 12 apart, weights 0.35 / 0.65 -- the target Sec. 4 "
          "built tempering for.")
    print("   No transition here ever crosses the barrier. 'right-mode wt' is "
          "the weighted\n   share of particles ending at x > 0, which is 0.65 "
          "if the weights are right;\n   'predicted err' is log of the weight a "
          "one-sided start cannot see.")
    print_table(rows, ["start", "T", "log Z", "err", "predicted err",
                       "ESS frac", "right-mode wt"])
    print("   The two failures are exact, not approximate -- and both report an "
          "ESS of\n   1.000, because the weights *within* the mode they can see "
          "are uniform. The\n   diagnostic is at its most reassuring precisely "
          "when the answer is wrong.")
    return rows


# -- figure --------------------------------------------------------------------


def figure(ladder4, ladder8, dims, bias, modes):
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.4), constrained_layout=True)

    ax = axes[0]
    for rows, dim, style in ((ladder4, 4, "o-"), (ladder8, 8, "s-")):
        ax.plot([r["T"] for r in rows], [r["rmse"] for r in rows], style, ms=4,
                label=f"$d = {dim}$")
        best = min(rows, key=lambda r: r["rmse"])
        ax.plot([best["T"]], [best["rmse"]], "*", ms=13, color="0.25",
                zorder=5)
    ax.set(xscale="log", yscale="log", xlabel="temperatures $T$ (budget fixed)",
           ylabel="RMSE of $\\log Z$ (nats)",
           title="1. Ladder length at fixed cost\n($T=1$ is plain IS; star = best)")
    ax.legend()

    ax = axes[1]
    d = [r["d"] for r in dims]
    ax.plot(d, [r["IS (T=1) err"] for r in dims], "o-", ms=4, label="IS (T=1)")
    ax.plot(d, [r["AIS (T=200) err"] for r in dims], "s-", ms=4,
            label="AIS (T=200)")
    ax.set(xscale="log", yscale="log", xlabel="dimension d",
           ylabel="RMSE of log Z (nats)", title="2. The same mismatch,\nin more coordinates")
    ax.legend()

    ax = axes[2]
    N = [r["N"] for r in bias]
    ax.errorbar(N, [-r["bias"] for r in bias], yerr=[r["se"] for r in bias],
                fmt="o-", ms=4, label="raw log Z")
    ax.errorbar(N, [-r["jack bias"] for r in bias],
                yerr=[r["se"] for r in bias], fmt="s-", ms=4,
                label="jackknifed")
    ref = bias[0]["N"] * -bias[0]["bias"]
    ax.plot(N, [ref / n for n in N], ":", color="0.5", lw=1, label="$1/N$")
    ax.axhline(0.0, color="0.8", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="particles N",
           ylabel="downward bias in log Z (nats)",
           title="3. Unbiased in $Z$,\nbiased low in $\\log Z$")
    ax.legend()

    ax = axes[3]
    shown = [r for r in modes if r["T"] == 200]
    x = np.arange(len(shown))
    ax.bar(x - 0.2, [r["right-mode wt"] for r in shown], 0.4,
           label="weight in right mode")
    ax.bar(x + 0.2, [r["ESS frac"] for r in shown], 0.4, label="ESS fraction")
    ax.axhline(0.65, color="0.3", ls="--", lw=1)
    ax.annotate("true weight 0.65", (len(shown) - 0.5, 0.66), fontsize=8,
                color="0.3", ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [r["start"].replace("narrow on ", "only\n").replace(
            "broad, covers both", "covers\nboth") + f"\nerr {r['err']:+.3f}"
         for r in shown], fontsize=8)
    ax.set(ylabel="fraction", ylim=(0, 1.35),
           title="4. What $p_0$ misses, $Z$ misses\n(error in $\\log Z$ under each)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    savefig(fig, "ais.png")


def main():
    ladder4 = ladder_sweep(dim=4)
    ladder8 = ladder_sweep(dim=8)
    tuning_check(dim=4)
    dims = dimension_sweep()
    bias = bias_study()
    modes = separated_modes()
    figure(ladder4, ladder8, dims, bias, modes)


if __name__ == "__main__":
    main()
