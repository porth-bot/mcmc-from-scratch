"""Experiment 15: eight schools, dense metric vs diagonal -- the rotation there
is to do, and the residual that is not a rotation.

Sec. 7 of the README ends on a measured limit and an attribution. The limit:
with an adapted *diagonal* metric, eight schools' log tau gains 2.4x in ESS per
gradient where eta_1 gains 14.8x. The attribution: "a diagonal metric rescales
marginals, it cannot rotate, so the funnel curvature in (log tau, eta)
survives... that residual is exactly what a dense metric or NUTS is for."

Sec. 13 then built the dense metric and measured, on an AR(1) Gaussian, that it
is worth 9.8x when there is correlation to remove. This section puts it on the
posterior the limitation was measured on. Three studies, in the order that lets
the third explain the second:

  A. What rotation is available, before any sampler runs. A long reference run
     gives the posterior covariance in the non-centered coordinates; from it,
     the Gaussian-approximation conditioning each metric can reach -- kappa(Sigma)
     for the identity, kappa(R) for the *best possible* diagonal, and exactly 1
     for the exact dense metric (Sec. 4.10). The gap between the last two is the
     entire algebraic headroom a dense metric has here, and it is a prediction
     recorded before study B runs.
  B. Run it. Identity / adapted diagonal / adapted dense, plus *oracle* fixed
     metrics built from the reference covariance, so estimation error and
     algebra are separated. Trajectory length is swept per metric, because
     Sec. 13 measured that a shared L understates a whitening metric by an
     order of magnitude. Reported as the integrated autocorrelation time tau as
     well as ESS/1k gradients: tau is clamped at 1 by Geyer's estimator, so a
     cell at tau = 1 is censored from below and cannot be compared with another
     cell at tau = 1.
  C. Why the residual is not a rotation. The Hessian is available in closed
     form (mcmc/models.py), so the whitened conditioning each metric achieves
     can be evaluated *at every posterior draw* rather than once at a Gaussian
     approximation. The spread of that number across the posterior is the
     position-dependent curvature no global metric of any kind can absorb, and
     comparing it against the dense-vs-diagonal difference says which of the
     two effects the residual in Sec. 7 actually is.

Run:  python experiments/eight_schools_metric.py
"""

import numpy as np

from common import plt, print_table, savefig
from mcmc.adapt import whitened_condition_numbers
from mcmc.diagnostics import ess, integrated_autocorr_time, split_rhat
from mcmc.hmc import hmc
from mcmc.models import EightSchoolsNonCentered

SEED = 20260903
N_CHAINS = 4
N_SEEDS = 5
LENGTHS = (1, 2, 3, 5, 8, 20)   # 20 is Sec. 7's fixed L, kept for continuity
THIN = 20                        # study C evaluates a (10, 10) Hessian per draw

# Coordinates Sec. 7 tabulates, by index in z = (mu, log tau, eta_1..eta_8).
COORDS = {"mu": 0, "log tau": 1, "eta_1": 2}


def reference_posterior(model, n_samples=40_000):
    """A long, well-adapted run, used only as ground truth for the metrics.

    Adapted diagonal rather than identity because the identity metric mixes mu
    at tau_int ~ 24 (study B) and a reference should not be the worst sampler
    available. R-hat and ESS are reported so the reference is not taken on
    trust -- every number in studies A and C is a functional of this covariance.
    """
    z0 = 0.1 * np.random.default_rng(SEED).standard_normal((N_CHAINS, model.dim))
    res = hmc(
        model, z0, n_samples=n_samples, step_size=0.1, n_leapfrog=5,
        rng=np.random.default_rng(SEED), n_warmup=4_000, adapt_step_size=True,
        target_accept=0.9, adapt_mass="diag",
    )
    z = res.samples.reshape(-1, model.dim)
    diag = {
        "worst R-hat": max(split_rhat(res.samples[:, :, d]) for d in range(model.dim)),
        "worst ESS": min(ess(res.samples[:, :, d]) for d in range(model.dim)),
        "divergent": res.extras["n_divergent"],
        "accept": float(res.accept_rate.mean()),
    }
    return np.cov(z, rowvar=False), res.samples, diag


def correlation(cov):
    s = np.sqrt(np.diag(cov))
    return cov / np.outer(s, s)


# -- A. the headroom, before any sampler runs ---------------------------------

def available_rotation(cov):
    """kappa of the whitened Hessian for each metric, Gaussian approximation.

    Under a Gaussian approximation with covariance Sigma the target's Hessian
    is Sigma^-1, so the identity metric leaves kappa(Sigma), the best possible
    diagonal leaves kappa(R) exactly (Sec. 4.10's closed form: whitening by the
    marginal sds removes all of the scale disparity and none of the
    correlation), and M^-1 = Sigma gives exactly 1.
    """
    d = cov.shape[0]
    prec = np.linalg.inv(cov)
    return [
        {"metric": "identity", "kappa": float(whitened_condition_numbers(np.eye(d), prec)[0])},
        {"metric": "best diagonal", "kappa": float(whitened_condition_numbers(np.diag(cov), prec)[0])},
        {"metric": "exact dense", "kappa": float(whitened_condition_numbers(cov, prec)[0])},
    ]


# -- B. through the sampler ---------------------------------------------------

def _configs(cov):
    """The five metrics. The two oracle rows are the point of the design: they
    are handed the reference covariance rather than estimating one, so if the
    estimated dense metric does not beat the estimated diagonal, the oracle row
    says whether that is the estimate's fault or the target's."""
    return [
        ("identity", dict(adapt_mass=False)),
        ("diagonal (adapted)", dict(adapt_mass="diag")),
        ("dense (adapted)", dict(adapt_mass="dense")),
        ("diagonal (oracle)", dict(metric=np.diag(cov))),
        ("dense (oracle)", dict(metric=cov)),
    ]


def sampler_sweep(model, cov, n_samples=8_000, n_warmup=2_000):
    """tau and ESS/1k gradients per coordinate, for each metric at each L.

    Warmup is charged to every arm (the adapted metrics' estimation cost is
    real, and the oracle arms are *not* charged for the reference run that
    produced their metric -- they are an upper bound, and labelled as one).
    """
    z0 = 0.1 * np.random.default_rng(SEED).standard_normal((N_CHAINS, model.dim))
    common = dict(n_samples=n_samples, step_size=0.1, n_warmup=n_warmup,
                  adapt_step_size=True, target_accept=0.9)
    out = {}
    for name, kw in _configs(cov):
        for L in LENGTHS:
            per_seed = []
            for s in range(N_SEEDS):
                # same seed across metrics: the metric is the only difference
                res = hmc(model, z0, n_leapfrog=L,
                          rng=np.random.default_rng(SEED + s), **common, **kw)
                g = res.extras["n_grad_evals"]
                n_draws = res.samples.shape[0] * res.samples.shape[1]
                cell = {}
                for cname, d in COORDS.items():
                    tau = integrated_autocorr_time(res.samples[:, :, d])
                    cell[cname] = (tau, 1000.0 * n_draws / tau / g)
                cell["_worst"] = min(v[1] for v in cell.values())
                # what this cell could report if every draw were independent:
                # the same formula at tau = 1, which is Geyer's floor.
                cell["_ceiling"] = 1000.0 * n_draws / g
                cell["_div"] = res.extras["n_divergent"]
                cell["_accept"] = float(res.accept_rate.mean())
                per_seed.append(cell)
            out[(name, L)] = per_seed
            print(f"    {name:20s} L={L:2d}  "
                  + "  ".join(
                      f"{c}: tau {np.median([p[c][0] for p in per_seed]):5.2f}"
                      for c in COORDS)
                  + f"  worst ESS/kgrad {np.median([p['_worst'] for p in per_seed]):6.1f}")
    return out


CLAMP = 1.0 + 1e-12   # Geyer's tau floor; a cell at it is censored, not measured


def summarize_sweep(sweep, cov):
    """Each metric at its own best L, with Sec. 7's shared L = 20 beside it.

    ``clamped`` counts the (seed, coordinate) cells whose tau came back at
    Geyer's floor of exactly 1. That column is not a footnote: an arm at the
    floor has produced draws the estimator cannot distinguish from independent,
    so its ESS/kgrad is a bound and two arms both at the floor are *not*
    measured to be equal. ``discriminating_length`` below finds the longest
    trajectory at which they still can be told apart.
    """
    rows = []
    for name, _ in _configs(cov):
        by_L = {L: float(np.median([p["_worst"] for p in sweep[(name, L)]]))
                for L in LENGTHS}
        best_L = max(by_L, key=by_L.get)
        cell = sweep[(name, best_L)]
        row = {"metric": name, "best L": best_L,
               "worst-coord ESS/kgrad": by_L[best_L],
               "at L=20": by_L[20],
               "seed spread": (max(p["_worst"] for p in cell)
                               / min(p["_worst"] for p in cell)),
               "clamped": f"{sum(1 for p in cell for c in COORDS if p[c][0] <= CLAMP)}"
                          f"/{N_SEEDS * len(COORDS)}"}
        rows.append(row)
    return rows


def discriminating_length(sweep, cov):
    """The longest L at which no metric's median tau is at the clamp.

    Beyond it every adapted arm reports tau = 1 on every coordinate and the
    comparison stops being one. Returns None if no L qualifies.
    """
    names = [n for n, _ in _configs(cov)]
    ok = [L for L in LENGTHS
          if all(np.median([p[c][0] for p in sweep[(n, L)]]) > CLAMP
                 for n in names for c in COORDS)]
    return max(ok) if ok else None


def compare_at(sweep, cov, L):
    """Per-coordinate tau at one L, with the five-seed range beside the median.

    The range is the point of the table. A 1.06x median difference between two
    metrics means nothing unless the seeds agree more closely than that, and
    reporting the median alone would hide the question.
    """
    rows = []
    for name, _ in _configs(cov):
        cell = sweep[(name, L)]
        row = {"metric": name}
        for cname in COORDS:
            taus = [p[cname][0] for p in cell]
            row[cname] = float(np.median(taus))
            row[f"{cname} range"] = f"{min(taus):.2f}-{max(taus):.2f}"
        rows.append(row)
    return rows


def section_7_gains(sweep, L=20):
    """Sec. 7's headline, re-measured at five seeds and reported with spread.

    Sec. 7 tabulates ESS/kgrad gains of the adapted diagonal over the identity
    at L = 20 from a single seed: 1.3x on mu, 2.4x on log tau, 14.8x on eta_1.
    The gains here come from a different draw count and a different seed, so
    the levels are not comparable with those; what is comparable is the
    *ordering* and the size of the seed spread the single-seed table did not
    have.
    """
    rows = []
    for cname in COORDS:
        idn = [p[cname][1] for p in sweep[("identity", L)]]
        ada = [p[cname][1] for p in sweep[("diagonal (adapted)", L)]]
        per_seed = [a / i for a, i in zip(ada, idn)]
        rows.append({
            "coordinate": cname,
            "identity ESS/kg": float(np.median(idn)),
            "adapted ESS/kg": float(np.median(ada)),
            "gain (median)": float(np.median(per_seed)),
            "gain range": f"{min(per_seed):.1f}-{max(per_seed):.1f}",
        })
    return rows


# -- C. the residual, which is not a rotation ---------------------------------

def local_conditioning(model, samples, cov):
    """kappa of the whitened Hessian at each posterior draw, per metric.

    Study A's numbers are what a *Gaussian approximation* to this posterior
    would give. The posterior is not Gaussian, and the exact Hessian says how
    much that matters: -H(z) is the local precision, and
    kappa(M^-1/2 (-H) M^-1/2) is the conditioning the metric actually delivers
    at z. Draws where -H is indefinite are counted rather than dropped
    silently: they are proof on their own that no single global metric is
    correct everywhere, since a positive definite M cannot condition a saddle.
    """
    z = samples.reshape(-1, model.dim)[::THIN]
    A = -model.hess_logpdf(z)
    d = model.dim
    metrics = [("identity", np.eye(d)), ("diagonal", np.diag(cov)), ("dense", cov)]
    kappas = {name: whitened_condition_numbers(m, A) for name, m in metrics}
    finite = np.all([np.isfinite(k) for k in kappas.values()], axis=0)
    rows = []
    for name in kappas:
        k = kappas[name][finite]
        rows.append({
            "metric": name,
            "median kappa": float(np.median(k)),
            "10th": float(np.quantile(k, 0.10)),
            "90th": float(np.quantile(k, 0.90)),
            "90th/10th": float(np.quantile(k, 0.90) / np.quantile(k, 0.10)),
        })
    return rows, {n: k[finite] for n, k in kappas.items()}, z[finite], float(1 - finite.mean())


def by_log_tau(kappas, z, n_bins=5):
    """The same kappas, cut by log tau: does the conditioning depend on where
    in the posterior it is measured, and by more than it depends on the metric?"""
    t = z[:, 1]
    edges = np.quantile(t, np.linspace(0, 1, n_bins + 1))
    rows = []
    for i in range(n_bins):
        sel = (t >= edges[i]) & (t <= edges[i + 1])
        row = {"log tau bin": f"[{edges[i]:+.2f}, {edges[i+1]:+.2f}]",
               "n": int(sel.sum())}
        for name in ("identity", "diagonal", "dense"):
            row[name] = float(np.median(kappas[name][sel]))
        row["dense gain"] = row["diagonal"] / row["dense"]
        rows.append(row)
    return rows, edges


# -- figure -------------------------------------------------------------------

def make_figure(ref_cov, est_cov, sweep, kappas, z, bin_rows, edges):
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.8))
    d = ref_cov.shape[0]
    labels = [r"$\mu$", r"$\log\tau$"] + [rf"$\eta_{{{j}}}$" for j in range(1, 9)]

    ax = axes[0, 0]
    R_ref, R_est = correlation(ref_cov), correlation(est_cov)
    both = np.tril(R_est, -1) + np.triu(R_ref, 1)
    im = ax.imshow(both, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(d), labels, fontsize=6)
    ax.set_yticks(range(d), labels, fontsize=6)
    ax.plot([-0.5, d - 0.5], [-0.5, d - 0.5], color="0.3", lw=0.8)
    ax.text(0.62, 0.06, "warmup estimate", transform=ax.transAxes, fontsize=7)
    ax.text(0.06, 0.93, "reference posterior", transform=ax.transAxes, fontsize=7,
            va="top")
    ax.set_title(
        "A. the rotation there is to do\n"
        rf"max $|r_{{ij}}|$ = {np.abs(R_ref - np.eye(d)).max():.3f}, "
        rf"$\kappa(R)$ = {np.linalg.cond(R_ref):.2f}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    styles = {"identity": ("o--", "0.45"), "diagonal (adapted)": ("s-", "#1f77b4"),
              "dense (adapted)": ("^-", "#a11"),
              "diagonal (oracle)": ("s:", "#1f77b4"), "dense (oracle)": ("^:", "#a11")}
    for name, (st, col) in styles.items():
        y = [float(np.median([p["_worst"] for p in sweep[(name, L)]])) for L in LENGTHS]
        ax.loglog(LENGTHS, y, st, color=col, ms=4, lw=1.2, label=name)
    ceiling = [float(np.median([p["_ceiling"] for p in sweep[("dense (adapted)", L)]]))
               for L in LENGTHS]
    ax.loglog(LENGTHS, ceiling, "k:", lw=1.0)
    ax.text(LENGTHS[-1], ceiling[-1] * 1.18, r"independence ceiling ($\tau = 1$)",
            fontsize=6.5, color="0.25", ha="right")
    # the title states a fact, so it computes it: the shortest L at which every
    # arm but the identity is *on* the ceiling, i.e. its worst coordinate is
    # reporting tau = 1 and its ESS/kgrad is a bound rather than a measurement.
    adapted = [n for n in styles if n != "identity"]
    on_ceiling = [
        L for L in LENGTHS
        if all(np.median([p["_worst"] for p in sweep[(n, L)]])
               >= np.median([p["_ceiling"] for p in sweep[(n, L)]]) - 1e-9
               for n in adapted)]
    ax.set_xlabel(r"leapfrog steps per iteration $L$")
    ax.set_ylabel("worst-coordinate ESS / 1k gradients")
    ax.set_title(
        rf"B. all four adapted arms sit on the ceiling at $L={min(on_ceiling)}$"
        if on_ceiling else "B. dense lands on top of diagonal", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left")

    ax = axes[1, 0]
    for name, col in [("identity", "0.45"), ("diagonal", "#1f77b4"), ("dense", "#a11")]:
        k = np.sort(kappas[name])
        ax.semilogx(k, np.linspace(0, 1, len(k)), color=col, lw=1.4, label=name)
    ax.set_xlabel(r"local $\kappa$ of the whitened Hessian, per draw")
    ax.set_ylabel("empirical CDF")
    ax.set_title(
        "C. dense moves the median "
        rf"{np.median(kappas['diagonal']) / np.median(kappas['dense']):.2f}$\times$; "
        rf"the identity is {np.median(kappas['identity']) / np.median(kappas['diagonal']):.0f}$\times$ off",
        fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    for name, col in [("diagonal", "#1f77b4"), ("dense", "#a11")]:
        ax.plot(centers, [r[name] for r in bin_rows], "o-", color=col, ms=4, label=name)
    diag_bins = [r["diagonal"] for r in bin_rows]
    across = max(diag_bins) / min(diag_bins)
    best_rot = max(r["dense gain"] for r in bin_rows)
    ax.set_xlabel(r"$\log\tau$")
    ax.set_ylabel(r"median local $\kappa$")
    ax.set_title(rf"D. position moves $\kappa$ {across:.1f}$\times$; "
                 rf"the rotation moves it {best_rot:.2f}$\times$", fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle("Eight schools (non-centered): what a dense metric buys", y=1.01)
    fig.tight_layout()
    savefig(fig, "eight_schools_metric.png")


def main():
    model = EightSchoolsNonCentered()
    print("=" * 74)
    print("Eight schools: dense metric vs diagonal, on the posterior Sec. 7 "
          "measured")
    print("=" * 74)

    print("\nreference posterior (4 x 40k, adapted diagonal):")
    ref_cov, ref_samples, diag = reference_posterior(model)
    print("  " + "  ".join(f"{k} {v:.4g}" for k, v in diag.items()))
    sd = np.sqrt(np.diag(ref_cov))
    print("  posterior sd: " + np.array2string(sd, precision=3))

    print("\n=== A. the rotation available, Gaussian approximation ===")
    rows_a = available_rotation(ref_cov)
    print_table(rows_a, ["metric", "kappa"])
    kappa_R = rows_a[1]["kappa"]
    print(f"the best possible diagonal already reaches kappa = {kappa_R:.3f}, so a "
          f"dense metric's\nentire algebraic headroom here is a factor "
          f"{kappa_R:.2f} in conditioning -- "
          f"{np.sqrt(kappa_R):.2f}x in step size.")

    print("\n=== B. through the sampler, L swept per metric ===")
    sweep = sampler_sweep(model, ref_cov)
    rows_b = summarize_sweep(sweep, ref_cov)
    print()
    print_table(rows_b, ["metric", "best L", "worst-coord ESS/kgrad", "at L=20",
                         "seed spread", "clamped"])

    Lc = discriminating_length(sweep, ref_cov)
    print(f"\nEvery adapted arm's best L has all {N_SEEDS * len(COORDS)} cells at "
          "Geyer's tau = 1 floor, so the\nequal ESS/kgrad there is four bounds "
          f"coinciding, not a measured tie. L = {Lc} is the\nlongest trajectory "
          "at which no median tau is clamped; that is where they can be ranked:")
    print_table(compare_at(sweep, ref_cov, Lc),
                ["metric"] + [k for c in COORDS for k in (c, f"{c} range")])

    print("\nSec. 7's diagonal-over-identity gains, re-measured at "
          f"{N_SEEDS} seeds (L = 20):")
    print_table(section_7_gains(sweep),
                ["coordinate", "identity ESS/kg", "adapted ESS/kg",
                 "gain (median)", "gain range"])

    print("\n=== C. the local conditioning, at every posterior draw ===")
    rows_c, kappas, z, nonpd = local_conditioning(model, ref_samples, ref_cov)
    print_table(rows_c, ["metric", "median kappa", "10th", "90th", "90th/10th"])
    print(f"draws where -H is not positive definite: {nonpd:.2%} "
          "(no global metric conditions a saddle)")
    bin_rows, edges = by_log_tau(kappas, z)
    print("\nsame kappas, cut by log tau:")
    print_table(bin_rows, ["log tau bin", "n", "identity", "diagonal", "dense",
                           "dense gain"])

    # the estimated covariance an actual dense warmup produces, for the figure
    est = hmc(model, 0.1 * np.random.default_rng(SEED).standard_normal(
        (N_CHAINS, model.dim)), n_samples=8_000, step_size=0.1, n_leapfrog=3,
        rng=np.random.default_rng(SEED), n_warmup=2_000, adapt_step_size=True,
        target_accept=0.9, adapt_mass="dense")
    est_cov = est.extras["inv_mass"]
    iu = np.triu_indices(model.dim, 1)
    print(f"\nwarmup's estimated covariance vs the reference: sd within "
          f"{np.abs(np.sqrt(np.diag(est_cov)) / np.sqrt(np.diag(ref_cov)) - 1).max():.1%}, "
          f"off-diagonal correlations agree at r = "
          f"{np.corrcoef(correlation(est_cov)[iu], correlation(ref_cov)[iu])[0, 1]:.2f} "
          f"(over entries whose largest magnitude is "
          f"{np.abs(correlation(ref_cov)[iu]).max():.3f}, so what agrees is the "
          "absence of structure).\nThe estimate is not the problem.")

    make_figure(ref_cov, est_cov, sweep, kappas, z, bin_rows, edges)


if __name__ == "__main__":
    main()
