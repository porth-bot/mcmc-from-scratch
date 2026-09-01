"""Experiment 14: estimating a dense metric, on a target whose answer is known.

Sec. 4.10 gives the dense metric's closed form for a Gaussian: with covariance
Sigma_pi, the choice M^-1 = Sigma_pi makes the whitened Hessian exactly I, so
kappa = 1 in every direction at once, while the *best possible diagonal* metric
leaves kappa(R), the correlation matrix's own conditioning -- 199 at rho = 0.99
however well the marginals are scaled. That is the whole point of a dense
metric, and it is a statement about a metric we are handed.

Warmup is not handed one. It has to estimate Sigma from a window of draws, and
that is where the dense metric can lose what the algebra won: d(d+1)/2 numbers
from the same window that gave the diagonal d of them. This experiment asks
whether the estimate is good enough to keep the win, on a target where "good
enough" can be checked against the exact answer.

Three studies, all on AR(1) Gaussians (Sigma_ij = rho^|i-j|), which are
strongly correlated, positive definite for any |rho| < 1, and have a closed-form
optimal metric:

  A. Estimator quality vs window size. Frobenius error AND the loss a metric is
     actually judged by -- kappa(Sigma_hat Sigma^-1), the whitened Hessian's
     conditioning. These are different losses and the ranking is not the same
     under both, which is the finding this study exists to record.
  B. Where the estimated dense metric stops paying. Sweep the dimension at a
     fixed warmup budget: the algebraic win kappa(R) -> 1 is dimension-free,
     the estimation cost is not, so there is a d at which they cross.
  C. End to end through the sampler. ESS per 1000 gradient evaluations for
     identity / adapted-diagonal / adapted-dense on the same target, same
     seeds, warmup charged to all three.

Run:  python experiments/dense_metric_estimation.py
"""

import numpy as np

from common import plt, print_table, savefig
from mcmc.adapt import estimate_covariance, WindowMoments, whitened_condition_number
from mcmc.diagnostics import ess
from mcmc.hmc import hmc
from mcmc.targets import Gaussian

SEED = 20260826
N_CHAINS = 4
SHRINKAGES = ["stan", "ledoit-wolf"]


def ar1(d, rho):
    i = np.arange(d)
    return rho ** np.abs(i[:, None] - i[None, :])


def kappa_of_correlation(cov):
    """kappa(R): the floor no diagonal metric can go below (Sec. 4.10)."""
    s = np.sqrt(np.diag(cov))
    return float(np.linalg.cond(cov / np.outer(s, s)))


def _window(rng, cov, n):
    """A window of n i.i.d. draws from N(0, cov), as WindowMoments would hold it.

    Drawing i.i.d. rather than running a chain is deliberate for studies A and B:
    it isolates the estimator from the sampler's autocorrelation, so a bad
    number is the estimator's fault and not the chain's. Study C puts the real
    chain back and pays the real price.
    """
    chol = np.linalg.cholesky(cov)
    w = WindowMoments(cov.shape[0], mode="dense")
    w.add(rng.standard_normal((n, cov.shape[0])) @ chol.T)
    return w


# -- A. two losses, and they disagree ----------------------------------------

def estimator_quality(d=10, rho=0.95, n_rep=40):
    cov = ar1(d, rho)
    kappa_diag = whitened_condition_number(np.diag(cov), cov)
    rng = np.random.default_rng(SEED)
    ns = [d // 2, d, 2 * d, 5 * d, 20 * d, 100 * d]
    rows = []
    for n in ns:
        acc = {s: {"frob": [], "kappa": []} for s in SHRINKAGES}
        acc["sample"] = {"frob": [], "kappa": []}
        for _ in range(n_rep):
            w = _window(rng, cov, n)
            S = w.covariance()
            acc["sample"]["frob"].append(
                np.linalg.norm(S - cov) / np.linalg.norm(cov)
            )
            # the raw sample covariance is not always a metric; record when it
            # is not rather than silently dropping the case.
            try:
                acc["sample"]["kappa"].append(whitened_condition_number(S, cov))
            except np.linalg.LinAlgError:
                acc["sample"]["kappa"].append(np.inf)
            for sh in SHRINKAGES:
                est = estimate_covariance(w, shrinkage=sh)
                acc[sh]["frob"].append(
                    np.linalg.norm(est - cov) / np.linalg.norm(cov)
                )
                acc[sh]["kappa"].append(whitened_condition_number(est, cov))
        for name in ["sample"] + SHRINKAGES:
            rows.append({
                "n/d": n / d,
                "estimator": name,
                "frob_err": float(np.mean(acc[name]["frob"])),
                "kappa": float(np.median(acc[name]["kappa"])),
            })
    return rows, cov, kappa_diag


# -- B. where estimation eats the algebraic win -------------------------------

def dimension_sweep(rho=0.95, n_window=400, n_rep=25, shrinkage="ledoit-wolf"):
    """Fixed warmup window, growing d. kappa(R) grows too, so the question is
    which grows faster: what the dense metric could win, or what estimating it
    from a fixed budget costs."""
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for d in [2, 4, 8, 16, 32, 64, 128]:
        cov = ar1(d, rho)
        kappa_diag = whitened_condition_number(np.diag(cov), cov)
        ks = []
        for _ in range(n_rep):
            w = _window(rng, cov, n_window)
            ks.append(whitened_condition_number(
                estimate_covariance(w, shrinkage=shrinkage), cov))
        rows.append({
            "d": d,
            "n/d": n_window / d,
            "kappa_diag": kappa_diag,
            "kappa_dense_est": float(np.median(ks)),
            "gain": kappa_diag / float(np.median(ks)),
        })
    return rows


# -- C. through the sampler ---------------------------------------------------

def _ess_per_keval(res, coord):
    """ESS normalized by 1000 gradient evaluations, warmup included -- the
    metric's estimation cost is real gradient work and must be charged."""
    return 1000.0 * ess(res.samples[:, :, coord]) / res.extras["n_grad_evals"]


def sampler_benchmark(d=10, rho=0.95, n_seeds=3, lengths=(1, 2, 5, 10, 25, 50)):
    """ESS/gradient for four metrics -- each at its *own* best trajectory length.

    Holding n_leapfrog fixed across metrics is the obvious comparison and it is
    wrong, by an order of magnitude. HMC's optimal trajectory length is set by
    the slowest direction's period, which scales like sqrt(kappa): the identity
    metric on this target (kappa = 324) needs a long trajectory to cross the
    correlated direction at all, while a whitened target (kappa ~ 1.3) has one
    period and is decorrelated in a couple of steps. Fix L at the identity's
    optimum and the dense metric is charged ~20 gradients per sample it does
    not need; the measured 9x win collapses to 1.1x. So sweep L for every
    metric and report each at its best -- and report the fixed-L number too,
    since that is the comparison a reader would otherwise assume.
    """
    cov = ar1(d, rho)
    target = Gaussian(mean=np.zeros(d), cov=cov)
    x0 = np.zeros((N_CHAINS, d))
    common = dict(n_samples=4_000, step_size=0.25, n_warmup=2_000,
                  adapt_step_size=True)
    configs = [
        ("identity", dict(adapt_mass=False)),
        ("diagonal", dict(adapt_mass="diag")),
        ("dense (stan)", dict(adapt_mass="dense", dense_shrinkage="stan")),
        ("dense (LW)", dict(adapt_mass="dense", dense_shrinkage="ledoit-wolf")),
    ]
    curves = {}
    rows = []
    for name, kw in configs:
        by_L = []
        kap = []
        for L in lengths:
            eff = []
            for s in range(n_seeds):
                # same seed across configs: the metric is the only difference
                res = hmc(target, x0, n_leapfrog=L,
                          rng=np.random.default_rng(SEED + s), **common, **kw)
                # average over coordinates rather than cherry-picking one
                eff.append(np.mean(
                    [_ess_per_keval(res, c) for c in range(d)]))
                if L == lengths[0]:
                    kap.append(
                        whitened_condition_number(res.extras["inv_mass"], cov))
            by_L.append(float(np.mean(eff)))
        curves[name] = by_L
        best = int(np.argmax(by_L))
        rows.append({
            "metric": name,
            "kappa": float(np.median(kap)),
            "best_L": lengths[best],
            "ess_per_kgrad": by_L[best],
            "at_L=25": by_L[list(lengths).index(25)],
        })
    return rows, curves, list(lengths)


# -- figure -------------------------------------------------------------------

def make_figure(rows_a, kappa_diag, rows_b, curves, lengths):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6))

    ax = axes[0, 0]
    for name, style in [("sample", "o--"), ("stan", "s-"), ("ledoit-wolf", "^-")]:
        sub = [r for r in rows_a if r["estimator"] == name]
        ax.loglog([r["n/d"] for r in sub], [r["frob_err"] for r in sub],
                  style, ms=4, label=name)
    ax.set_xlabel("window size / dimension")
    ax.set_ylabel(r"$\|\hat\Sigma - \Sigma\|_F\,/\,\|\Sigma\|_F$")
    ax.set_title("A. Frobenius error: the raw sample covariance wins")
    ax.legend()

    ax = axes[0, 1]
    for name, style in [("sample", "o--"), ("stan", "s-"), ("ledoit-wolf", "^-")]:
        sub = [r for r in rows_a if r["estimator"] == name]
        ax.loglog([r["n/d"] for r in sub],
                  [min(r["kappa"], 1e18) for r in sub], style, ms=4, label=name)
    ax.axhline(kappa_diag, color="k", ls=":", lw=1)
    ax.text(0.55, kappa_diag * 1.3, r"best diagonal: $\kappa(R)$", fontsize=7)
    ax.axhline(1.0, color="0.5", ls=":", lw=1)
    ax.text(0.55, 1.1, r"exact dense: $\kappa=1$", fontsize=7, color="0.4")
    ax.set_xlabel("window size / dimension")
    ax.set_ylabel(r"$\kappa(\hat\Sigma\,\Sigma^{-1})$")
    ax.set_title("B. the loss HMC pays: it loses, catastrophically")
    ax.legend()

    ax = axes[1, 0]
    ax.semilogx([r["d"] for r in rows_b], [r["kappa_diag"] for r in rows_b],
                "k:", marker="o", ms=4, label=r"best diagonal, $\kappa(R)$")
    ax.semilogx([r["d"] for r in rows_b],
                [r["kappa_dense_est"] for r in rows_b],
                "^-", ms=4, label="estimated dense (400-draw window)")
    ax.set_yscale("log")
    ax.set_xlabel("dimension")
    ax.set_ylabel(r"$\kappa$")
    ax.set_title(r"C. a fixed window degrades slower than $\kappa(R)$ grows")
    ax.legend()

    ax = axes[1, 1]
    for name, style in [("identity", "o--"), ("diagonal", "s--"),
                        ("dense (stan)", "^-"), ("dense (LW)", "v-")]:
        ax.loglog(lengths, curves[name], style, ms=4, label=name)
    ax.set_xlabel(r"leapfrog steps per iteration $L$")
    ax.set_ylabel("ESS per 1000 gradients")
    ax.set_title(r"D. each metric has its own best $L$")
    ax.legend(fontsize=7)

    fig.suptitle(
        r"Estimating a dense metric from warmup, AR(1) Gaussian $\rho = 0.95$",
        y=1.01)
    fig.tight_layout()
    savefig(fig, "dense_metric_estimation.png")


def main():
    rows_a, cov_a, kappa_diag = estimator_quality()
    print(f"A. estimator quality, d=10, rho=0.95 "
          f"(kappa(R) = {kappa_diag:.1f}, exact dense = 1.0)")
    print_table(rows_a, ["n/d", "estimator", "frob_err", "kappa"])

    rows_b = dimension_sweep()
    print("\nB. fixed 400-draw window, growing d (Ledoit-Wolf)")
    print_table(rows_b, ["d", "n/d", "kappa_diag", "kappa_dense_est", "gain"])

    rows_c, curves, lengths = sampler_benchmark()
    print("\nC. through the sampler, d=10, rho=0.95, warmup charged")
    print_table(rows_c, ["metric", "kappa", "best_L", "ess_per_kgrad", "at_L=25"])

    make_figure(rows_a, kappa_diag, rows_b, curves, lengths)


if __name__ == "__main__":
    main()
