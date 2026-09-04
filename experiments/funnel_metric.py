"""Experiment 19: the funnel, where the dense metric fails too -- and in closed form.

Sec. 14 measured a null: on eight schools a dense metric beats a diagonal by
1.06x, inside the seed range, because the residual Sec. 7 named is
position-dependent curvature rather than a rotation. That was one posterior and
the argument rested on a reference run.

Neal's funnel makes the same argument out of identities (Sec. 4.12). Everything
here is exact -- the target's covariance, its Hessian, its scaling symmetry --
so the null is *predicted* before the sampler runs and the sampler's job is
only to confirm it did not miss something. Four studies:

  A. The headroom, in closed form. Cov(v, x_i) = Cov(x_i, x_j) = 0 exactly, so
     R = I, so the best diagonal metric already reaches kappa = 1 and a dense
     metric's algebraic advantage over it is exactly 1.00. Beside that, what
     warmup's estimate of that same covariance actually looks like: its
     diagonal is biased low by a factor and its off-diagonals are noise around
     an exact zero, so the estimated dense metric is strictly the estimated
     diagonal plus a rotation of nothing.
  B. Run it, and check the prediction survives contact. Five metrics, L swept
     per metric, several seeds, warmup charged. The oracle arms are handed the
     exact covariance rather than estimating one, so a null cannot be blamed on
     the estimate. sd[v] against its true value of sigma_v is reported beside
     ESS, because on this target a fast chain that never reaches the neck is
     the failure mode and ESS alone will not say so.
  C. Why. The local curvature at every draw, as max|lambda|/min|lambda| of the
     whitened Hessian -- absolute, because -H here is positive definite at
     essentially no draw at all (Sec. 4.12), which is itself reported and is
     metric-free by Sylvester's law. Then the same numbers cut by v, which is
     where the position dependence becomes visible: a global metric slides that
     curve, it does not flatten it.
  D. The step size, which is the mechanism. eps_max(v) for each metric,
     against the e^{v/2} law, with the neck cost of each metric's single
     adapted step size read off directly.

And the fix, which is not a metric: the non-centered parameterization
(experiments/funnel.py, Sec. 4.6) is a *position-dependent* change of
coordinates, and it mixes. Riemannian HMC is the general form of that trick.
Neither is a mass matrix, which is the point.

Run:  python experiments/funnel_metric.py
"""

import math

import numpy as np

from common import plt, print_table, savefig
from mcmc.adapt import definite_fraction, whitened_abs_condition_numbers
from mcmc.diagnostics import ess, integrated_autocorr_time, split_rhat
from mcmc.hmc import hmc
from mcmc.targets import NealsFunnel

SEED = 20260928
N_CHAINS = 4
N_SEEDS = 4
DIM = 10
SIGMA_V = 3.0
LENGTHS = (1, 2, 5, 10, 25, 40)   # 40 is experiments/funnel.py's fixed L
THIN = 40                          # study C builds a (10, 10) Hessian per draw
CLAMP = 1.0 + 1e-12                # Geyer's tau floor, as in Sec. 14


# -- A. the headroom, closed form ---------------------------------------------

def available_rotation(cov):
    """kappa of the whitened Hessian per metric, Gaussian approximation.

    Same three rows as Sec. 14's study A, but every entry here is exact: the
    target's covariance is diagonal in closed form, so the best diagonal metric
    and the exact dense metric are the *same matrix* and must return the same
    number. Computed rather than asserted -- if this table ever shows the two
    differing, the closed form and the linear algebra have stopped agreeing.
    """
    d = cov.shape[0]
    prec = np.linalg.inv(cov)
    return [
        {"metric": "identity", "kappa": float(whitened_abs_condition_numbers(np.eye(d), prec)[0])},
        {"metric": "best diagonal", "kappa": float(whitened_abs_condition_numbers(np.diag(np.diag(cov)), prec)[0])},
        {"metric": "exact dense", "kappa": float(whitened_abs_condition_numbers(cov, prec)[0])},
    ]


def estimated_metric(model, n_samples=8_000, n_warmup=2_000, seed=SEED):
    """The covariance an actual dense warmup produces here, for comparison."""
    z0 = 0.1 * np.random.default_rng(seed).standard_normal((N_CHAINS, model.dim))
    res = hmc(model, z0, n_samples=n_samples, step_size=0.05, n_leapfrog=10,
              rng=np.random.default_rng(seed), n_warmup=n_warmup,
              adapt_step_size=True, target_accept=0.9, adapt_mass="dense")
    return res.extras["inv_mass"], res


def estimate_quality(est, cov):
    """What the estimate got right and wrong, against the closed form.

    Two columns because they fail differently: the diagonal is biased (the
    x-variance estimator is dominated by its largest e^v draw, so a window
    typically underestimates e^{sigma_v^2/2}), while the off-diagonals are
    unbiased noise around an exact zero -- and it is the off-diagonals that are
    the dense metric's entire extra degree of freedom over the diagonal.
    """
    d = cov.shape[0]
    iu = np.triu_indices(d, 1)
    s = np.sqrt(np.diag(est))
    corr = est / np.outer(s, s)
    return [
        {"quantity": "sd[v]        (true %.3f)" % np.sqrt(cov[0, 0]),
         "estimate": float(s[0]), "ratio": float(s[0] / np.sqrt(cov[0, 0]))},
        {"quantity": "sd[x_i] med  (true %.3f)" % np.sqrt(cov[1, 1]),
         "estimate": float(np.median(s[1:])),
         "ratio": float(np.median(s[1:]) / np.sqrt(cov[1, 1]))},
        {"quantity": "max |corr|   (true 0.000)",
         "estimate": float(np.abs(corr[iu]).max()), "ratio": float("inf")},
        {"quantity": "rms |corr|   (true 0.000)",
         "estimate": float(np.sqrt(np.mean(corr[iu] ** 2))), "ratio": float("inf")},
    ]


# -- B. through the sampler ---------------------------------------------------

def _configs(cov):
    return [
        ("identity", dict(adapt_mass=False)),
        ("diagonal (adapted)", dict(adapt_mass="diag")),
        ("dense (adapted)", dict(adapt_mass="dense")),
        ("diagonal (oracle)", dict(metric=np.diag(cov))),   # 1-D -> DiagonalMetric
        ("dense (oracle)", dict(metric=cov)),
    ]


def sampler_sweep(model, cov, n_samples=5_000, n_warmup=1_500):
    """Per metric and per L: tau(v), ESS/1k gradients, sd[v], divergences.

    sd[v] is the column Sec. 14 did not need. The funnel's failure is not slow
    mixing, it is a chain that never enters the neck: such a chain reports a
    short autocorrelation time and a *wrong answer*, so ESS on its own would
    rank it well. sigma_v = 3 is the exact truth to compare against.
    """
    z0 = 0.1 * np.random.default_rng(SEED).standard_normal((N_CHAINS, model.dim))
    common = dict(n_samples=n_samples, step_size=0.05, n_warmup=n_warmup,
                  adapt_step_size=True, target_accept=0.9)
    out = {}
    for name, kw in _configs(cov):
        for L in LENGTHS:
            per_seed = []
            for s in range(N_SEEDS):
                res = hmc(model, z0, n_leapfrog=L,
                          rng=np.random.default_rng(SEED + s), **common, **kw)
                v = res.samples[:, :, 0]
                g = res.extras["n_grad_evals"]
                tau = integrated_autocorr_time(v)
                per_seed.append({
                    "tau": tau,
                    "ess_kgrad": 1000.0 * v.size / tau / g,
                    "sd_v": float(v.std(ddof=1)),
                    "min_v": float(v.min()),
                    "div": int(res.extras["n_divergent"]),
                    "accept": float(res.accept_rate.mean()),
                    "rhat": split_rhat(v),
                    "eps": float(res.extras["step_size"]),
                })
            out[(name, L)] = per_seed
            print(f"    {name:20s} L={L:2d}  tau {np.median([p['tau'] for p in per_seed]):6.2f}"
                  f"  ESS/kgrad {np.median([p['ess_kgrad'] for p in per_seed]):7.2f}"
                  f"  sd[v] {np.median([p['sd_v'] for p in per_seed]):5.2f}"
                  f"  min v {np.median([p['min_v'] for p in per_seed]):+6.2f}"
                  f"  div {int(np.median([p['div'] for p in per_seed])):5d}")
    return out


def summarize_sweep(sweep, cov):
    """Each metric at its own best L, with the accuracy column beside it."""
    rows = []
    for name, _ in _configs(cov):
        by_L = {L: float(np.median([p["ess_kgrad"] for p in sweep[(name, L)]]))
                for L in LENGTHS}
        best_L = max(by_L, key=by_L.get)
        cell = sweep[(name, best_L)]
        rows.append({
            "metric": name,
            "best L": best_L,
            "ESS(v)/kgrad": by_L[best_L],
            "tau(v)": float(np.median([p["tau"] for p in cell])),
            "sd[v] (true 3.00)": float(np.median([p["sd_v"] for p in cell])),
            "min v": float(np.median([p["min_v"] for p in cell])),
            "eps": float(np.median([p["eps"] for p in cell])),
            "divergent": float(np.median([p["div"] for p in cell])),
            "seed spread": (max(p["ess_kgrad"] for p in cell)
                            / min(p["ess_kgrad"] for p in cell)),
        })
    return rows


def dense_over_diagonal(sweep, cov):
    """The one comparison the section exists to make, per seed and per L.

    Paired by seed: both arms see the same rng stream, so the ratio is a
    within-seed contrast and its spread across seeds is the noise floor the
    median has to clear.
    """
    rows = []
    for L in LENGTHS:
        for tag, a, b in [("adapted", "dense (adapted)", "diagonal (adapted)"),
                          ("oracle", "dense (oracle)", "diagonal (oracle)")]:
            r = [x["ess_kgrad"] / y["ess_kgrad"]
                 for x, y in zip(sweep[(a, L)], sweep[(b, L)])]
            rows.append({"L": L, "arms": tag, "dense/diagonal (median)": float(np.median(r)),
                         "range": f"{min(r):.2f}-{max(r):.2f}",
                         "clamped": sum(1 for p in sweep[(a, L)] + sweep[(b, L)]
                                        if p["tau"] <= CLAMP)})
    return rows


# -- C. the local curvature ---------------------------------------------------

def local_conditioning(model, cov, n=200_000, seed=SEED + 77):
    """kappa of the whitened Hessian at exact draws from the funnel.

    Exact draws rather than a reference chain, which the funnel uniquely
    allows: any sampler's draws under-represent the neck, and using them would
    understate exactly the region the section is about. The generative sampler
    is i.i.d. and correct by construction.
    """
    z = model.sample(n, np.random.default_rng(seed))[::THIN]
    A = -model.hess_logpdf(z)
    d = model.dim
    metrics = [("identity", np.eye(d)), ("diagonal", np.diag(np.diag(cov))),
               ("dense", cov)]
    kappas = {name: whitened_abs_condition_numbers(m, A) for name, m in metrics}
    rows = [{
        "metric": name,
        "median kappa": float(np.median(k)),
        "10th": float(np.quantile(k, 0.10)),
        "90th": float(np.quantile(k, 0.90)),
        "90th/10th": float(np.quantile(k, 0.90) / np.quantile(k, 0.10)),
    } for name, k in kappas.items()]
    return rows, kappas, z, definite_fraction(A)


def by_v(kappas, z, n_bins=5):
    t = z[:, 0]
    edges = np.quantile(t, np.linspace(0, 1, n_bins + 1))
    rows = []
    for i in range(n_bins):
        sel = (t >= edges[i]) & (t <= edges[i + 1])
        row = {"v bin": f"[{edges[i]:+.2f}, {edges[i+1]:+.2f}]", "n": int(sel.sum())}
        for name in ("identity", "diagonal", "dense"):
            row[name] = float(np.median(kappas[name][sel]))
        row["dense gain"] = row["diagonal"] / row["dense"]
        rows.append(row)
    return rows, edges


# -- D. the step size ---------------------------------------------------------

def step_size_limit(model, cov, vs, seed=SEED + 5):
    """eps_max(v) per metric, along one scaling orbit of the funnel.

    Walking the orbit v -> (v + c, e^{c/2} x) rather than resampling at each v
    isolates the effect: Sec. 4.12's congruence says the orbit is where the
    metric dependence lives, and resampling x at each height would mix that
    with the chi^2 spread of ||x||.
    """
    z = model.sample(1, np.random.default_rng(seed))[0]
    d = model.dim
    out = {}
    for name, S in [("identity", np.eye(d)), ("diagonal", np.diag(np.diag(cov))),
                    ("dense", cov)]:
        L = np.linalg.cholesky(S)
        eps = []
        for v in vs:
            c = v - z[0]
            zc = np.concatenate([[v], np.exp(0.5 * c) * z[1:]])
            A = -model.hess_logpdf(zc[None, :])[0]
            eps.append(2.0 / np.sqrt(np.abs(np.linalg.eigvalsh(L.T @ A @ L)).max()))
        out[name] = np.array(eps)
    return out


# -- figure -------------------------------------------------------------------

def make_figure(cov, est, sweep, kappas, z, bin_rows, edges, vs, eps_curves,
                adapted_eps):
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.8))
    d = cov.shape[0]

    ax = axes[0, 0]
    s = np.sqrt(np.diag(est))
    corr = est / np.outer(s, s)
    im = ax.imshow(corr - np.eye(d), cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    labels = ["$v$"] + [rf"$x_{{{j}}}$" for j in range(1, d)]
    ax.set_xticks(range(d), labels, fontsize=6)
    ax.set_yticks(range(d), labels, fontsize=6)
    iu = np.triu_indices(d, 1)
    ax.set_title("A. warmup's estimated correlations,\n"
                 rf"against a true $R = I$: max $|r|$ = {np.abs(corr[iu]).max():.3f}",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    styles = {"identity": ("o--", "0.45"), "diagonal (adapted)": ("s-", "#1f77b4"),
              "dense (adapted)": ("^-", "#a11"),
              "diagonal (oracle)": ("s:", "#1f77b4"), "dense (oracle)": ("^:", "#a11")}
    for name, (st, col) in styles.items():
        y = [float(np.median([p["sd_v"] for p in sweep[(name, L)]])) for L in LENGTHS]
        ax.semilogx(LENGTHS, y, st, color=col, ms=4, lw=1.2, label=name)
    ax.axhline(SIGMA_V, color="k", lw=1.2, ls="-")
    ax.text(LENGTHS[0], SIGMA_V * 1.01, r"true $\sigma_v = 3$", fontsize=6.5)
    ax.set_xlabel(r"leapfrog steps per iteration $L$")
    ax.set_ylabel(r"$\mathrm{sd}[v]$")
    best = {n: float(np.median([p["sd_v"] for p in sweep[(n, LENGTHS[-1])]]))
            for n in styles}
    ax.set_title(rf"B. at $L={LENGTHS[-1]}$ every metric still misses "
                 rf"$\sigma_v$ ({min(best.values()):.2f}–{max(best.values()):.2f})",
                 fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right")

    ax = axes[1, 0]
    # the dense curve lies exactly on the diagonal one (same matrix), so it is
    # drawn dashed on top rather than hidden underneath
    for name, col, st, lw in [("identity", "0.45", "-", 1.4),
                              ("diagonal", "#1f77b4", "-", 2.6),
                              ("dense", "#a11", "--", 1.4)]:
        k = np.sort(kappas[name])
        ax.semilogx(k, np.linspace(0, 1, len(k)), color=col, ls=st, lw=lw,
                    label="dense (exactly on top)" if name == "dense" else name)
    ax.set_xlabel(r"local $\max|\lambda| / \min|\lambda|$, per exact draw")
    ax.set_ylabel("empirical CDF")
    ax.set_title(rf"C. the rotation moves the median "
                 rf"{np.median(kappas['diagonal']) / np.median(kappas['dense']):.2f}$\times$; "
                 rf"position moves it {max(r['diagonal'] for r in bin_rows) / min(r['diagonal'] for r in bin_rows):.0f}$\times$",
                 fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    for name, col, st, lw in [("identity", "0.45", "-", 1.4),
                              ("diagonal", "#1f77b4", "-", 2.6),
                              ("dense", "#a11", "--", 1.4)]:
        ax.semilogy(vs, eps_curves[name], color=col, ls=st, lw=lw,
                    label="dense (exactly on top)" if name == "dense" else name)
    # the e^{v/2} law is a neck law (Sec. 4.12), so the reference is drawn only
    # where it is claimed to hold rather than across the saturating mouth
    neck_only = vs <= 0.0
    ref = eps_curves["diagonal"][0] * np.exp((vs[neck_only] - vs[0]) / 2.0)
    ax.semilogy(vs[neck_only], ref, "k:", lw=1.2)
    ax.text(vs[neck_only][-1] + 0.4, ref[-1], r"$e^{v/2}$", fontsize=8, va="center")
    for name, col in [("diagonal (adapted)", "#1f77b4"), ("dense (adapted)", "#a11")]:
        ax.axhline(adapted_eps[name], color=col, lw=0.9, ls="--", alpha=0.7)
    ax.set_ylim(min(c.min() for c in eps_curves.values()) * 0.6, None)
    ax.set_xlabel("$v$")
    ax.set_ylabel(r"$\varepsilon_{\max}(v)$")
    ax.set_title("D. every metric collapses like $e^{v/2}$ in the neck;\n"
                 "dashed = the single step size each one adapted to", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Neal's funnel: a dense metric has nothing to rotate, and "
                 "position is the problem", y=1.01)
    fig.tight_layout()
    savefig(fig, "funnel_metric.png")


def main():
    model = NealsFunnel(dim=DIM, sigma_v=SIGMA_V)
    _, cov = model.moments()
    print("=" * 78)
    print(f"Neal's funnel, {DIM}D: dense metric vs diagonal, against closed forms")
    print("=" * 78)

    print("\n=== A. the headroom, in closed form ===")
    print("exact covariance (mcmc/targets.py): diag(%.2f, %.2f x %d), R = I exactly"
          % (cov[0, 0], cov[1, 1], DIM - 1))
    print_table(available_rotation(cov), ["metric", "kappa"])
    print("The best diagonal and the exact dense metric are the same matrix here,\n"
          "so a dense metric's algebraic headroom on this target is exactly 1.00.\n"
          "Sec. 14 had to measure that number on eight schools (1.42); here it is\n"
          "a consequence of Cov(x_i, x_j) = E[E[x_i|v] E[x_j|v]] = 0.")

    est, est_res = estimated_metric(model)
    print("\nwhat a dense warmup actually estimates:")
    print_table(estimate_quality(est, cov), ["quantity", "estimate", "ratio"])
    print("So the estimated dense metric is a diagonal it gets wrong by a factor,\n"
          "plus off-diagonal entries that are noise around an exact zero. Its one\n"
          "extra degree of freedom over the diagonal is spent rotating nothing.")

    print("\n=== B. through the sampler, L swept per metric ===")
    sweep = sampler_sweep(model, cov)
    print()
    print_table(summarize_sweep(sweep, cov),
                ["metric", "best L", "ESS(v)/kgrad", "tau(v)", "sd[v] (true 3.00)",
                 "min v", "eps", "divergent", "seed spread"])
    print("\ndense over diagonal, paired by seed:")
    print_table(dense_over_diagonal(sweep, cov),
                ["L", "arms", "dense/diagonal (median)", "range", "clamped"])
    print("The oracle rows are the same *matrix* -- Sigma is diagonal, so the dense\n"
          "oracle is DenseMetric(diag(v)) against the diagonal oracle's\n"
          "DiagonalMetric(v). mcmc/metric.py says those associate the same reals\n"
          "differently, agree to ~1e-14 along a trajectory, and are nevertheless\n"
          "different chains once an accept comparison lands on the other side of\n"
          "its uniform draw. The ratio column shows exactly that: 1.000 at L = 1\n"
          "and 2, then scattering by tens of percent. It is the noise floor of this\n"
          "comparison, measured, and every adapted ratio sits inside it.")

    print("\n=== C. the local curvature, at exact draws ===")
    rows_c, kappas, z, pd_frac = local_conditioning(model, cov)
    print_table(rows_c, ["metric", "median kappa", "10th", "90th", "90th/10th"])
    assert np.array_equal(kappas["diagonal"], kappas["dense"])
    print("The diagonal and dense rows are identical because they are the same\n"
          "matrix: diag(diag(Sigma)) = Sigma when Sigma is diagonal. That is study\n"
          "A's headroom of 1.00, restated at every draw rather than at a Gaussian\n"
          "approximation -- and asserted, not eyeballed.")
    print(f"draws where -H is positive definite: {pd_frac:.4%} "
          "(Sec. 4.12: needs chi^2_9 < 0.22).\n"
          "Metric-free by Sylvester's law of inertia, so it is reported once.")
    bin_rows, edges = by_v(kappas, z)
    print("\nsame kappas, cut by v:")
    print_table(bin_rows, ["v bin", "n", "identity", "diagonal", "dense", "dense gain"])
    across = max(r["diagonal"] for r in bin_rows) / min(r["diagonal"] for r in bin_rows)
    best_rot = max(r["dense gain"] for r in bin_rows)
    print(f"position moves the diagonal metric's local kappa {across:.0f}x across "
          f"quintiles;\nthe rotation moves it at most {best_rot:.2f}x in any of them.")

    print("\n=== D. the step size, along one scaling orbit ===")
    vs = np.linspace(-9.0, 9.0, 73)
    eps_curves = step_size_limit(model, cov, vs)
    rows_d = []
    neck, mouth = np.argmin(np.abs(vs + 7.73)), np.argmin(np.abs(vs - 7.73))
    for name in ("identity", "diagonal", "dense"):
        e = eps_curves[name]
        rows_d.append({
            "metric": name,
            "eps_max at v=-7.73": float(e[neck]),
            "at v=0": float(e[len(vs) // 2]),
            "at v=+7.73": float(e[mouth]),
            "mouth/neck": float(e[mouth] / e[neck]),
        })
    print_table(rows_d, ["metric", "eps_max at v=-7.73", "at v=0", "at v=+7.73",
                         "mouth/neck"])
    assert np.array_equal(eps_curves["diagonal"], eps_curves["dense"])
    print("Note which way round the first column runs. The metric built from the\n"
          "*exact* covariance admits a step an order of magnitude smaller in the\n"
          "neck than the identity does, because Var(x_i) = e^{sigma_v^2/2} = 90 is\n"
          "a number the mouth chose: whitening by it inflates the neck's curvature\n"
          "by 90. Sec. 4.12's congruence again -- a global metric slides the curve\n"
          "along the orbit, and sliding it toward the mouth is sliding it away\n"
          "from the neck. mouth/neck falls short of the e^{7.73} = 2280 the neck\n"
          "law would give because eps saturates in the mouth, where the binding\n"
          "direction passes to v and its curvature is orbit-invariant.")
    adapted_eps = {n: float(np.median([p["eps"] for p in sweep[(n, LENGTHS[-1])]]))
                   for n in ("diagonal (adapted)", "dense (adapted)")}
    for n, e in adapted_eps.items():
        below = vs[eps_curves[n.split()[0]] < e]
        edge = float(below.max()) if below.size else float("-inf")
        # eps_max(v) increases with v, so {v : eps_max(v) < eps} is the neck
        # below `edge`, and the v-marginal is exactly N(0, sigma_v^2).
        frac = 0.5 * (1.0 + math.erf(edge / (SIGMA_V * math.sqrt(2.0))))
        print(f"{n:20s} adapted to eps = {e:.4f}: the leapfrog is unstable below "
              f"v = {edge:+.2f},\n{'':22s}which is {frac:.2%} of the v-marginal it "
              "cannot integrate through.")

    make_figure(cov, est, sweep, kappas, z, bin_rows, edges, vs, eps_curves,
                adapted_eps)


if __name__ == "__main__":
    main()
