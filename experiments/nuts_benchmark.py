"""Experiment 9: NUTS vs fixed-length HMC vs RWMH -- who pays the least gradient.

Fixed-length HMC (Sec. 4) still carries one hand-set knob after the step size is
dual-averaged: the trajectory length ``L``. Too short and the proposal barely
moves; too long and the trajectory U-turns back on itself, spending gradients to
return roughly where it started. NUTS (``mcmc.nuts``) removes the knob -- it
doubles the trajectory until it starts to double back -- so the honest question
is whether that automation *costs* efficiency or *buys* it.

The yardstick is **ESS per gradient** (per 1000 model evaluations), the
hardware-independent currency: an HMC/NUTS "evaluation" is one gradient, a RWMH
"evaluation" is one density (cheaper -- so the per-eval column flatters RWMH, the
same honest asterisk as ``external_benchmark.py``). Two targets:

- **Neal's funnel, non-centered** (10-dim). The reparameterization
  ``x_i = e^{v/2} z_i`` turns the funnel into an independent Gaussian
  ``N(0, diag(9, 1, ..., 1))`` (Sec. 4.6), a benign geometry where the point is
  purely trajectory length: NUTS should match or beat a *tuned* fixed ``L``
  while choosing its length automatically, and both should crush RWMH's random
  walk over a 3-sigma-wide direction.
- **Eight schools, non-centered** (10-dim). A real hierarchical posterior whose
  hard coordinate is ``tau`` (the funnel-shaped scale). ESS(tau) per gradient is
  the number that matters.

The third panel is the honest limit. Run NUTS on the *centered* funnel and its
divergences cluster in the neck: our integrator has only a diagonal metric, so
the neck's curvature defeats it exactly as it defeats fixed-L HMC (Sec. 4.9,
"simplifications vs Stan" -- Stan diverges here too; the real fix is the
non-centering, not a fancier sampler). NUTS is not magic; it removes the
length knob, not the geometry.

Run:  python experiments/nuts_benchmark.py   (~1-2 min)
"""

import time

import numpy as np

from common import plt, print_table, savefig
from mcmc.diagnostics import efficiency_summary, split_rhat
from mcmc.hmc import hmc
from mcmc.metropolis import random_walk_metropolis
from mcmc.models import EightSchoolsNonCentered
from mcmc.nuts import nuts
from mcmc.targets import Gaussian, NealsFunnel

SEED = 20260703
N_CHAINS = 4
DIM = 10


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def _row(name, gradient, chains, seconds, n_evals, hard_idx, hard_label, extras):
    """One table row summarizing a sampler on a set of scalar coordinates.

    ``chains`` is (n_chains, n_samples, dim); we report the hardest coordinate's
    ESS/grad and the worst coordinate's raw ESS (the honest summary of a target
    where one direction lags), plus NUTS's mean tree depth and divergence count.
    """
    hard = efficiency_summary(chains[:, :, hard_idx], seconds, n_evals)
    per_dim = [efficiency_summary(chains[:, :, d], seconds, n_evals)
               for d in range(chains.shape[2])]
    worst = min(per_dim, key=lambda s: s["ess"])
    depth = extras.get("tree_depth")
    return {
        "sampler": name,
        "grad": "yes" if gradient else "no",
        "draws": chains.shape[0] * chains.shape[1],
        "wall s": seconds,
        f"ESS({hard_label})": hard["ess"],
        "min ESS": worst["ess"],
        f"ESS({hard_label})/1k ev": hard["ess_per_keval"],
        "depth": float(depth.mean()) if depth is not None else float("nan"),
        "div": int(extras.get("n_divergent", 0)),
        f"R-hat({hard_label})": split_rhat(chains[:, :, hard_idx]),
    }


# --------------------------------------------------------------------------
# Problem A: Neal's funnel, non-centered (v, z) ~ N(0, diag(9, 1, ..., 1))
# --------------------------------------------------------------------------
def funnel_noncentered():
    print("=" * 78)
    print("Problem A: Neal's funnel (non-centered), 10-dim -- length knob only")
    print("=" * 78)
    target = Gaussian(np.zeros(DIM), np.diag([9.0] + [1.0] * (DIM - 1)))
    rng = np.random.default_rng(SEED)
    x0 = rng.standard_normal((N_CHAINS, DIM)) * np.sqrt([9.0] + [1.0] * (DIM - 1))
    rows = []

    # RWMH: a single step size must straddle sd-3 (v) and sd-1 (z) directions.
    res, sec = _timed(lambda: random_walk_metropolis(
        target, x0, n_samples=60_000, step_size=0.7, rng=rng, n_warmup=5_000))
    rows.append(_row("RWMH", False, res.samples, sec,
                     (65_000) * N_CHAINS, 0, "v", res.extras))

    # fixed-L HMC: L hand-picked at 20 (a reasonable guess for this geometry).
    res, sec = _timed(lambda: hmc(
        target, x0, n_samples=12_000, step_size=0.5, n_leapfrog=20, rng=rng,
        n_warmup=1_000, adapt_step_size=True))
    rows.append(_row("HMC (fixed L=20)", True, res.samples, sec,
                     res.extras["n_grad_evals"], 0, "v", res.extras))

    # NUTS: no L; it grows each trajectory to its own U-turn.
    res, sec = _timed(lambda: nuts(
        target, x0, n_samples=4_000, step_size=0.5, rng=rng,
        n_warmup=1_000, adapt_step_size=True))
    rows.append(_row("NUTS", True, res.samples, sec,
                     res.extras["n_grad_evals"], 0, "v", res.extras))

    cols = ["sampler", "grad", "draws", "wall s", "ESS(v)", "min ESS",
            "ESS(v)/1k ev", "depth", "div", "R-hat(v)"]
    print_table(rows, cols)
    print("\nOn this benign (independent-Gaussian) geometry the only question is\n"
          "trajectory length: NUTS matches or beats a hand-tuned fixed L per\n"
          "gradient while choosing the length itself, and both gradient methods\n"
          "leave RWMH's random walk over the sd-3 direction far behind.")
    return rows


# --------------------------------------------------------------------------
# Problem B: eight schools, non-centered -- a real hierarchical posterior
# --------------------------------------------------------------------------
def eight_schools_noncentered():
    print("\n" + "=" * 78)
    print("Problem B: eight schools (non-centered), 10-dim -- hard coord is tau")
    print("=" * 78)
    model = EightSchoolsNonCentered()
    rng = np.random.default_rng(SEED + 1)
    z0 = 0.1 * rng.standard_normal((N_CHAINS, model.dim))
    rows = []

    # For eight schools we score the sampler in (mu, tau) space. Build a
    # (n_chains, n_samples, 2) array of [mu, tau] so _row's coordinate machinery
    # applies directly; tau is the hard coordinate (index 1).
    def mutau(res):
        p = model.transform(res.samples)
        return np.stack([p["mu"], p["tau"]], axis=-1)

    res, sec = _timed(lambda: random_walk_metropolis(
        model, z0, n_samples=60_000, step_size=0.25, rng=rng, n_warmup=5_000))
    rows.append(_row("RWMH", False, mutau(res), sec,
                     65_000 * N_CHAINS, 1, "tau", res.extras))

    res, sec = _timed(lambda: hmc(
        model, z0, n_samples=12_000, step_size=0.1, n_leapfrog=20, rng=rng,
        n_warmup=2_000, adapt_step_size=True, target_accept=0.9))
    rows.append(_row("HMC (fixed L=20)", True, mutau(res), sec,
                     res.extras["n_grad_evals"], 1, "tau", res.extras))

    res, sec = _timed(lambda: nuts(
        model, z0, n_samples=4_000, step_size=0.1, rng=rng,
        n_warmup=2_000, adapt_step_size=True, target_accept=0.9))
    rows.append(_row("NUTS", True, mutau(res), sec,
                     res.extras["n_grad_evals"], 1, "tau", res.extras))

    cols = ["sampler", "grad", "draws", "wall s", "ESS(tau)", "min ESS",
            "ESS(tau)/1k ev", "depth", "div", "R-hat(tau)"]
    print_table(rows, cols)
    print("\ntau is the coordinate every sampler finds hard. NUTS delivers the\n"
          "most ESS(tau) per gradient with no length tuning; fixed-L HMC needs L\n"
          "chosen by hand and RWMH's per-eval number flatters it (its eval is a\n"
          "cheap density, not a gradient) yet it still trails on the coordinate\n"
          "that matters.")
    return rows


# --------------------------------------------------------------------------
# Problem C (the honest limit): NUTS diverges in the CENTERED funnel neck
# --------------------------------------------------------------------------
def funnel_divergences():
    print("\n" + "=" * 78)
    print("Problem C: NUTS on the CENTERED funnel -- divergences mark the neck")
    print("=" * 78)
    target = NealsFunnel(dim=DIM, sigma_v=3.0)
    rng = np.random.default_rng(SEED + 2)
    x0 = rng.standard_normal((N_CHAINS, DIM))

    # Centered: the conditional scale e^{v} spans orders of magnitude, so a
    # single step size cannot integrate the neck (v << 0) and the mouth at once.
    res_c = nuts(
        target, x0, n_samples=3_000, step_size=0.5, rng=rng, n_warmup=1_000,
        adapt_step_size=True, target_accept=0.8,
    )
    v_c = res_c.samples[:, :, 0]
    div = res_c.extras["divergent"]
    frac = 100.0 * div.mean()
    print(f"centered NUTS: {res_c.extras['n_divergent']} divergent iterations "
          f"({frac:.1f}% of draws), mean depth {res_c.extras['tree_depth'].mean():.1f}")
    print(f"  E[v] = {v_c.mean():.3f} (true 0), sd[v] = {v_c.std(ddof=1):.3f} "
          f"(true 3.0) -- the neck is under-visited")

    # Non-centered: same target, fixed geometry, essentially no divergences.
    nctarget = Gaussian(np.zeros(DIM), np.diag([9.0] + [1.0] * (DIM - 1)))
    res_n = nuts(
        nctarget, x0, n_samples=3_000, step_size=0.5, rng=rng, n_warmup=1_000,
        adapt_step_size=True,
    )
    w = res_n.samples
    samples_n = np.concatenate(
        [w[:, :, :1], np.exp(0.5 * w[:, :, :1]) * w[:, :, 1:]], axis=2)
    v_n = samples_n[:, :, 0]
    print(f"non-centered NUTS: {res_n.extras['n_divergent']} divergent, "
          f"sd[v] = {v_n.std(ddof=1):.3f} (true 3.0)")

    exact = target.sample(60_000, rng)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharex=True, sharey=True,
                             constrained_layout=True)
    # centered: draws + divergences highlighted where they land (the neck)
    ax = axes[0]
    pts = res_c.pooled()
    dmask = div.ravel()
    ax.plot(pts[~dmask, 1][::20], pts[~dmask, 0][::20], ".", ms=1.5, alpha=0.25,
            color="C0", label="draws")
    ax.plot(pts[dmask, 1], pts[dmask, 0], "x", ms=4, color="C3", mew=1.0,
            label=f"divergent ({int(dmask.sum())})")
    ax.set_title("NUTS on the centered funnel", loc="left")
    ax.set_ylabel("$v$")
    ax.set_xlabel("$x_1$")
    ax.legend(fontsize=7, loc="upper right")
    # non-centered: mapped back, clean, covers the neck
    ax = axes[1]
    pts_n = samples_n.reshape(-1, DIM)
    ax.plot(pts_n[::20, 1], pts_n[::20, 0], ".", ms=1.5, alpha=0.25, color="C2")
    ax.set_title("NUTS non-centered (geometry fixed)", loc="left")
    ax.set_xlabel("$x_1$")
    for ax in axes:
        ax.set_xlim(-25, 25)
        ax.set_ylim(-10, 10)
    fig.suptitle("NUTS removes the length knob, not the geometry: "
                 "divergences flag the neck", x=0.02, ha="left", y=1.05)
    savefig(fig, "nuts_funnel_divergences.png")


def figure(rows_a, rows_b):
    """Two-panel ESS-per-gradient bar chart (the headline of this experiment)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)

    def panel(ax, rows, key, ylabel, title):
        names = [r["sampler"] for r in rows]
        vals = [r[key] for r in rows]
        colors = ["C0" if r["grad"] == "no" else "C3" for r in rows]
        ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)),
                      [n.replace(" ", "\n") for n in names], fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")

    panel(axes[0], rows_a, "ESS(v)/1k ev", "ESS(v) / 1k evals",
          "Funnel non-centered (10-dim)")
    panel(axes[1], rows_b, "ESS(tau)/1k ev", "ESS(tau) / 1k evals",
          "Eight schools (10-dim)")

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="C3", label="gradient-based (eval = gradient)"),
                        Patch(color="C0", label="gradient-free (eval = density)")],
               loc="upper right", fontsize=7)
    fig.suptitle("ESS per model evaluation: NUTS vs fixed-L HMC vs RWMH",
                 x=0.02, ha="left")
    savefig(fig, "nuts_benchmark.png")


def main():
    rows_a = funnel_noncentered()
    rows_b = eight_schools_noncentered()
    figure(rows_a, rows_b)
    funnel_divergences()


if __name__ == "__main__":
    main()
