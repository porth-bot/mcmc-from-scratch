"""Experiment 7: diagonal mass-matrix adaptation -- the cheap preconditioner.

Unit-metric HMC uses one step size for every direction. On an axis-aligned
target whose coordinates have very different scales, that single step size is
squeezed by the *tightest* direction (leapfrog's stability limit is set by the
largest curvature), so the *widest* direction is integrated with a step far
smaller than it could tolerate and mixes slowly. The fix is a change of
kinetic energy, not a better step-size tuner: give the momentum a mass matrix
M with K(p) = p^T M^{-1} p / 2, chosen so that M^{-1} = diag(marginal
variances). Each coordinate is then preconditioned to unit scale and one step
size fits all of them (full derivation: theory/derivations.md Sec. 4.8).

We adapt diag(M^{-1}) from windowed warmup variances (Stan's memoryless-window
scheme) and measure the payoff in the honest currency -- ESS per 1000 gradient
evaluations, so a metric that needs no extra gradients is credited fairly.

Two studies:

  A. Anisotropy sweep. Diagonal Gaussians N(0, diag(1, r^2)) for a range of
     scale ratios r (r=2 is the plan's sd=(1,2) case). The gain on the wide
     coordinate grows with r -- exactly the regime the metric is for.
  B. Eight schools (non-centered). A real posterior where the metric helps the
     population-mean coordinate mu (posterior sd ~ several) relative to the
     unit-scale eta_j, while being honest that a *diagonal* metric cannot cure
     the funnel curvature in (log tau, eta) -- that is NUTS/Riemannian territory
     (Days 17-18).

Run:  python experiments/mass_matrix.py
"""

import numpy as np

from common import plt, print_table, savefig
from mcmc.diagnostics import ess
from mcmc.hmc import hmc
from mcmc.models import EightSchoolsNonCentered

SEED = 20260718
N_CHAINS = 4
N_SEEDS = 5  # ESS of a slow-mixing coordinate is a high-variance estimate; average it


def _ess_per_keval(res, d):
    """ESS of coordinate d normalized by 1000 gradient evaluations (warmup
    included -- adaptation's cost is real and must be charged for)."""
    e = ess(res.samples[:, :, d])
    return 1000.0 * e / res.extras["n_grad_evals"]


def anisotropy_sweep():
    ratios = [2.0, 5.0, 10.0, 25.0, 50.0]
    rows = []
    for r in ratios:
        var = np.array([1.0, r**2])
        g = _make_diag_gaussian(var)
        x0 = np.zeros((N_CHAINS, 2))
        common = dict(
            n_samples=4_000, step_size=0.2, n_leapfrog=25, n_warmup=1_000,
            adapt_step_size=True,
        )
        eff_id, eff_ad, invm = [], [], []
        for s in range(N_SEEDS):
            # same seed for the two metrics so the ONLY difference is the metric
            res_id = hmc(g, x0, rng=np.random.default_rng(SEED + s),
                         adapt_mass=False, **common)
            res_ad = hmc(g, x0, rng=np.random.default_rng(SEED + s),
                         adapt_mass=True, **common)
            eff_id.append(_ess_per_keval(res_id, 1))  # coord 1 is the wide one
            eff_ad.append(_ess_per_keval(res_ad, 1))
            invm.append(res_ad.extras["inv_mass"][1])
        eff_id, eff_ad = float(np.mean(eff_id)), float(np.mean(eff_ad))
        rows.append({
            "ratio r": r,
            "ident ESS/keval": eff_id,
            "adapt ESS/keval": eff_ad,
            "gain x": eff_ad / eff_id,
            "inv_mass[1]": float(np.mean(invm)),
            "true var[1]": r**2,
        })
    return rows


def _make_diag_gaussian(var):
    from mcmc.targets import Gaussian
    return Gaussian(np.zeros(len(var)), np.diag(var))


def eight_schools_study():
    model = EightSchoolsNonCentered()
    z0 = 0.1 * np.random.default_rng(SEED).standard_normal((N_CHAINS, model.dim))
    common = dict(
        n_samples=20_000, step_size=0.1, n_leapfrog=20, n_warmup=2_000,
        adapt_step_size=True, target_accept=0.9,
    )
    res_id = hmc(model, z0, rng=np.random.default_rng(SEED + 1), adapt_mass=False, **common)
    res_ad = hmc(model, z0, rng=np.random.default_rng(SEED + 1), adapt_mass=True, **common)

    # mu (coord 0) is the wide one; t = log tau (coord 1) carries the funnel.
    coords = {"mu": 0, "log tau": 1, "eta_1": 2}
    rows = []
    for name, d in coords.items():
        eff_id = _ess_per_keval(res_id, d)
        eff_ad = _ess_per_keval(res_ad, d)
        rows.append({
            "param": name,
            "ident ESS/keval": eff_id,
            "adapt ESS/keval": eff_ad,
            "gain x": eff_ad / eff_id,
        })
    return rows, res_ad.extras["inv_mass"]


def make_figure(sweep_rows):
    ratios = [r["ratio r"] for r in sweep_rows]
    eff_id = [r["ident ESS/keval"] for r in sweep_rows]
    eff_ad = [r["adapt ESS/keval"] for r in sweep_rows]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(ratios, eff_ad, "o-", color="#a11", label="adapted diagonal metric")
    ax.plot(ratios, eff_id, "s--", color="0.45", label="identity metric")
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("scale ratio  r  (target sd = 1 : r)")
    ax.set_ylabel(f"wide-coord ESS / 1k gradients\n(mean of {N_SEEDS} seeds)")
    ax.set_title("Adapted metric is scale-free; identity is not")
    ax.legend()
    savefig(fig, "mass_matrix_gain.png")


def main():
    print("=== A. Anisotropy sweep: diagonal Gaussian N(0, diag(1, r^2)) ===")
    sweep = anisotropy_sweep()
    print_table(sweep, ["ratio r", "ident ESS/keval", "adapt ESS/keval",
                        "gain x", "inv_mass[1]", "true var[1]"])
    print("The adapted metric recovers the true variance (inv_mass[1] ~ r^2) and "
          "\nwhitens every target to the SAME isotropic problem -> a flat "
          "~30 ESS/keval\nregardless of r. The identity metric is at the mercy "
          "of the scale: its\nsingle step size resonates unpredictably with the "
          "wide direction (fixed-L\nHMC), so its efficiency swings from ~3 to "
          "~34 with no reliability.")
    make_figure(sweep)

    print("\n=== B. Eight schools (non-centered): ESS/keval by coordinate ===")
    es_rows, inv_mass = eight_schools_study()
    print_table(es_rows, ["param", "ident ESS/keval", "adapt ESS/keval", "gain x"])
    print(f"\nadapted inv_mass (diag M^-1): {np.array2string(inv_mass, precision=2)}")
    print("mu's entry is the largest -- the metric widens the coordinate the "
          "identity step size under-served; the diagonal cannot fix the funnel "
          "in (log tau, eta), which is why log tau's gain is modest.")


if __name__ == "__main__":
    main()
