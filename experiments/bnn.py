"""Bayesian neural network: HMC posterior predictive vs point estimate vs
deep ensemble, on the gapped-sine toy.

The question this experiment answers: when a small MLP is fit to data with a
hole in it, which method *knows* it is guessing across the hole?

- A single **MAP point estimate** (Adam to the mode of the log-posterior) has
  no epistemic uncertainty at all: its only spread is the fixed observation
  noise, so its band is a constant-width ribbon that stays just as confident
  inside the gap as on the data. It is overconfident exactly where it should
  not be.
- A **5-member deep ensemble** (the same net trained from 5 random inits) does
  widen in the gap -- different inits extrapolate differently -- and is the
  strong cheap baseline. But its spread is an ad-hoc proxy for the posterior.
- **HMC** samples the weight posterior directly, so its predictive band is the
  genuine posterior-predictive spread. It widens the most across the gap and
  stays calibrated there.

A methodological point the experiment is careful about: convergence is judged
in **function space**, never on raw weights. The weight posterior is invariant
to permuting hidden units and to sign-flipping (tanh is odd), so it is
massively multimodal and split-R-hat on any single weight coordinate is
meaningless -- we print one to show it screaming, then show R-hat on the
*predictions* (a permutation-invariant functional) sitting at ~1.

Run:  python experiments/bnn.py   (~1-2 min)
"""

import numpy as np
from common import plt, print_table, savefig

from mcmc.bnn import BayesianNNRegression, make_gapped_sine, train_map
from mcmc.diagnostics import ess, split_rhat
from mcmc.hmc import hmc

SEED = 20260703
GAP = (-0.5, 0.5)
NOISE_STD = 0.1
PRIOR_STD = 1.0
N_HIDDEN = 16


def _in_gap(x):
    return (x > GAP[0]) & (x < GAP[1])


def prediction_chains(model, samples, grid):
    """(n_chains, n_samples, n_grid): the network output at every grid point
    for every posterior draw, kept per-chain so function-space R-hat/ESS can be
    computed. ``samples`` is a ``SamplerResult.samples`` array."""
    n_chains = samples.shape[0]
    return np.stack([model.forward(samples[c], grid) for c in range(n_chains)])


def gaussian_nll(y, mean, std):
    """Mean negative log predictive density under a Gaussian N(mean, std^2)."""
    var = std**2
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (y - mean) ** 2 / var))


def coverage(y, mean, std, level=0.95):
    """Fraction of targets inside the central ``level`` predictive interval."""
    z = {0.95: 1.959963985, 0.90: 1.644853627, 0.68: 0.994457883}[level]
    return float(np.mean(np.abs(y - mean) <= z * std))


def run():
    rng = np.random.default_rng(SEED)
    X, y = make_gapped_sine(rng, n=40, noise_std=NOISE_STD, gap=GAP)
    model = BayesianNNRegression(
        X, y, n_hidden=N_HIDDEN, noise_std=NOISE_STD, prior_std=PRIOR_STD
    )

    # ---- HMC over the weight posterior -------------------------------------
    x0 = 0.1 * rng.standard_normal((8, model.dim))
    hmc_res = hmc(
        model, x0, n_samples=2000, step_size=0.01, n_leapfrog=30, rng=rng,
        n_warmup=1500, adapt_step_size=True, target_accept=0.9,
    )

    # ---- MAP point estimate and 5-member deep ensemble ---------------------
    point = train_map(model, 0.1 * rng.standard_normal((1, model.dim)),
                      n_steps=3000, lr=0.01)
    ensemble = train_map(model, 0.7 * rng.standard_normal((5, model.dim)),
                        n_steps=3000, lr=0.01)

    # ---- held-out test set spanning the whole range (incl. the gap) --------
    X_test = rng.uniform(-2.0, 2.0, size=400)
    y_test = np.sin(3.0 * X_test) + NOISE_STD * rng.standard_normal(X_test.size)

    return model, (X, y), (X_test, y_test), hmc_res, point, ensemble


def predict(model, hmc_res, point, ensemble, grid):
    """Predictive mean/std for each method on ``grid``. std includes the
    observation-noise variance so all three are bands for a *new observation*
    and the calibration comparison is apples-to-apples."""
    nv = model.noise_var

    hmc_mean, hmc_std = model.posterior_predictive(hmc_res.samples, grid,
                                                   include_noise=True)

    p = model.forward(point, grid)[0]  # single net: epistemic std = 0
    point_mean, point_std = p, np.full_like(p, np.sqrt(nv))

    e = model.forward(ensemble, grid)  # (5, n_grid)
    ens_mean = e.mean(axis=0)
    ens_std = np.sqrt(e.var(axis=0) + nv)

    return {
        "HMC (posterior)": (hmc_mean, hmc_std),
        "deep ensemble (5)": (ens_mean, ens_std),
        "point estimate (MAP)": (point_mean, point_std),
    }


def mass_matrix_study(model, grid, n_samples=2000, n_warmup=1500, n_seeds=3):
    """Identity metric vs an adapted diagonal metric on the weight posterior.

    Section 7 measures diagonal mass-matrix adaptation on targets whose scales
    are known (a diagonal Gaussian, eight schools). This is the same knob on a
    posterior nobody wrote down: 49 weights whose marginal scales are set by
    where each hidden unit happened to land, not by a modeller.

    The two arms are the *same* run apart from ``adapt_mass``: paired seeds,
    identical initial points, identical budget and dual-averaging target.
    Efficiency is reported in function space (raw-weight ESS is symmetry junk,
    as above) and per 1000 gradient evaluations, since the metric costs no
    extra gradients. Repeated over ``n_seeds`` because a single ESS estimate on
    a slow coordinate is high-variance -- the same reason experiment 9 averages
    its cells -- and reported as the median across seeds.
    """
    out = []
    for adapt in (False, True):
        rows = []
        for s in range(n_seeds):
            rng = np.random.default_rng(SEED + 1 + s)
            x0 = 0.1 * rng.standard_normal((8, model.dim))
            res = hmc(
                model, x0, n_samples=n_samples, step_size=0.01, n_leapfrog=30,
                rng=rng, n_warmup=n_warmup, adapt_step_size=True,
                target_accept=0.9, adapt_mass=adapt,
            )
            pred = prediction_chains(model, res.samples, grid)
            e = np.array([ess(pred[:, :, j]) for j in range(grid.size)])
            rows.append((
                res.extras["step_size"], float(res.accept_rate.mean()),
                res.extras["n_divergent"], float(e.min()), float(np.median(e)),
                float(np.median(e) / res.extras["n_grad_evals"] * 1000),
                float(res.extras["inv_mass"].max() / res.extras["inv_mass"].min()),
            ))
        med = np.median(np.array(rows), axis=0)
        out.append({
            "metric": "adapted diagonal" if adapt else "identity",
            "step": med[0], "accept": med[1],
            "diverg": int(sum(r[2] for r in rows)),
            "ESS_min": med[3], "ESS_med": med[4], "ESS/1k_grad": med[5],
            "scale_spread": med[6],
        })
    return out


def main():
    model, (X, y), (X_test, y_test), hmc_res, point, ensemble = run()
    print(f"HMC mean accept {hmc_res.accept_rate.mean():.2f}, "
          f"step {hmc_res.extras['step_size']:.4f}, "
          f"divergences {hmc_res.extras['n_divergent']}")

    # ---- function-space vs weight-space convergence ------------------------
    grid = np.linspace(-2.0, 2.0, 200)
    pchains = prediction_chains(model, hmc_res.samples, grid)  # (C, S, G)
    pred_rhat = np.array([split_rhat(pchains[:, :, j]) for j in range(grid.size)])
    pred_ess = np.array([ess(pchains[:, :, j]) for j in range(grid.size)])
    # the same statistic on raw weight coordinates is meaningless (multimodal)
    w_rhat = np.array([split_rhat(hmc_res.samples[:, :, d])
                       for d in range(model.dim)])
    print("\nConvergence -- weight space is meaningless, function space is not:")
    print(f"  raw weight split-R-hat      : median {np.median(w_rhat):.2f}, "
          f"max {w_rhat.max():.2f}   (symmetry-broken junk)")
    print(f"  prediction split-R-hat      : median {np.median(pred_rhat):.3f}, "
          f"max {pred_rhat.max():.3f}")
    print(f"  prediction ESS              : min {pred_ess.min():.0f}, "
          f"median {np.median(pred_ess):.0f}  (of {8 * 2000} draws)")

    # ---- does a diagonal metric help this 49-dim posterior? ----------------
    mm = mass_matrix_study(model, grid)
    print("\nDiagonal mass matrix on the weight posterior (Sec. 7's knob, "
          "here on a posterior nobody wrote down):")
    print_table(mm, ["metric", "step", "accept", "diverg",
                     "ESS_min", "ESS_med", "ESS/1k_grad", "scale_spread"])

    # ---- calibration on held-out points, split observed vs gap -------------
    bands = predict(model, hmc_res, point, ensemble, X_test)
    gap_mask = _in_gap(X_test)
    print("\nHeld-out calibration (400 points; 95% target coverage):")
    rows = []
    for name, (mean, std) in bands.items():
        for region, mask in [("observed", ~gap_mask), ("gap", gap_mask)]:
            rows.append({
                "method": name,
                "region": region,
                "cover95": coverage(y_test[mask], mean[mask], std[mask]),
                "nll": gaussian_nll(y_test[mask], mean[mask], std[mask]),
                "mean_std": float(std[mask].mean()),
            })
    print_table(rows, ["method", "region", "cover95", "nll", "mean_std"])

    # ---- figure: the three predictive bands --------------------------------
    gbands = predict(model, hmc_res, point, ensemble, grid)
    truth = np.sin(3.0 * grid)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, (name, (mean, std)) in zip(axes, gbands.items()):
        ax.axvspan(GAP[0], GAP[1], color="0.92", zorder=0)
        ax.fill_between(grid, mean - 1.96 * std, mean + 1.96 * std,
                        color="C0", alpha=0.25, lw=0)
        ax.plot(grid, truth, "k--", lw=1, label="truth")
        ax.plot(grid, mean, "C0", lw=1.5, label="pred. mean")
        ax.plot(X, y, "o", ms=3, color="0.35", label="train")
        ax.set_title(f"{name}\n95% band width in gap: "
                     f"{2 * 1.96 * std[_in_gap(grid)].mean():.2f}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle(
        "Predictive uncertainty over a data gap (grey). The point estimate "
        "stays overconfident across the gap; the ensemble widens; HMC widens "
        "most and stays calibrated.", fontsize=9,
    )
    savefig(fig, "bnn_predictive.png")


if __name__ == "__main__":
    main()
