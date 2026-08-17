"""SGHMC measured: the closed form, the price of the correction, and what
momentum actually buys at matched cost.

``mcmc/sghmc.py`` derives the exact stationary covariance of the sampler on a
Gaussian, so on that target every claim here is scored against a number rather
than against another sampler's opinion. Three questions, in order of how much
they assume:

1. **Does the implementation match its own derivation, across the range?**
   The step is swept over a decade at fixed friction, in three arms -- exact
   gradient, noisy gradient uncorrected, noisy gradient corrected -- and each
   arm is compared to the closed form for *that* arm. An implementation that
   matched the target instead of the prediction would be the broken one.

2. **Is momentum worth anything at matched cost?** SGLD and SGHMC both spend
   exactly one gradient per step, so cost matching is step matching. The fair
   comparison holds the *bias* fixed rather than the step size: each sampler's
   step is solved so that its exact stationary variance is 1% too wide, and
   what is compared is effective sample size per gradient evaluation. Friction
   is swept, because it is the knob that interpolates between Langevin
   (gamma large: momentum forgotten every step) and Hamiltonian dynamics
   (gamma small: ballistic, but nothing removes the discretization's energy).

3. **What does the friction condition cost on a real posterior?** The
   correction needs the minibatch gradient-noise variance, and the condition
   ``2 gamma >= h Vhat`` caps the step. On this repo's BNN posterior that cap
   is measured, at three batch sizes, and compared with the step SGLD actually
   uses there -- the same posterior on which ``mcmc/sgld.py`` found its
   noise-domination crossover to sit *below* any usable step.

Not here, and named rather than implied: the sampling bias of either
unadjusted sampler against exact HMC *in function space* on that BNN posterior.
The weight posterior is invariant to permuting hidden units and to sign flips,
so a weight-space comparison is meaningless (``experiments/bnn.py`` shows
split-R-hat screaming on a raw coordinate), and doing it properly is a
predictive-band study, not a line of this script.

Run:  python experiments/sghmc.py   (~11 s)
"""

import numpy as np
from common import plt, print_table, savefig

from mcmc.bnn import BayesianNNRegression, make_gapped_sine, train_map
from mcmc.diagnostics import ess
from mcmc.sgld import sgld, ula_gaussian_variance
from mcmc.sghmc import estimate_grad_noise, sghmc, sghmc_gaussian_cov
from mcmc.targets import Gaussian

SEED = 20260817
TARGET_VAR = 1.0
FRICTION = 1.0
STEPS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6)
NOISE_VAR = 4.0          # the "minibatch" noise used in the noisy arms
N_CHAINS = 512
N_SAMPLES = 8000
N_WARMUP = 2000


class NoisyGaussian:
    """``N(0, s^2)`` with a gradient estimator of known noise variance.

    The same device ``tests/test_sghmc.py`` uses: nothing is subsampled, so V
    is a number chosen here and the prediction is exact. A real minibatch
    gradient is only assumed to be unbiased with some variance, which is
    exactly what this is.
    """

    def __init__(self, var=TARGET_VAR, noise_var=NOISE_VAR):
        self.var, self.noise_var = float(var), float(noise_var)

    def grad_logpdf(self, x):
        return -np.asarray(x, dtype=float) / self.var

    def grad_logpdf_minibatch(self, x, batch_size, rng):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return self.grad_logpdf(x) + np.sqrt(self.noise_var) * rng.standard_normal(
            x.shape
        )


# ---------------------------------------------------------------------------
# 1. the sampler against its own closed form, in three arms
# ---------------------------------------------------------------------------
ARMS = (
    ("exact gradient", 0.0, 0.0),
    ("noisy, uncorrected", NOISE_VAR, 0.0),
    ("noisy, corrected", NOISE_VAR, NOISE_VAR),
)


def closed_form_study(steps=STEPS, friction=FRICTION, seed=SEED):
    """Observed vs predicted stationary variance, per arm, per step."""
    rows = []
    for label, v_true, v_hat in ARMS:
        target = (Gaussian(mean=[0.0], cov=[[TARGET_VAR]]) if v_true == 0
                  else NoisyGaussian(noise_var=v_true))
        batch = None if v_true == 0 else 1
        for h in steps:
            # The corrected arm cannot be run at every step: the condition
            # 2 gamma >= h Vhat caps it at 2 gamma / Vhat, and the cap is the
            # method's, not this script's -- so the refusal is printed and the
            # cell is left out rather than the arm being quietly re-tuned.
            try:
                predicted = sghmc_gaussian_cov(h, friction, TARGET_VAR,
                                               grad_noise_var=v_true,
                                               est_noise_var=v_hat)[0, 0]
            except ValueError as exc:
                print(f"  {label:<20s} h={h:.3f}  refused: {exc}", flush=True)
                continue
            rng = np.random.default_rng(seed)
            res = sghmc(target, np.zeros((N_CHAINS, 1)), n_samples=N_SAMPLES,
                        step_size=h, friction=friction, rng=rng,
                        n_warmup=N_WARMUP, batch_size=batch, est_noise_var=v_hat)
            observed = float(res.pooled().var())
            rows.append({
                "arm": label, "step": f"{h:.3f}",
                "predicted": f"{predicted:.6f}", "observed": f"{observed:.6f}",
                "ratio": f"{observed / predicted:.4f}",
                "excess_over_target": f"{predicted / TARGET_VAR - 1:.6f}",
            })
            print(f"  {label:<20s} h={h:.3f}  predicted {predicted:.5f}  "
                  f"observed {observed:.5f}  ratio {observed / predicted:.4f}",
                  flush=True)
    return rows


# ---------------------------------------------------------------------------
# 2. momentum at matched cost: step solved for equal bias, then ESS per gradient
# ---------------------------------------------------------------------------
BIAS = 0.01              # each sampler is tuned to a stationary variance 1% wide
FRICTIONS = (0.25, 0.5, 1.0, 2.0, 4.0)


def _solve_step(variance_of, target_excess=BIAS, lo=1e-4, hi=1.9):
    """Bisect for the step whose *exact* stationary variance is 1 + excess.

    Both samplers have a closed form for the variance they converge to, so the
    matching is done on the exact bias rather than on a simulated estimate of
    it -- no Monte Carlo enters the tuning, only the comparison.
    """
    want = TARGET_VAR * (1.0 + target_excess)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        try:
            val = variance_of(mid)
        except ValueError:          # past the stability limit
            hi = mid
            continue
        if val > want:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def matched_cost_study(frictions=FRICTIONS, seed=SEED, n_samples=20_000,
                       n_warmup=5000, n_chains=64):
    """SGLD vs SGHMC at equal bias and equal gradient evaluations.

    ESS is computed per chain and summed, then divided by the gradient
    evaluations the run spent, so the reported quantity is effective samples
    per gradient -- the only currency in which a one-gradient-per-step sampler
    can be compared with another one.
    """
    rows = []
    eps = _solve_step(lambda e: ula_gaussian_variance(e, TARGET_VAR))
    rng = np.random.default_rng(seed)
    res = sgld(Gaussian(mean=[0.0], cov=[[TARGET_VAR]]), np.zeros((n_chains, 1)),
               n_samples=n_samples, step_size=eps, rng=rng, n_warmup=n_warmup)
    rows.append(_ess_row("SGLD", "-", eps, res, n_chains))

    for g in frictions:
        h = _solve_step(lambda s: sghmc_gaussian_cov(s, g, TARGET_VAR)[0, 0])
        rng = np.random.default_rng(seed)
        res = sghmc(Gaussian(mean=[0.0], cov=[[TARGET_VAR]]),
                    np.zeros((n_chains, 1)), n_samples=n_samples, step_size=h,
                    friction=g, rng=rng, n_warmup=n_warmup)
        rows.append(_ess_row("SGHMC", f"{g:g}", h, res, n_chains))
    return rows


def _ess_row(name, friction, step, res, n_chains):
    draws = res.samples[:, :, 0]                      # (n_chains, n_samples)
    n_eff = ess(draws)
    per_grad = n_eff / res.extras["n_grad_evals"]
    observed = float(res.pooled().var())
    # The variance estimate's own error, from the effective sample size the
    # same run reports: sd(s^2) = s^2 sqrt(2 / ESS) for a Gaussian. Without it
    # "1.0155 against a targeted 1.0100" reads as a discrepancy rather than as
    # 1.3 standard errors.
    se = observed * np.sqrt(2.0 / n_eff)
    row = {"sampler": name, "friction": friction, "step": f"{step:.6f}",
           "ess": f"{n_eff:.1f}", "ess_per_grad": f"{per_grad:.5f}",
           "observed_var": f"{observed:.5f}", "var_stderr": f"{se:.5f}",
           "grad_evals": res.extras["n_grad_evals"]}
    print(f"  {name:<6s} friction {friction:>4s}  step {step:.4f}  "
          f"ESS {n_eff:9.1f}  per gradient {per_grad:.5f}  "
          f"var {observed:.4f} +- {se:.4f}", flush=True)
    return row


# ---------------------------------------------------------------------------
# 3. the friction condition on a real posterior
# ---------------------------------------------------------------------------
BATCHES = (10, 20, 50)
N_DATA = 200
N_HIDDEN = 16


def bnn_noise_study(batches=BATCHES, seed=SEED, n_probe=200):
    """What the friction condition costs on this repo's BNN posterior.

    Two constraints, and it is their *intersection* that matters:

    - the correction needs ``2 gamma >= h Vhat``, i.e. ``h <= 2 gamma / Vhat``;
    - the momentum recursion ``v <- (1 - h gamma) v + ...`` needs
      ``h gamma < 2``, or ``|1 - h gamma| > 1`` and the velocity amplifies
      every step regardless of the target.

    Raising gamma to afford the correction therefore runs into the second
    condition, and eliminating gamma between them leaves a bound on the step
    alone: ``h < 2 / sqrt(Vhat)``, with gamma then pinned into
    ``[h Vhat / 2, 2/h)``. That bound involves no property of the target at
    all -- only the gradient noise -- which is why it is worth reporting for a
    real posterior.

    Vhat is the *largest* per-coordinate minibatch-gradient variance, because
    the sampler applies one scalar correction to every coordinate and the
    condition has to hold for the worst of them. It is measured at two points,
    a draw near the prior and a MAP fit, since the noise is a local quantity
    and a badly-fit point has large residuals: if the two disagreed by orders
    of magnitude the headline would belong to the point, not to the posterior.
    """
    rng = np.random.default_rng(seed)
    X, y = make_gapped_sine(rng, n=N_DATA)
    model = BayesianNNRegression(X, y, n_hidden=N_HIDDEN)
    points = {
        "prior draw": 0.1 * rng.standard_normal((8, model.dim)),
        "MAP fit": train_map(model, 0.1 * rng.standard_normal((8, model.dim)),
                             n_steps=3000, lr=0.01),
    }

    rows = []
    for where, theta in points.items():
        for b in batches:
            v = estimate_grad_noise(model, theta, b, rng, n_probe=n_probe)
            worst, mean = float(v.max()), float(v.mean())
            rows.append({
                "where": where, "batch": b, "n_data": N_DATA, "dim": model.dim,
                "worst_coord_var": f"{worst:.6e}",
                "mean_coord_var": f"{mean:.6e}",
                "max_step_gamma1": f"{2.0 / worst:.6e}",
                "max_step_any_gamma": f"{2.0 / np.sqrt(worst):.6e}",
            })
            print(f"  {where:<11s} batch {b:3d}/{N_DATA}:  worst coordinate "
                  f"variance {worst:.3e} (mean {mean:.3e})  ->  h <= "
                  f"{2.0 / worst:.2e} at gamma=1, and h < "
                  f"{2.0 / np.sqrt(worst):.2e} at any gamma", flush=True)
    return rows


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(closed_rows, matched_rows):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

    ax = axes[0]
    # The corrected arm is drawn dashed and on top: its closed form is the
    # exact-gradient one *identically*, so a solid line would simply hide the
    # curve it is supposed to be compared with.
    styles = {"exact gradient": ("C0", "-", 2.0),
              "noisy, uncorrected": ("C3", "-", 1.3),
              "noisy, corrected": ("C2", "--", 1.3)}
    for label, (color, ls, lw) in styles.items():
        cells = [r for r in closed_rows if r["arm"] == label]
        h = [float(r["step"]) for r in cells]
        ax.plot(h, [float(r["predicted"]) for r in cells], ls,
                color=color, lw=lw, label=f"{label} (closed form)")
        ax.plot(h, [float(r["observed"]) for r in cells], "o", ms=4,
                color=color, mfc="none")
    ax.axhline(TARGET_VAR, color="0.4", ls=":", lw=1)
    ax.set_xlabel("step size $h$")
    ax.set_ylabel(r"stationary Var$[\theta]$")
    ax.set_title(f"lines: closed form, circles: sampled "
                 f"($\\gamma$ = {FRICTION:g}, $V$ = {NOISE_VAR:g})")
    ax.annotate("correction refused above $h = 2\\gamma/V$", xy=(0.45, 1.30),
                fontsize=7, color="C2", ha="center")
    ax.legend(fontsize=7)

    ax = axes[1]
    sgld_rows = [r for r in matched_rows if r["sampler"] == "SGLD"]
    sghmc_rows = [r for r in matched_rows if r["sampler"] == "SGHMC"]
    g = [float(r["friction"]) for r in sghmc_rows]
    ax.plot(g, [float(r["ess_per_grad"]) for r in sghmc_rows], "o-", color="C0",
            ms=5, label="SGHMC")
    if sgld_rows:
        ax.axhline(float(sgld_rows[0]["ess_per_grad"]), color="C3", ls="--",
                   lw=1.2, label="SGLD")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"friction $\gamma$")
    ax.set_ylabel("effective samples per gradient")
    ax.set_title(f"matched cost, matched bias ({100 * BIAS:g}% wide)")
    ax.legend(fontsize=8)

    fig.suptitle("SGHMC: against its closed form, and against SGLD at equal cost",
                 y=1.02, fontsize=10)
    savefig(fig, "sghmc.png")


def main():
    print("1. sampler vs closed form, three arms")
    closed_rows = closed_form_study()

    print("\n2. matched cost, matched bias: SGLD vs SGHMC")
    matched_rows = matched_cost_study()

    print("\n3. the friction condition on the BNN posterior")
    bnn_rows = bnn_noise_study()

    print()
    print_table(matched_rows,
                ["sampler", "friction", "step", "ess_per_grad", "observed_var",
                 "var_stderr"])
    print()
    print_table(bnn_rows, ["where", "batch", "worst_coord_var",
                           "max_step_gamma1", "max_step_any_gamma"])
    make_figure(closed_rows, matched_rows)


if __name__ == "__main__":
    main()
