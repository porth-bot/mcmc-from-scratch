# MCMC from scratch

![ci](https://github.com/porth-bot/mcmc-from-scratch/actions/workflows/ci.yml/badge.svg)

Metropolis–Hastings, Gibbs, and Hamiltonian Monte Carlo implemented in pure
NumPy — no PyMC, no Stan, no autograd — and **validated against exact
answers** at every level: hand-derived gradients against finite differences,
sampler moments against closed-form posteriors, the ESS estimator against the
AR(1) closed form, and (where no closed form exists) two independent
inference routes against each other.

![funnel](figures/funnel_scatter.png)

*Neal's funnel, 10D. True marginal: $v \sim N(0, 3^2)$. A random walk never
reaches the neck (sd$[v]$ = 2.22), gradient-guided HMC gets closer but
diverges in the neck (sd = 2.60, 6% divergent), and reparameterizing the
geometry solves the problem outright (sd = 2.99, zero divergences). Fixing
the geometry beats tuning the sampler.*

## Problem

Bayesian inference needs expectations under a posterior known only up to a
constant: $\pi(\theta) \propto p(y \mid \theta)\,p(\theta)$. MCMC builds a
Markov chain with stationary distribution $\pi$ using only density *ratios*,
so the intractable normalizer cancels. This repo implements the three
classical kernels, the diagnostics needed to trust them, and experiments
designed so that **every claim has a ground truth or a cross-check**.

Full derivations (detailed balance → MH → Gibbs-as-MH → the HMC involution
argument → dual averaging → ESS/R-hat) are in
[`theory/derivations.md`](theory/derivations.md). The short version of why
HMC works:

$$\pi(x, p) \propto e^{-H(x,p)}, \quad H = -\log\tilde\pi(x) + \tfrac12\lVert p\rVert^2$$

Hamiltonian flow conserves $H$, preserves phase-space volume (Liouville), and
is reversible — an exact-flow proposal would always be accepted. The leapfrog
integrator keeps volume preservation and reversibility *exactly* (each
substep is a unit-Jacobian shear; the composition is palindromic) and loses
only energy conservation to $O(\varepsilon^2)$, which a Metropolis step with
$\alpha = \min(1, e^{-\Delta H})$ repairs. Both structural properties are
pinned by tests: reversibility to $10^{-10}$, and a measured ~4× drop in peak
$|\Delta H|$ when $\varepsilon$ is halved at fixed trajectory time.

## What's implemented

| Module | Contents |
|---|---|
| [`mcmc/metropolis.py`](mcmc/metropolis.py) | Random-walk MH, log-space accept, batched chains |
| [`mcmc/gibbs.py`](mcmc/gibbs.py) | Systematic- **or** random-scan driver over state dicts + Gaussian full conditionals derived via the precision matrix. At matched work, systematic scan is ~2× more efficient than random scan on the correlated Gaussian ([`experiments/gibbs_scan.py`](experiments/gibbs_scan.py)) — random scan can leave a coordinate stale for a sweep |
| [`mcmc/hmc.py`](mcmc/hmc.py) | Leapfrog, HMC with jittered trajectory length, dual-averaging warmup (Hoffman & Gelman 2014, Alg. 5), optional **diagonal mass-matrix adaptation** from windowed warmup variances (Stan-style memoryless windows — a per-axis preconditioner so one step size fits an axis-aligned target of unequal scales), divergence tracking |
| [`mcmc/nuts.py`](mcmc/nuts.py) | No-U-Turn Sampler (multinomial, Betancourt 2017): recursive doubling with the generalized U-turn criterion, canonical (multinomial) state selection, gradient-cached leapfrog (one gradient per leaf), max-depth **and** per-iteration divergence handling, same dual-averaging warmup — HMC with the trajectory-length knob removed. ~4–6× the ESS per gradient of hand-tuned fixed-$L$ HMC (§9) |
| [`mcmc/mala.py`](mcmc/mala.py) | Metropolis-adjusted Langevin: one gradient-drift Euler step with the full asymmetric Hastings correction — RWMH plus a score-driven drift, and the exact bridge toward score-based diffusion (unadjusted annealed Langevin is this proposal minus the accept step) |
| [`mcmc/tempering.py`](mcmc/tempering.py) | Parallel tempering (replica exchange): geometric temperature ladder, even/odd swap moves, per-pair swap-rate diagnostics — for multimodal targets |
| [`mcmc/diagnostics.py`](mcmc/diagnostics.py) | FFT autocorrelation, $\tau_{\text{int}}$ via Geyer initial monotone sequence, bulk ESS, tail ESS (Vehtari et al. 2021 — min over the 5%/95% tail-indicator ESSs, so a poorly-explored tail is flagged even when the bulk mixes), classic split-$\hat R$ **and** rank-normalized split-$\hat R$ (Vehtari et al. 2021 — Blom rank-normal transform + a folded term for scale, robust on heavy-tailed targets where the variance-based statistic reads a false 1; §8), compute-normalized efficiency (ESS per second / per evaluation), and `thinning_variance_ratio` — the closed-form price of thinning an AR(1) chain, $R = k(1+\rho^k)(1-\rho)/[(1-\rho^k)(1+\rho)] \ge 1$, proved and measured in [theory](theory/derivations.md) §6.3 (thinning never improves accuracy; it costs most when the chain mixes *well*) |
| [`mcmc/targets.py`](mcmc/targets.py) | Correlated Gaussians, Neal's funnel, Rosenbrock, Student-t, Gaussian mixtures — with analytic gradients and exact reference samplers |
| [`mcmc/models.py`](mcmc/models.py) | Conjugate Bayesian linear regression (closed-form posterior as answer key); eight schools with conjugate Gibbs conditionals *and* a non-centered HMC parameterization with hand-derived, Jacobian-corrected gradients |
| [`mcmc/bnn.py`](mcmc/bnn.py) | Bayesian neural network (1-hidden-layer tanh MLP) with hand-written backprop log-posterior gradient, sampled by HMC; plus an Adam MAP/deep-ensemble trainer sharing the same model and objective |

All log-densities are batched over chains, so 4 chains advance in lockstep as
one NumPy computation. Everything is seeded and reproducible.

## Results

### 1. Exact-posterior validation (`experiments/validate_exact.py`)

Correlated Gaussian ($\rho = 0.9$), 4 chains, overdispersed starts:

| sampler | draws | accept | max mean err | rel cov err | $\tau(x_0)$ | ESS$(x_0)$ | ESS/1k evals | $\hat R$ |
|---|---|---|---|---|---|---|---|---|
| RWMH | 160k | 0.50 | 0.027 | 0.002 | 67.8 | 2 358 | 14.7 | 1.000 |
| Gibbs | 160k | 1.00 | 0.009 | 0.003 | 9.5 | 16 778 | 52.4 | 1.000 |
| HMC | 40k | 0.82 | 0.018 | 0.013 | **1.8** | **21 760** | 26.0 | 1.000 |

All three reproduce the exact moments. The efficiency story has a nuance
worth stating precisely: **per draw**, HMC dominates ($\tau$ 37× smaller than
RWMH); **per density evaluation**, exact-conditional Gibbs wins on this
target — when conjugacy hands you the full conditionals, use them. On the
conjugate linear-regression posterior (samplers see only the unnormalized
density), sampled means match the closed form to $\le 0.003$:

<p align="center"><img src="figures/linreg_posterior.png" width="420"></p>

<p align="center"><img src="figures/gaussian_autocorr.png" width="520"></p>

### 2. Neal's funnel (`experiments/funnel.py`)

True $v$-marginal is $N(0, 3^2)$ exactly — so bias is measurable:

| sampler | draws | E$[v]$ (true 0) | sd$[v]$ (true 3) | $\tau(v)$ | ESS$(v)$ | $\hat R(v)$ | divergent |
|---|---|---|---|---|---|---|---|
| RWMH | 400k | 0.53 | 2.22 | 2361 | 169 | 1.04 | — |
| HMC (centered) | 100k | 0.49 | 2.60 | 190 | 528 | 1.02 | 5 998 |
| HMC (non-centered) | 100k | **0.02** | **2.99** | 12.5 | 8 001 | 1.000 | 0 |

Two honest lessons the numbers force on you: (1) $\hat R = 1.04$ while
missing the neck entirely — $\hat R \approx 1$ is *necessary, not
sufficient*; only the exact marginal exposes the bias. (2) The centered HMC
divergences aren't noise to suppress; they're the sampler reporting the
region it cannot enter. The non-centered change of variables
$x_i = e^{v/2} z_i$ makes the target an independent Gaussian (the Jacobian
cancels the varying scale exactly — derivation in Sec. 4.6), and every
pathology disappears.

<p align="center"><img src="figures/funnel_v_marginal.png" width="520"></p>

### 3. Real data: eight schools (`experiments/eight_schools.py`)

Rubin's (1981) SAT coaching study under the hierarchical model
$y_j \sim N(\theta_j, \sigma_j^2)$, $\theta_j \sim N(\mu, \tau^2)$,
$p(\mu) \propto 1$, $\tau^2 \sim \text{InvGamma}(1, 1)$. No closed form
exists, so correctness rests on **two independent routes agreeing**:
conjugate Gibbs on the centered parameterization vs HMC on the non-centered
one (different parameterizations, different kernels, different code paths).

Result: all 10 posterior means agree to **0.131** (posterior sds are ~4.3,
so this is within Monte Carlo error), $\hat R \le 1.002$ everywhere. HMC's
ESS on $\mu$ is 63k from 80k draws vs Gibbs's 1.5k from 160k — the centered
Gibbs chain suffers exactly the $\mu$–$\theta$ coupling that non-centering
removes.

<p align="center"><img src="figures/eight_schools_agreement.png" width="560"></p>
<p align="center"><img src="figures/eight_schools_shrinkage.png" width="560"></p>

**Prior sensitivity, stated plainly:** the InvGamma(1,1) prior on $\tau^2$
was chosen to keep all Gibbs conditionals conjugate, and it concentrates
$\tau$ near ~1.5, i.e. strong pooling. The classic half-Cauchy analysis
(Gelman 2006) is far more diffuse in $\tau$. With $J = 8$ noisy groups the
data genuinely cannot pin $\tau$ down, so the prior matters — both routes
share the prior, which is what makes their agreement a valid check of the
*samplers* rather than a claim about the *science*.

### 4. Multimodal targets: parallel tempering (`experiments/tempering.py`)

Every sampler above assumes it can reach the whole distribution. On a
well-separated mixture that assumption breaks: the barrier between modes is
crossed with exponentially small probability, so a single chain reports
whichever mode it started in. Two Gaussians 12 units apart (weights
0.35 / 0.65), **both samplers started entirely in the left mode**:

| sampler | E$[x_0]$ (true 1.8) | left-mode frac (true 0.35) |
|---|---|---|
| single random walk | −6.0 | 1.00 (never crossed) |
| parallel tempering (8 replicas) | **1.82** | **0.35** |

Parallel tempering runs replicas at inverse temperatures $\beta_k$ from 1 down
to 0.01; the hot replicas roam freely across the flattened landscape and
adjacent-replica swaps ferry that mobility down to the cold ($\beta=1$) chain.
Swap acceptance holds at ~0.7 across the ladder, so the mode-hopping actually
reaches the bottom.

<p align="center"><img src="figures/tempering_bimodal.png" width="620"></p>

### 5. A real posterior: Bayesian neural network (`experiments/bnn.py`)

Every target above is a hand-written density. This one is a *model*: the
unknown is the full weight vector of a small tanh MLP (`mcmc/bnn.py`,
$3H+1 = 49$ dimensions at $H=16$), and the target is its Bayesian posterior —
Gaussian likelihood, isotropic Gaussian prior. The log-posterior gradient is a
backprop pass written out by hand and checked against finite differences, so
HMC is running on exactly the quantity training would compute. The data is
$\sin(3x)$ on $[-2, 2]$ **with a gap cut out of the middle**; the question is
which method reports that it is guessing across the gap.

Three predictive bands on the same model — HMC (samples the posterior), a
5-member deep ensemble (the same net from 5 random inits), and a single Adam
MAP point estimate:

<p align="center"><img src="figures/bnn_predictive.png" width="900"></p>

Held-out calibration on 400 fresh points, split into the observed region and
the gap (95% target coverage):

| method | region | 95% coverage | mean NLL | mean pred. std |
|---|---|---|---|---|
| HMC (posterior) | observed | 0.94 | −0.67 | 0.11 |
| HMC (posterior) | **gap** | **1.00** | **0.08** | **0.24** |
| deep ensemble (5) | observed | 0.88 | −0.50 | 0.11 |
| deep ensemble (5) | **gap** | 0.57 | 0.84 | 0.14 |
| point estimate (MAP) | observed | 0.91 | −0.63 | 0.10 |
| point estimate (MAP) | **gap** | **0.30** | **2.47** | 0.10 |

The point estimate has no epistemic uncertainty — its band is a constant-width
noise ribbon, so it stays just as confident inside the gap (coverage collapses
to 0.30, NLL blows up to 2.47). The deep ensemble widens and is the strong
cheap baseline, but still under-covers the gap (0.57). HMC widens the most and
stays calibrated (1.00 / NLL 0.08).

**Convergence is judged in function space, on purpose.** The weight posterior
is invariant to permuting hidden units and to sign-flipping (tanh is odd), so
it is massively multimodal and split-$\hat R$ on a raw weight coordinate is
meaningless — measured here, median 1.55 and up to 2.58 across coordinates.
The *predictions* are a permutation-invariant functional of the weights, and
their split-$\hat R$ sits at 1.02 (max 1.08) with ESS in the hundreds. Always
diagnose the quantity you care about, not the raw parameters.

### 6. External benchmark: ours vs emcee (`experiments/external_benchmark.py`)

Every section above validates against an *exact answer*. This one validates
against another *sampler*: [emcee](https://emcee.readthedocs.io) (Foreman-Mackey
et al. 2013), the widely used affine-invariant ensemble sampler. emcee is
gradient-free and its stretch move is invariant under affine reparameterization
— which is the entire benchmark. It is run **vectorized** on our batched
`logpdf` (same NumPy-over-an-ensemble computation as ours, so the wall-clock gap
is algorithmic), ESS is computed with *our* estimator for every sampler, and
"evaluations" counts every call touching the whole model over the full run —
a density eval (RWMH/emcee), a full-conditional draw (Gibbs), or a gradient eval
(HMC), with emcee's counted exactly by wrapping its log-prob.

**Correlated Gaussian** ($\rho = 0.9$), worst-dimension ESS:

| sampler | grad? | draws | min ESS | ESS / 1k evals | $\hat R$ |
|---|---|---|---|---|---|
| RWMH (ours) | no | 160k | 2 112 | 12.6 | 1.001 |
| Gibbs (ours) | no | 160k | 16 778 | **49.9** | 1.000 |
| HMC (ours) | yes | 40k | **20 608** | 24.6 | 1.000 |
| emcee (stretch) | no | 256k | 8 050 | 27.9 | 1.003 |

Per evaluation, emcee's affine-invariant stretch beats the naive
coordinate-wise random walk with **zero tuning** (27.9 vs 12.6 — the
correlation an affine map removes costs it nothing) and even edges HMC's
per-eval number. But recall an HMC evaluation is a *gradient*, the rest are
*densities* (a gradient costs a constant factor more — the honest asterisk on
the per-eval column). On a target this cheap and low-dimensional, exact-
conditional Gibbs wins outright — both per evaluation and, at ~38k ESS/s,
per wall-clock second. No single method leads on every axis.

**Eight schools** (10-dim), ESS on $\mu$ and the hard funnel-neck coordinate $\tau$:

| sampler | grad? | ESS($\mu$) | ESS($\tau$) | ESS($\tau$) / 1k evals | $\hat R(\tau)$ |
|---|---|---|---|---|---|
| Gibbs (ours, centered) | no | 601 | 6 690 | **7.6** | 1.001 |
| HMC (ours, non-centered) | yes | **30 274** | 5 872 | 6.4 | 1.001 |
| emcee (stretch, non-centered) | no | 6 845 | 6 035 | 6.9 | 1.004 |

HMC is the only sampler uniformly efficient across all ten coordinates —
gradient plus non-centering give ESS($\mu$) ~30k — but that took a hand-derived,
Jacobian-corrected gradient *and* the reparameterization, and it still logged
~1% divergences in the neck. **emcee's real case is here:** with no gradient and
no reparameterization, it reaches ESS comparable to HMC on the hard $\tau$
coordinate and balanced ESS elsewhere, for the price of writing down the
log-density alone. Centered Gibbs is fastest per second but its $\mu$–$\theta$
coupling wrecks ESS($\mu$) (~600) — the same coupling non-centering removes.

<p align="center"><img src="figures/external_benchmark.png" width="820"></p>

The honest summary: **use the gradient when you have it and the dimension isn't
tiny** (HMC's uniform, high per-coordinate ESS), **use conjugacy when you have
it** (Gibbs's cheap exact conditionals), and **reach for a gradient-free
ensemble like emcee when deriving a gradient is impractical** — it is
genuinely competitive per evaluation and needs nothing but the log-density.

### 7. Diagonal mass-matrix adaptation (`experiments/mass_matrix.py`)

Every HMC run above used the identity metric — one step size for every
direction. On an axis-aligned target of unequal scales that single step size is
squeezed by the *tightest* direction (leapfrog's stability limit is set by the
largest curvature), so the *widest* direction is under-stepped and mixes slowly.
The fix is a mass matrix $M$ with $K(p) = \tfrac12 p^\top M^{-1} p$, adapted so
$M^{-1} = \operatorname{diag}(\text{marginal variances})$ — each coordinate is
preconditioned to unit scale and one step size fits all of them (drift becomes
`x += eps * inv_mass * p`; a diagonal rescale is still a shear, so exactness is
untouched — derivation and the whitening argument in [theory](theory/derivations.md) §4.8).
The diagonal is learned during warmup from Stan-style memoryless expanding
windows, with the step-size dual averaging restarted after each metric change.

**Anisotropy sweep**, diagonal Gaussian $N(0, \operatorname{diag}(1, r^2))$,
wide-coordinate ESS per 1000 gradients (mean of 5 seeds; the two metrics share
seeds, so the metric is the only difference):

| scale ratio $r$ | identity metric | adapted diagonal | recovered $M^{-1}_{22}$ (true $r^2$) |
|---|---|---|---|
| 2  | 26.9 | 30.2 | 3.9 (4) |
| 5  | **3.4** | 28.5 | 24.2 (25) |
| 10 | 34.0 | 28.0 | 96.6 (100) |
| 25 | 16.1 | 30.5 | 603 (625) |
| 50 | **3.3** | 28.3 | 2395 (2500) |

The adapted metric recovers the true variance and **whitens every $r$ to the
same isotropic problem** — a flat $\approx 30$ ESS/1k-grad regardless of scale.
The identity metric is at the mercy of the scale: its single step size resonates
unpredictably with the wide direction (fixed-$L$ HMC), swinging from 3 to 34 with
no reliability. The win is not a fixed multiplier — it is *scale-independence*.

<p align="center"><img src="figures/mass_matrix_gain.png" width="440"></p>

**Eight schools** (non-centered), ESS per 1000 gradients by coordinate:

| coordinate | identity | adapted | gain |
|---|---|---|---|
| $\mu$ (wide) | 36.5 | 47.8 | 1.3× |
| $\log\tau$ (funnel) | 9.1 | 21.5 | 2.4× |
| $\eta_1$ | 3.2 | 47.8 | **14.8×** |

The metric widens $\mu$ and the $\eta_j$ — the coordinates the unit step size
under-served — driving them to near-independence ($\tau_{\text{int}}\to 1$, hence
the shared ceiling of 47.8). But $\log\tau$ gains only 2.4×: **a diagonal metric
rescales marginals, it cannot rotate**, so the funnel curvature in $(\log\tau,
\eta)$ survives. That residual is exactly what a dense metric or NUTS is for
(Days 17–18) — the honest limit of the cheap fix.

### 8. Rank-normalized split-$\hat R$ (`experiments/rank_rhat.py`)

Classic split-$\hat R$ (§6.2) is a ratio of a between-chain to a within-chain
*variance* — meaningful only when the target has one. On a heavy-tailed
posterior, $W$ is dominated by a few enormous draws and is so noisy that a real
between-chain disagreement disappears into it: the statistic reads a falsely
reassuring $\approx 1.00$ on chains that have plainly not mixed. Since heavy
tails are exactly where mixing is hardest, this is the case you most want a
diagnostic to catch. Vehtari et al. (2021) work with *ranks* instead — finite no
matter how heavy the tails: pool the draws, replace each by its (average) rank,
map ranks to normal scores via the Blom transform
$z = \Phi^{-1}\!\big((r-\tfrac38)/(mn-\tfrac14)\big)$, and run ordinary
split-$\hat R$ on those (`bulk`). A disagreement in *scale* (same centre,
different spread) slips past a location statistic, so the reported value also
folds to $|x-\text{median}|$ and repeats (`folded`); the rank-normalized
$\hat R$ is the max.

Three controlled cases, each with a known verdict (deterministic; `SEED = 20260719`):

| case | classic $\hat R$ | rank bulk | rank folded | rank $\hat R$ | binds |
|---|---|---|---|---|---|
| A — mixed $N(0,1)$ (converged) | 1.00 | 1.00 | 1.00 | **1.00** | — |
| B — Cauchy, location shift of 6 | 1.00 | **1.27** | 1.00 | **1.27** | bulk |
| C — Cauchy, scale $1$ vs $6$, same median | 1.00 | 1.00 | **1.18** | **1.18** | folded |

<p align="center"><img src="figures/rank_rhat.png" width="720"></p>

Reading the three: (A) where the classic statistic is valid the rank version
agrees — it is not allowed to invent a problem. (B) two of four Cauchy chains
shifted by three inter-quartile ranges are genuinely unmixed, but the infinite
variance fools the classic statistic; the rank *bulk* term catches the shift.
(C) equal medians, unequal spread — now the *location* terms (classic **and**
rank-bulk) are both blind, and only the folded term sees it. C is the case that
justifies folding rather than stopping at the bulk rank statistic. `summarize()`
reports `rhat_rank` next to the classic `rhat` so the gap is visible per
coordinate.

### 9. Why NUTS: adaptive trajectory length per gradient (`experiments/nuts_benchmark.py`)

Dual averaging tunes the step size and the diagonal metric (§7) fixes one scale
mismatch, but fixed-length HMC still carries a hand-set knob: the number of
leapfrog steps $L$. Too few and the proposal barely moves; too many and the
trajectory U-turns back toward the start, so the extra gradients buy nothing.
NUTS ([`mcmc/nuts.py`](mcmc/nuts.py), multinomial form of Betancourt 2017) grows
each trajectory until it starts to fold back on itself — no $L$. The honest
question is whether removing the knob *costs* efficiency; the yardstick is **ESS
per gradient** (per 1000 model evaluations — the hardware-independent currency,
with the same asterisk as §6: a gradient eval does more work than RWMH's density
eval, so the per-eval column flatters the gradient-free row).

| non-centered funnel (10-dim) | grad | ESS($v$) | min ESS | **ESS($v$) / 1k ev** | mean depth |
|---|---|---|---|---|---|
| RWMH | density | 1,102 | 1,102 | 4.2 | — |
| HMC (fixed $L=20$) | gradient | 6,616 | 6,616 | 6.7 | — |
| **NUTS** | gradient | 4,545 | 4,545 | **29.2** | 1.9 |

| eight schools (10-dim) | grad | ESS($\tau$) | min ESS | **ESS($\tau$) / 1k ev** | div |
|---|---|---|---|---|---|
| RWMH | density | 5,366 | 181 | 20.6\* | 0 |
| HMC (fixed $L=20$) | gradient | 6,148 | 6,148 | 5.8 | 547 |
| **NUTS** | gradient | 10,779 | 1,963 | **33.6** | 172 |

<p align="center"><img src="figures/nuts_benchmark.png" width="720"></p>

On the benign non-centered funnel the win is purely length: a fixed $L=20$
overshoots the U-turn on the easy directions, while NUTS picks a mean depth of
~2 (≈ a handful of steps) and gets **~4× the ESS per gradient**. On eight
schools NUTS has the most ESS($\tau$) per gradient of the three; RWMH's 20.6\*
carries the asterisk (its eval is a cheap density, and its worst-coordinate ESS
is 181 against NUTS's 1,963).

**NUTS removes the length knob, not the geometry.** Run it on the *centered*
funnel and its divergences pile into the neck — 13% of iterations, and $v$ is
under-covered ($\mathrm{sd}\,2.7$ vs the true $3.0$). Non-centering (§2) drops
that to **zero** divergences and $\mathrm{sd}\,3.0$. This is the honest limit and
the note on simplifications vs Stan: our NUTS uses only a *diagonal* metric and
the endpoint-momentum U-turn check, so the neck's curvature defeats it exactly as
it defeats fixed-L HMC — but the centered funnel defeats Stan's NUTS too, because
the neck is a property of the parameterization, not the sampler. The fix is
choosing good coordinates, not a fancier integrator.

<p align="center"><img src="figures/nuts_funnel_divergences.png" width="720"></p>

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest                          # 98 tests; RuntimeWarnings are errors
cd experiments
python validate_exact.py        # ~30 s
python funnel.py                # ~2 min
python eight_schools.py         # ~1 min
python tempering.py             # ~20 s  (bimodal: tempering vs a trapped chain)
python bnn.py                   # ~1 min  (Bayesian NN: HMC vs ensemble vs MAP)
python external_benchmark.py    # ~10 s  (ours vs emcee; needs `pip install emcee`)
python mass_matrix.py           # ~30 s  (diagonal metric: scale-free efficiency)
python rank_rhat.py             # ~5 s   (rank-normalized R-hat: heavy-tail robustness)
python nuts_benchmark.py        # ~30 s  (NUTS vs fixed-L HMC vs RWMH: ESS per gradient)
```

`emcee` is used *only* by the external benchmark — it is not a dependency of the
package or the tests (CI installs numpy + pytest only). Install it with
`pip install emcee` or `pip install -e '.[bench]'`.

Figures land in `figures/`; every table above is printed by the scripts.
Seeds are fixed (`SEED = 20260703`).

## Design notes

- **Tests assert theory, not just plumbing.** Leapfrog reversibility at
  $10^{-10}$; $O(\varepsilon^2)$ energy scaling; Gibbs's lag-1
  autocorrelation equal to $\rho^2$ on a bivariate Gaussian; the ESS
  estimator recovering $\tau = (1+\rho)/(1-\rho)$ on AR(1) data; every
  hand-derived gradient against central differences.
- **Divergences are a feature.** Trajectories that leave the typical set
  overflow to `inf`/`NaN`, which propagates to a $-\infty$ acceptance ratio
  and a rejection — the mechanism *is* the diagnostic. `np.errstate` is
  scoped to exactly those computations, and the test suite turns any other
  `RuntimeWarning` into a failure.
- **Adaptation stops at warmup's end.** Tuning $\varepsilon$ from chain
  history during sampling would break invariance; dual averaging freezes at
  the averaged iterate.
- **Samplers never see closed forms.** Models expose only
  $\log\tilde\pi$ / $\nabla\log\tilde\pi$; exact posteriors live in separate
  methods used purely for validation.

## Limitations / next

- Trajectory length is now adaptive (NUTS, done, §9): the U-turn criterion
  removes the fixed-$L$ knob and buys ~4–6× the ESS per gradient. The remaining
  metric is still *diagonal* (§7) — it rescales marginals but cannot rotate, so a
  correlated funnel's curvature survives. Both our NUTS and Stan's diverge in the
  *centered* funnel neck; the fix there is non-centering, not the sampler. A
  *dense* or Riemannian metric is the principled next step for curvature the
  reparameterization cannot remove.
- **Phase 2 (done):** Bayesian neural network posterior via this repo's HMC on
  a small MLP — predictive uncertainty and calibration vs a MAP point estimate
  and a deep ensemble ([`experiments/bnn.py`](experiments/bnn.py), section 5).
  Next on this thread: a diagonal mass matrix so HMC does not need such a small
  step size on the ~49-dim weight posterior.

## References

Key sources: Neal (2011) *MCMC using Hamiltonian dynamics*; Hoffman & Gelman
(2014) JMLR (dual averaging); Geyer (1992) *Statist. Sci.* (initial sequence
estimators); Gelman & Rubin (1992); Roberts, Gelman & Gilks (1997) (0.234);
Neal (2003) (funnel); Rubin (1981) (data); Betancourt (2017) arXiv:1701.02434;
Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021) *Bayesian Anal.* 16
(rank-normalized $\hat R$, folding, tail-ESS); Blom (1958) (rankit transform);
Foreman-Mackey et al. (2013) PASP (emcee, the external-benchmark baseline).
Full list with roles in [`theory/derivations.md`](theory/derivations.md).

## Part of a from-scratch series

Same bar in each: the core written out by hand, every non-obvious claim checked
against a closed form or an independent oracle, limitations stated rather than
buried.

| Repo | Built from scratch |
| --- | --- |
| **mcmc-from-scratch** *(this repo)* | Metropolis-Hastings, Gibbs, HMC, NUTS, MALA, parallel tempering — validated against exact posteriors |
| [gp-from-scratch](https://github.com/porth-bot/gp-from-scratch) | GP regression, kernels with hand-derived gradients, ML-II, and the NTK/NNGP wide-network correspondence |
| [grokking-transformer](https://github.com/porth-bot/grokking-transformer) | A transformer that groks modular arithmetic, and the Fourier circuit it learns |
| [pinn-from-scratch](https://github.com/porth-bot/pinn-from-scratch) | Physics-informed networks: exact autograd PDE residuals against closed-form solutions |

The tie to gp-from-scratch is concrete, not decorative. Its NTK section shows
that an *infinite*-width network's posterior is a Gaussian process with a
closed-form mean and variance; section 5 here samples a *finite*-width
network's weight posterior with HMC precisely because that closed form is gone
— the same Bayesian question, on the two sides of the width limit. And the
MALA sampler above is the bridge in the other direction: unadjusted annealed
Langevin is that proposal minus the accept step, which is where score-based
generative models start.

## Provenance

Built as a study resource: implemented from scratch with AI assistance
(Claude), with every derivation written out in
[`theory/derivations.md`](theory/derivations.md) and every non-obvious claim
tested. MIT license.

*Suggested GitHub topics:* `mcmc` `hamiltonian-monte-carlo` `gibbs-sampling`
`metropolis-hastings` `bayesian-inference` `numpy` `from-scratch`
