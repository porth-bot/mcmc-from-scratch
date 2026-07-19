"""Convergence and efficiency diagnostics.

MCMC output is a *correlated* sample, so two questions must be answered
before trusting any estimate:

1. Have the chains converged to (a common) stationary distribution?
   -> split-R-hat (Gelman & Rubin 1992; split form from Gelman et al., BDA3):
   compare between-chain and within-chain variance. R-hat near 1 is
   necessary (not sufficient) for convergence; standard practice flags
   R-hat > 1.01.

2. How much information do N correlated draws carry?
   -> integrated autocorrelation time tau = 1 + 2 sum_{k>=1} rho_k, the
   variance inflation factor of the sample mean:
   Var(x_bar) = (sigma^2 / N) * tau. Effective sample size ESS = N / tau.
   The truncation of the empirical rho_k sum uses Geyer's (1992) initial
   monotone positive sequence: pair sums Gamma_m = rho_{2m} + rho_{2m+1}
   of a reversible chain are strictly positive and decreasing in theory,
   so we sum pairs only while the (monotonized) empirical pairs stay
   positive -- an adaptive, nearly assumption-free cutoff.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def autocorrelation(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Normalized autocorrelation rho_k of one or more scalar chains.

    x : (n,) or (m, n). For multiple chains, each chain is centered by its
    own mean, per-lag autocovariances are averaged across chains, then
    normalized by the averaged lag-0 term.

    Uses the FFT: the sample autocovariance is the inverse transform of the
    periodogram (Wiener-Khinchin), computed with zero-padding to >= 2n so the
    circular convolution does not wrap. O(n log n) instead of O(n^2).
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    m, n = x.shape
    if max_lag is None:
        max_lag = n - 1
    xc = x - x.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(xc, n=nfft, axis=1)
    acov = np.fft.irfft(f * np.conj(f), n=nfft, axis=1)[:, : max_lag + 1] / n
    acov = acov.mean(axis=0)
    return acov / acov[0]


def _geyer_tau(rho: np.ndarray) -> tuple[float, int]:
    """Geyer's initial monotone positive sequence estimate of tau from rho_k.

    Returns (tau, cutoff_lag): tau >= 1 and the last lag included in the sum
    (the pair sums Gamma_m = rho_{2m} + rho_{2m+1} are summed for m < cutoff,
    covering lags 0 .. 2*cutoff-1). Shared by integrated_autocorr_time and
    autocorr_summary so the plotted cutoff is exactly the one tau uses.
    """
    n_pairs = len(rho) // 2
    gamma = rho[0 : 2 * n_pairs : 2] + rho[1 : 2 * n_pairs : 2]
    # initial positive sequence: truncate at the first non-positive pair
    positive = np.nonzero(gamma <= 0)[0]
    cutoff = int(positive[0]) if len(positive) else len(gamma)
    g = gamma[:cutoff]
    # monotone envelope: pair sums of a reversible chain are non-increasing
    g = np.minimum.accumulate(g) if len(g) else g
    # sum of pair sums counts rho_0 = 1 once: tau = 2 * sum(Gamma) - 1
    tau = max(1.0, 2.0 * float(np.sum(g)) - 1.0)
    cutoff_lag = max(0, 2 * cutoff - 1)
    return tau, cutoff_lag


def integrated_autocorr_time(x: np.ndarray) -> float:
    """Integrated autocorrelation time tau via Geyer's initial monotone
    positive sequence.

    tau = 1 + 2 sum_{k=1}^{K} rho_k, with K chosen where the pair sums
    Gamma_m = rho_{2m} + rho_{2m+1} first fail to be positive, after
    enforcing monotone non-increase. Returns tau >= 1.
    """
    tau, _ = _geyer_tau(autocorrelation(x))
    return tau


def autocorr_summary(x: np.ndarray, max_lag: int | None = None) -> dict[str, Any]:
    """Everything an autocorrelation plot needs, as pure-NumPy data.

    Returns a dict with the lags and normalized autocorrelations rho_k (out to
    ``max_lag``), the Geyer initial-monotone-sequence ``cutoff_lag`` beyond
    which lags are discarded as noise, and the resulting ``tau``/``ess``. The
    curve is truncated for display but ``tau``/``ess``/``cutoff_lag`` are
    computed from the full-length autocorrelation, so they match
    integrated_autocorr_time / ess exactly. Kept separate from the drawing so
    it stays testable without a plotting backend.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    m, n = x.shape
    rho_full = autocorrelation(x)
    tau, cutoff_lag = _geyer_tau(rho_full)
    if max_lag is None:
        max_lag = min(len(rho_full) - 1, 4 * int(np.ceil(tau)) + 10)
    max_lag = min(max_lag, len(rho_full) - 1)
    return {
        "lags": np.arange(max_lag + 1),
        "rho": rho_full[: max_lag + 1],
        "cutoff_lag": cutoff_lag,
        "tau": tau,
        "ess": m * n / tau,
    }


def plot_autocorrelation(
    x: np.ndarray,
    ax: Any = None,
    max_lag: int | None = None,
    label: str | None = None,
    color: Any = None,
    show_cutoff: bool = True,
) -> Any:
    """Draw the autocorrelation function rho_k of a scalar chain(s).

    Marks the Geyer truncation lag (where tau's sum stops) and annotates the
    estimated tau/ESS, so the figure shows *why* the reported ESS is what it
    is rather than just the raw curve. matplotlib is imported lazily -- it is
    an experiments-only dependency, not required to import mcmc.diagnostics.
    Returns the Axes.
    """
    import matplotlib.pyplot as plt

    s = autocorr_summary(x, max_lag=max_lag)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)
    line, = ax.plot(s["lags"], s["rho"], lw=1.4, color=color,
                    label=(label if label is None
                           else f"{label} (τ≈{s['tau']:.1f})"))
    if show_cutoff:
        ax.axvline(s["cutoff_lag"], color=line.get_color(), ls=":", lw=1,
                   alpha=0.7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation")
    if label is not None:
        ax.legend()
    return ax


def ess(x: np.ndarray) -> float:
    """Effective sample size of one scalar parameter across chains.

    x : (n,) or (m, n). ESS = (m * n) / tau, with tau estimated from
    chain-averaged autocorrelations (each chain centered by its own mean,
    so a between-chain mean shift shows up in R-hat, not hidden here).
    """
    x = np.atleast_2d(x)
    m, n = x.shape
    return m * n / integrated_autocorr_time(x)


def tail_ess(x: np.ndarray, prob: float = 0.05) -> float:
    """Tail effective sample size (Vehtari et al. 2021, Sec. 4.3).

    The ordinary (bulk) ESS is computed from the raw draws and is dominated by
    how well the *centre* of the distribution mixes. But a sampler can explore
    the bulk fine while barely visiting the tails -- and it is the tails that
    matter for quantiles, credible-interval endpoints, and tail probabilities.
    Bulk-ESS is blind to this; tail-ESS is designed to catch it.

    The construction "localizes" the ESS at the tails via indicator variables.
    For a lower/upper tail probability ``prob`` (default 5%), form the empirical
    ``prob`` and ``1 - prob`` quantiles from the *pooled* draws, and for each
    turn the chains into a 0/1 indicator series -- "is this draw past the tail
    cutoff?". That indicator is a Bernoulli process whose autocorrelation
    measures how the chain moves in and out of the tail specifically. Its ESS is
    the tail-ESS for that quantile, and we report the *minimum* over the two
    tails (the worse-explored side is the binding constraint):

        tail-ESS = min( ESS[ 1{x <= Q_prob} ],  ESS[ 1{x >= Q_{1-prob}} ] ).

    A large gap between tail-ESS and bulk-ESS is the flag that summaries about
    the tails are less trustworthy than the reported bulk-ESS suggests.

    x : (n,) or (m, n) draws of one scalar parameter across chains.
    prob : lower tail probability in (0, 0.5); the upper cutoff is 1 - prob.

    This is the localized-indicator tail-ESS. Vehtari et al. additionally
    rank-normalize before the bulk statistics; the indicator construction here
    is already on a bounded 0/1 scale, so it is left as-is -- the honest
    simplification is that we do not fold or rank-normalize, which matters for
    bulk-ESS on heavy tails but not for these Bernoulli indicators.
    """
    if not 0.0 < prob < 0.5:
        raise ValueError("prob must be in (0, 0.5)")
    x = np.atleast_2d(np.asarray(x, dtype=float))
    lo, hi = np.quantile(x, [prob, 1.0 - prob])
    # ess() of the tail indicators; ess handles the (m, n) chain shape.
    lower = ess((x <= lo).astype(float))
    upper = ess((x >= hi).astype(float))
    return float(min(lower, upper))


def thinning_variance_ratio(rho: float, k: int) -> float:
    """How much thinning an AR(1) chain by ``k`` inflates Var(mean). Always >= 1.

    Thinning -- keeping every k-th draw and discarding the rest -- is folklore
    ("the samples are correlated, so subsample them until they aren't"). For
    *accuracy* it is always a loss, and for AR(1) the loss has a closed form.

    Take a stationary AR(1) chain with lag-1 correlation ``rho`` in [0, 1). Its
    autocorrelations are ``rho_j = rho^j``, so (Sec. 6.1)

        tau = 1 + 2 sum_{j>=1} rho^j = (1 + rho) / (1 - rho),
        Var(mean of N draws) = (sigma^2 / N) * (1 + rho)/(1 - rho).

    Thin by ``k``: the kept draws are *themselves* an AR(1) chain with lag-1
    correlation ``rho^k`` (a Markov chain observed every k steps still is one),
    and there are only ``N / k`` of them. So

        Var(mean of the thinned chain) = (sigma^2 k / N) * (1 + rho^k)/(1 - rho^k),

    and the ratio of the two -- what this function returns -- is

        R(rho, k) = k * (1 + rho^k) (1 - rho) / [ (1 - rho^k) (1 + rho) ].

    R >= 1 for every k >= 1, with equality only at k = 1. Proof: write
    ``rho = exp(-lambda)``, so ``(1 + rho^k)/(1 - rho^k) = coth(lambda k / 2)``
    and ``R = [k coth(lambda k/2)] / coth(lambda/2)``. The function
    ``v -> v coth(v)`` is increasing on v > 0, because its derivative
    ``(sinh v cosh v - v) / sinh^2 v`` is positive whenever ``sinh(2v) > 2v``,
    which holds for all v > 0. Hence ``u -> u coth(lambda u / 2)`` is increasing
    in u, so ``R(k) >= R(1) = 1``. Thinning never helps and generally hurts.

    Two limits worth knowing, both reproduced in the tests:

    - ``rho = 0`` (independent draws): R = k exactly. Thinning throws away
      k - 1 of every k perfectly good samples, and the variance inflates by
      exactly that factor. This is the pure-waste case.
    - ``rho -> 1`` (an extremely sticky chain): R -> 1. Thinning a chain whose
      autocorrelation time is far longer than the thinning interval costs
      almost nothing -- because the discarded draws were nearly duplicates
      anyway. It still does not *help*.

    So the honest summary is: thinning trades accuracy for storage, and the
    trade is only near-free in the regime where the chain is badly mixing. The
    legitimate reasons to thin are about *cost* -- memory, disk, or an expensive
    per-draw post-processing step (a downstream simulation per sample) -- never
    about accuracy. If you can afford to keep the draws, keep them. See
    ``theory/derivations.md`` Sec. 6.3 and ``experiments/thinning.py``, which
    checks this formula against both simulated AR(1) chains and a real RWMH run.

    Parameters
    ----------
    rho : lag-1 autocorrelation in [0, 1).
    k : thinning interval, an integer >= 1.

    Returns
    -------
    The variance-inflation factor R(rho, k) = Var(thinned mean) / Var(full mean).
    Equivalently ESS_thinned / ESS_full = 1 / R.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    if k == 1:
        return 1.0
    q = rho ** k
    return float(k * (1.0 + q) * (1.0 - rho) / ((1.0 - q) * (1.0 + rho)))


def efficiency_summary(
    chains: np.ndarray, seconds: float, n_evals: int
) -> dict[str, float]:
    """Compute-normalized efficiency of one scalar parameter.

    The raw ESS answers "how many independent draws is this worth"; to
    *compare samplers* you must divide it by what each draw cost. Two honest
    currencies:

    - wall-clock: ``ess_per_sec = ESS / seconds`` -- the metric a practitioner
      actually feels, but hardware- and implementation-dependent.
    - target evaluations: ``ess_per_keval = 1000 * ESS / n_evals`` -- a
      hardware-independent proxy, where one "evaluation" is one call that
      touches the whole model (a density eval for RWMH/emcee, a full-conditional
      draw for Gibbs, a gradient eval for HMC). It is only *approximately*
      comparable across samplers: a gradient costs a constant factor more than
      a density, so this column flatters gradient-free methods relative to
      wall-clock -- which is exactly the gradient-free camp's argument.

    Parameters
    ----------
    chains : ndarray (m, n) or (n,)
        Post-warmup draws of a single scalar parameter across chains.
    seconds : float
        Wall-clock time for the whole run (warmup included -- it is a real cost).
    n_evals : int
        Total target evaluations for the whole run (warmup/burn-in included).

    Returns
    -------
    dict with ``tau``, ``ess``, ``ess_per_sec``, ``ess_per_keval``.
    """
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    m, n = chains.shape
    tau = integrated_autocorr_time(chains)
    ess_val = m * n / tau
    return {
        "tau": tau,
        "ess": float(ess_val),
        "ess_per_sec": float(ess_val / seconds) if seconds > 0 else float("nan"),
        "ess_per_keval": float(1000.0 * ess_val / n_evals) if n_evals > 0 else float("nan"),
    }


def _average_ranks(a: np.ndarray) -> np.ndarray:
    """Ranks 1..N of a flat array, ties sharing their average rank.

    Fractional (average) ranks are what Vehtari et al. use so that repeated
    states -- a Metropolis chain sits still on every rejection, producing many
    exact ties -- do not bias the rank transform toward whichever tied value
    happened to be visited first. Pure NumPy (no scipy.stats.rankdata).
    """
    a = np.asarray(a, dtype=float).ravel()
    n = a.size
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    # group id per sorted position: increments only when the value changes,
    # so all members of a tie group share one id.
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    is_new[1:] = sa[1:] != sa[:-1]
    group = np.cumsum(is_new) - 1
    ranks_1n = np.arange(1, n + 1, dtype=float)          # ranks in sorted order
    group_sum = np.zeros(int(group[-1]) + 1)
    np.add.at(group_sum, group, ranks_1n)
    group_avg = group_sum / np.bincount(group)
    avg_sorted = group_avg[group]
    out = np.empty(n, dtype=float)
    out[order] = avg_sorted
    return out


# Peter Acklam's rational approximation of the standard-normal quantile, then a
# single Halley step against the exact CDF (via math.erfc) to reach machine
# precision. Pure math/NumPy so the numpy-only package/CI stays dependency-free.
_ACKLAM_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
             1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_ACKLAM_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
             6.680131188771972e01, -1.328068155288572e01)
_ACKLAM_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
             -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_ACKLAM_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
             3.754408661907416e00)


def _standard_normal_ppf(p: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF Phi^{-1}(p) for p in (0, 1), vectorized.

    Acklam's piecewise-rational approximation (abs error ~1e-4 in the far tails)
    refined by one Halley iteration against the exact CDF, which drops the error
    below ~1e-8 over the whole (0, 1) range. Used only for the rank-normal (Blom)
    transform, where even the unrefined approximation would suffice.
    """
    p = np.asarray(p, dtype=float)
    x = np.empty_like(p)
    lo, hi = 0.02425, 1.0 - 0.02425
    # central region
    m = (p >= lo) & (p <= hi)
    q = p[m] - 0.5
    r = q * q
    num = (((((_ACKLAM_A[0] * r + _ACKLAM_A[1]) * r + _ACKLAM_A[2]) * r
             + _ACKLAM_A[3]) * r + _ACKLAM_A[4]) * r + _ACKLAM_A[5]) * q
    den = ((((_ACKLAM_B[0] * r + _ACKLAM_B[1]) * r + _ACKLAM_B[2]) * r
            + _ACKLAM_B[3]) * r + _ACKLAM_B[4]) * r + 1.0
    x[m] = num / den
    # tails: lower uses p, upper uses 1-p; the rational form is negative, so the
    # lower tail takes it as-is and the upper tail negates it (Acklam's signs).
    lower, upper = p < lo, p > hi
    for mask, pl, sign in ((lower, p[lower], 1.0), (upper, 1.0 - p[upper], -1.0)):
        if not np.any(mask):
            continue
        q = np.sqrt(-2.0 * np.log(pl))
        t = ((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q
              + _ACKLAM_C[3]) * q + _ACKLAM_C[4]) * q + _ACKLAM_C[5]
        b = (((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q
             + _ACKLAM_D[3]) * q + 1.0
        x[mask] = sign * (t / b)
    # Halley refinement: e = Phi(x) - p, step damped by the curvature term
    erfc = np.vectorize(math.erfc)
    e = 0.5 * erfc(-x / math.sqrt(2.0)) - p
    u = e * math.sqrt(2.0 * math.pi) * np.exp(0.5 * x * x)
    x = x - u / (1.0 + 0.5 * x * u)
    return x


def rank_normalize(x: np.ndarray) -> np.ndarray:
    """Rank-normal (Blom) transform of pooled draws, returned in chain shape.

    Pool all $mn$ draws, replace each by its fractional rank $r$, then map to a
    normal score $z = \\Phi^{-1}\\!\\big((r - 3/8)/(mn - 1/4)\\big)$ (Blom 1958).
    The result is invariant to any monotone reparameterization of the target and
    has finite moments even when the target does not -- the whole point, since
    the classic variance-based $\\hat R$ is undefined for an infinite-variance
    posterior. Shape (m, n) is preserved so split-R-hat can be run on z.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    m, n = x.shape
    ranks = _average_ranks(x)                       # over the pooled mn draws
    z = _standard_normal_ppf((ranks - 0.375) / (m * n - 0.25))
    return z.reshape(m, n)


def rank_normalized_rhat(x: np.ndarray) -> dict[str, float]:
    """Rank-normalized split-R-hat (Vehtari et al. 2021), with the folded term.

    Two failure modes need catching. **Location**: the chains disagree about the
    centre. **Scale**: they agree about the centre but not the spread (one chain
    stuck in a narrow mode, another wandering). Plain split-R-hat on
    rank-normalized draws catches the first (``bulk``); running the same statistic
    on the rank-normalized *folded* draws $|x - \\text{median}|$ catches the
    second (``folded``), because folding turns a scale disagreement into a
    location disagreement of the absolute deviations. The reported ``rhat`` is the
    max of the two -- convergence requires passing both.

    Returns ``{"bulk", "folded", "rhat"}``. Unlike :func:`split_rhat` this is
    well-defined for heavy-tailed (even infinite-variance) targets, where the
    classic statistic is dominated by a few enormous draws and drifts toward a
    falsely reassuring 1. x : (m, n) draws of one scalar parameter across chains.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    bulk = split_rhat(rank_normalize(x))
    med = np.median(x)
    folded = split_rhat(rank_normalize(np.abs(x - med)))
    return {"bulk": bulk, "folded": folded, "rhat": max(bulk, folded)}


def split_rhat(x: np.ndarray) -> float:
    """Split-R-hat for one scalar parameter.

    x : (m, n). Each chain is split in half (2m chains of length n//2), so a
    single chain that drifts between its own halves is caught even when all
    chains agree with each other.

        W = mean of within-chain variances        (underestimates var if not mixed)
        B/n' = variance of the chain means        (overestimates var if not mixed)
        var_plus = (n'-1)/n' W + B/n'             (overestimate of Var_pi)
        R-hat = sqrt(var_plus / W)  -> 1 from above as chains mix.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    m, n = x.shape
    if n < 4:
        raise ValueError("chains too short to split")
    half = n // 2
    halves = np.concatenate([x[:, :half], x[:, half : 2 * half]], axis=0)
    n_h = halves.shape[1]
    chain_means = halves.mean(axis=1)
    chain_vars = halves.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = n_h * chain_means.var(ddof=1)
    var_plus = (n_h - 1) / n_h * W + B / n_h
    return float(np.sqrt(var_plus / W))


def summarize(
    samples: np.ndarray, names: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Per-dimension diagnostic table for samples of shape (m, n, dim).

    Returns a list of dicts per dimension: mean, sd, bulk ESS, tail ESS, tau,
    classic split-R-hat (``rhat``) and rank-normalized R-hat (``rhat_rank``,
    the max of its bulk and folded terms). The tail ESS sits next to the bulk
    ESS so a poorly-explored tail is visible in the same row; ``rhat_rank`` sits
    next to ``rhat`` so a heavy-tailed target where the classic statistic reads
    a falsely reassuring 1 (Sec. 6.4) shows the discrepancy in the same row.
    """
    m, n, dim = samples.shape
    names = names or [f"x[{i}]" for i in range(dim)]
    rows = []
    for i in range(dim):
        chains = samples[:, :, i]
        tau = integrated_autocorr_time(chains)
        rows.append(
            {
                "name": names[i],
                "mean": float(chains.mean()),
                "sd": float(chains.std(ddof=1)),
                "tau": tau,
                "ess": m * n / tau,
                "tail_ess": tail_ess(chains),
                "rhat": split_rhat(chains),
                "rhat_rank": rank_normalized_rhat(chains)["rhat"],
            }
        )
    return rows
