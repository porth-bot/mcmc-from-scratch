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

import numpy as np


def autocorrelation(x, max_lag=None):
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


def _geyer_tau(rho):
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


def integrated_autocorr_time(x):
    """Integrated autocorrelation time tau via Geyer's initial monotone
    positive sequence.

    tau = 1 + 2 sum_{k=1}^{K} rho_k, with K chosen where the pair sums
    Gamma_m = rho_{2m} + rho_{2m+1} first fail to be positive, after
    enforcing monotone non-increase. Returns tau >= 1.
    """
    tau, _ = _geyer_tau(autocorrelation(x))
    return tau


def autocorr_summary(x, max_lag=None):
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


def plot_autocorrelation(x, ax=None, max_lag=None, label=None, color=None,
                         show_cutoff=True):
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


def ess(x):
    """Effective sample size of one scalar parameter across chains.

    x : (n,) or (m, n). ESS = (m * n) / tau, with tau estimated from
    chain-averaged autocorrelations (each chain centered by its own mean,
    so a between-chain mean shift shows up in R-hat, not hidden here).
    """
    x = np.atleast_2d(x)
    m, n = x.shape
    return m * n / integrated_autocorr_time(x)


def efficiency_summary(chains, seconds, n_evals):
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


def split_rhat(x):
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


def summarize(samples, names=None):
    """Per-dimension diagnostic table for samples of shape (m, n, dim).

    Returns a list of dicts: mean, sd, ESS, tau, split-R-hat per dimension.
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
                "rhat": split_rhat(chains),
            }
        )
    return rows
