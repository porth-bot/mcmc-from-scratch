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


def integrated_autocorr_time(x):
    """Integrated autocorrelation time tau via Geyer's initial monotone
    positive sequence.

    tau = 1 + 2 sum_{k=1}^{K} rho_k, with K chosen where the pair sums
    Gamma_m = rho_{2m} + rho_{2m+1} first fail to be positive, after
    enforcing monotone non-increase. Returns tau >= 1.
    """
    rho = autocorrelation(x)
    n_pairs = len(rho) // 2
    gamma = rho[0 : 2 * n_pairs : 2] + rho[1 : 2 * n_pairs : 2]
    # initial positive sequence: truncate at the first non-positive pair
    positive = np.nonzero(gamma <= 0)[0]
    cutoff = positive[0] if len(positive) else len(gamma)
    g = gamma[:cutoff]
    # monotone envelope: pair sums of a reversible chain are non-increasing
    g = np.minimum.accumulate(g) if len(g) else g
    # sum of pair sums counts rho_0 = 1 once: tau = 2 * sum(Gamma) - 1
    return max(1.0, 2.0 * float(np.sum(g)) - 1.0)


def ess(x):
    """Effective sample size of one scalar parameter across chains.

    x : (n,) or (m, n). ESS = (m * n) / tau, with tau estimated from
    chain-averaged autocorrelations (each chain centered by its own mean,
    so a between-chain mean shift shows up in R-hat, not hidden here).
    """
    x = np.atleast_2d(x)
    m, n = x.shape
    return m * n / integrated_autocorr_time(x)


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
