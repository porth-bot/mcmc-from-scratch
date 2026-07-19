"""Experiment 8: rank-normalized split-R-hat -- convergence for heavy tails.

Classic split-R-hat (Sec. 6.2) compares a between-chain to a within-chain
variance. That is only meaningful when the target *has* a variance. On a
heavy-tailed posterior -- the kind HMC is often deployed against, and exactly
where mixing is hardest -- the within-chain variance W is dominated by a few
enormous draws, is wildly noisy, and swamps any real between-chain disagreement.
The statistic then reads a falsely reassuring ~1.00 on chains that have plainly
not mixed.

Vehtari et al. (2021) fix this by working with *ranks*, which are finite no
matter how heavy the tails. Pool the draws, replace each by its (average) rank,
map ranks to normal scores by the Blom transform, and run ordinary split-R-hat
on those. That catches disagreements in *location*. A disagreement in *scale*
(same centre, different spread) slips past it, so the reported statistic also
folds the draws to |x - median| and repeats -- folding turns a scale gap into a
location gap of the absolute deviations. The rank-normalized R-hat is the max.

Three controlled cases, each with a KNOWN verdict:

  A. Mixed light-tailed chains (four N(0,1)). Converged. Both statistics must
     agree at ~1.00 -- rank-normalization is not allowed to invent a problem.
  B. Cauchy, shifted location. Two of four standard-Cauchy chains shifted by 6
     (three inter-quartile ranges): genuinely unmixed. Classic R-hat is fooled
     by the infinite variance; the rank *bulk* term catches the shift.
  C. Cauchy, shifted scale. Same median 0, two chains at scale 1 and two at
     scale 6. Now the location terms (classic AND rank-bulk) are both blind --
     only the *folded* term sees the scale gap.

Run:  python experiments/rank_rhat.py
"""

import numpy as np

from common import plt, print_table, savefig
from mcmc.diagnostics import rank_normalize, rank_normalized_rhat, split_rhat

SEED = 20260719
M, N = 4, 4000
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]


def cases(rng):
    """The three (name, raw draws, what-the-rank-stat-sees) cases."""
    gauss = rng.standard_normal((M, N)) + np.array([0, 0, 0, 0])[:, None]
    shift = rng.standard_cauchy((M, N)) + np.array([0.0, 0.0, 6.0, 6.0])[:, None]
    scale = rng.standard_cauchy((M, N)) * np.array([1.0, 1.0, 6.0, 6.0])[:, None]
    return [
        # name, raw draws, transformed draws the *binding* rank term sees, term
        ("A: mixed N(0,1)", gauss, rank_normalize(gauss), "bulk"),
        ("B: Cauchy, shifted location", shift, rank_normalize(shift), "bulk"),
        ("C: Cauchy, shifted scale",
         scale, rank_normalize(np.abs(scale - np.median(scale))), "folded"),
    ]


def figure(data):
    """Top row: raw draws (clipped) -- chains overlap, classic R-hat sees ~1.
    Bottom row: the transformed draws the rank statistic sees -- chains separate."""
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.0), constrained_layout=True)
    for j, (name, raw, seen, term) in enumerate(data):
        r = rank_normalized_rhat(raw)
        clip = np.clip(raw, *np.percentile(raw, [2, 98]))
        bins_raw = np.linspace(clip.min(), clip.max(), 45)
        bins_seen = np.linspace(seen.min(), seen.max(), 45)
        for c in range(M):
            axes[0, j].hist(clip[c], bins=bins_raw, histtype="step",
                            color=COLORS[c], lw=1.3, density=True)
            axes[1, j].hist(seen[c], bins=bins_seen, histtype="step",
                            color=COLORS[c], lw=1.3, density=True)
        axes[0, j].set_title(f"{name}\nclassic $\\hat R$ = {split_rhat(raw):.2f}")
        axes[1, j].set_title(
            f"rank $\\hat R$ = {r['rhat']:.2f}  "
            f"(bulk {r['bulk']:.2f}, fold {r['folded']:.2f})")
        axes[0, j].set_yticks([])
        axes[1, j].set_yticks([])
    axes[0, 0].set_ylabel("raw draws\n(clipped 2–98%)")
    axes[1, 0].set_ylabel("what rank-$\\hat R$ sees\n(rank-normalized)")
    fig.suptitle("Classic vs rank-normalized split-$\\hat R$: heavy tails hide "
                 "non-convergence from the variance-based statistic", fontsize=11)
    savefig(fig, "rank_rhat.png")


def main():
    rng = np.random.default_rng(SEED)
    data = cases(rng)
    rows = []
    for name, raw, _seen, binding in data:
        r = rank_normalized_rhat(raw)
        rows.append({
            "case": name,
            "classic": split_rhat(raw),
            "rank_bulk": r["bulk"],
            "rank_folded": r["folded"],
            "rank_rhat": r["rhat"],
            "binds": binding,
        })
    print_table(rows, ["case", "classic", "rank_bulk", "rank_folded",
                       "rank_rhat", "binds"])
    figure(cases(np.random.default_rng(SEED)))  # same seed -> table matches fig


if __name__ == "__main__":
    main()
