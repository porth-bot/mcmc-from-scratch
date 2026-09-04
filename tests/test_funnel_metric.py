"""The funnel-metric experiment's helpers, at settings small enough to test.

The claims in Sec. 15 are structural, not numerical -- they are true of the
funnel for algebraic reasons -- so they can be asserted at dim = 4 in
milliseconds rather than trusted from a several-minute run. What is checked
here is that the *reporting code* reproduces the closed forms, since a study
whose whole content is "these two numbers are exactly equal" fails silently if
the table is built from the wrong matrix.

This is the first test module in the repo to import an experiment rather than
restate it, so it comes with a constraint attached: CI installs numpy + pytest
only, and ``experiments/common.py`` imports matplotlib. ``funnel_metric``
therefore defers that import into the two functions that plot and print, and
these tests run on the numpy-only job rather than skipping there.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from funnel_metric import (  # noqa: E402
    available_rotation,
    by_v,
    estimate_quality,
    local_conditioning,
    step_size_limit,
)
from mcmc.targets import NealsFunnel  # noqa: E402


def test_available_rotation_gives_the_diagonal_and_dense_metrics_the_same_kappa():
    """Study A's headline: the headroom is 1.00, and the identity's is not."""
    f = NealsFunnel(dim=5, sigma_v=2.0)
    rows = {r["metric"]: r["kappa"] for r in available_rotation(f.moments()[1])}
    assert rows["best diagonal"] == pytest.approx(1.0, rel=1e-10)
    assert rows["exact dense"] == pytest.approx(1.0, rel=1e-10)
    # identity leaves kappa(Sigma) = e^{sigma_v^2/2} / sigma_v^2
    assert rows["identity"] == pytest.approx(np.exp(2.0) / 4.0, rel=1e-10)


def test_estimate_quality_reads_the_right_entries():
    """Fed a known perturbation of the truth, the table must report it."""
    f = NealsFunnel(dim=4, sigma_v=2.0)
    cov = f.moments()[1]
    est = cov.copy()
    est[0, 0] *= 0.25                       # sd[v] halved
    est[1, 2] = est[2, 1] = 0.5 * np.sqrt(est[1, 1] * est[2, 2])
    rows = {r["quantity"].split()[0]: r for r in estimate_quality(est, cov)}
    assert rows["sd[v]"]["ratio"] == pytest.approx(0.5)
    assert rows["sd[x_i]"]["ratio"] == pytest.approx(1.0)
    assert rows["max"]["estimate"] == pytest.approx(0.5)


def test_local_conditioning_finds_no_positive_definite_draws():
    """Study C's metric-free number, and the diagonal/dense identity."""
    f = NealsFunnel(dim=6, sigma_v=3.0)
    rows, kappas, z, pd_frac = local_conditioning(f, f.moments()[1], n=4_000, seed=3)
    assert pd_frac == 0.0
    assert np.array_equal(kappas["diagonal"], kappas["dense"])
    assert not np.array_equal(kappas["identity"], kappas["dense"])
    assert all(np.isfinite(r["median kappa"]) for r in rows)


def test_by_v_shows_position_moving_kappa_more_than_the_rotation_does():
    """The comparison the section turns on, at a size a test can afford."""
    f = NealsFunnel(dim=6, sigma_v=3.0)
    _, kappas, z, _ = local_conditioning(f, f.moments()[1], n=20_000, seed=4)
    rows, _ = by_v(kappas, z)
    across = max(r["diagonal"] for r in rows) / min(r["diagonal"] for r in rows)
    assert across > 5.0
    assert all(r["dense gain"] == 1.0 for r in rows)     # exactly, not nearly


def test_step_size_limit_recovers_the_neck_law_and_the_metric_penalty():
    """eps_max ~ e^{v/2} in the neck, and the exact metric loses to the identity.

    The second half is the finding: whitening by Var(x_i) = e^{sigma_v^2/2},
    a number set by the mouth, inflates the neck's curvature by that factor,
    so the 'correct' global metric admits a *smaller* neck step than doing
    nothing at all.
    """
    f = NealsFunnel(dim=6, sigma_v=3.0)
    vs = np.array([-9.0, -7.0, -5.0])
    curves = step_size_limit(f, f.moments()[1], vs, seed=5)
    for name, e in curves.items():
        law = (e / e[0]) / np.exp((vs - vs[0]) / 2.0)
        assert np.allclose(law, 1.0, rtol=0.05), name
    assert curves["diagonal"][0] < 0.5 * curves["identity"][0]
    assert np.array_equal(curves["diagonal"], curves["dense"])
