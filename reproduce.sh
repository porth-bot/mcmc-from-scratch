#!/usr/bin/env bash
#
# Regenerate every figure in figures/ and every number in the README, end to
# end, from a clean checkout.
#
#     ./reproduce.sh              # full suite
#     PYTHON=/path/to/python ./reproduce.sh
#
# There is nothing to download and no cached state: every sampler here is
# NumPy and seeded, so the "committed log" for this repo is the seed plus the
# code, and a rerun recomputes the figures rather than replaying stored ones.
# (The two torch repos in the series ship trained checkpoints instead, because
# retraining them takes hours. Here the whole suite is minutes.)
#
# Determinism: every experiment draws from np.random.default_rng(<fixed seed>)
# and NumPy guarantees its bit generators are stable across versions, so the
# sample paths -- hence the ESS/R-hat tables and figures -- reproduce exactly on
# the pinned environment in requirements.txt. Wall-clock columns (ESS/second in
# experiment 6, the chain-scaling appendix) are machine-dependent by nature and
# will differ; the ESS-per-gradient columns beside them are the portable ones.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "${PY}" ]; then
  if [ -x .venv/bin/python ]; then PY="$PWD/.venv/bin/python"; else PY="python3"; fi
fi

echo "=================================================================="
echo "mcmc-from-scratch: full reproduction"
echo "python:     $("$PY" -V 2>&1)  ($PY)"
"$PY" - <<'EOF'
import importlib
for name in ("numpy", "matplotlib", "emcee"):
    try:
        m = importlib.import_module(name)
        print(f"{name+':':11s} {m.__version__}")
    except ImportError:
        print(f"{name+':':11s} MISSING")
EOF
echo "=================================================================="

started=$SECONDS

step() {  # step <label> <script> [args...]
    local label="$1"; shift
    echo
    echo "------------------------------------------------------------------"
    echo ">>> $label"
    echo "------------------------------------------------------------------"
    local t0=$SECONDS
    "$PY" "$@"
    echo "    [${label}: $((SECONDS - t0))s]"
}

# The test suite first: if the samplers no longer match their exact targets,
# the figures below are not worth regenerating.
step "test suite (unit tests + doctests)" -m pytest -q

step "1. exact-posterior validation (Gaussian, linreg, Rosenbrock)" experiments/validate_exact.py
step "2. optimal scaling: the 0.234 acceptance rule"               experiments/optimal_scaling.py
step "3. eight schools (centered vs non-centered)"                 experiments/eight_schools.py
step "4. Neal's funnel"                                            experiments/funnel.py
step "5. parallel tempering on a bimodal target"                   experiments/tempering.py
step "6. Bayesian neural network via HMC"                          experiments/bnn.py
step "7. thinning: what it costs"                                  experiments/thinning.py
step "8. Gibbs scan order (systematic vs random)"                  experiments/gibbs_scan.py
step "9. diagonal mass-matrix adaptation"                          experiments/mass_matrix.py
step "10. rank-normalized split-R-hat"                             experiments/rank_rhat.py
step "11. NUTS vs fixed-L HMC vs RWMH"                             experiments/nuts_benchmark.py
step "12. vectorized-chains scaling appendix"                      experiments/vectorized_scaling.py
step "13. annealed importance sampling: log Z, scored exactly"     experiments/ais.py
step "14. SGHMC: friction, closed form, and cost vs SGLD"          experiments/sghmc.py
step "15. heavy tails: the estimand, the interval, and the sampler" experiments/heavy_tails.py
step "16. dense metric from warmup: shrinkage, and its loss"        experiments/dense_metric_estimation.py
step "17. eight schools: dense metric vs diagonal"                 experiments/eight_schools_metric.py
step "18. the funnel: where the dense metric fails too"           experiments/funnel_metric.py

# emcee is an optional [bench] extra: the external comparison is the only thing
# in the repo that needs it, and CI deliberately runs without it.
if "$PY" -c "import emcee" >/dev/null 2>&1; then
    step "19. external benchmark vs emcee" experiments/external_benchmark.py
else
    echo
    echo ">>> SKIPPED: 19. external benchmark vs emcee (emcee not installed)"
    echo "    pip install -r requirements.txt   # or: pip install -e '.[bench]'"
fi

echo
echo "=================================================================="
echo "done in $((SECONDS - started))s. figures/:"
ls -1 figures/
echo "=================================================================="

# Check the reproduction claim rather than restating it. Everything above was
# recomputed from seeded NumPy, so a figure that now differs from its committed
# copy is either the one file the README says will differ -- vectorized_scaling,
# which plots wall-clock per step and therefore measures the machine -- or real
# drift between the code and what the repo ships. gp-from-scratch has carried
# this check since a fresh clone found two of its figures sitting stale; the
# same claim is made here, so the same check belongs here.
TIMING_FIGURES="figures/vectorized_scaling.png"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    # Only worktree-vs-index differences count as drift. The porcelain format is
    # XY<space>path, X the index status and Y the worktree's, so a figure that is
    # merely newly staged reads "A  path" and is not drift. Untracked ("??") ones
    # do count: a PNG the experiments write but nobody committed is exactly the
    # stale-figure problem seen from the other side.
    changed=$(git status --porcelain -- figures/ \
        | awk '{ y = substr($0, 2, 1); if (y == "M" || substr($0,1,2) == "??") print $2 }')
    unexpected=""
    for f in ${changed}; do
        case " ${TIMING_FIGURES} " in
            *" ${f} "*) ;;
            *) unexpected="${unexpected} ${f}" ;;
        esac
    done
    echo
    if [ -z "${changed}" ]; then
        echo "reproduction: all 26 committed figures came back byte-for-byte."
        echo "(Even the wall-clock plot landed on identical bytes here.)"
    elif [ -z "${unexpected}" ]; then
        echo "reproduction: byte-for-byte except the wall-clock plot, as documented:"
        for f in ${changed}; do echo "    ${f}   (plots seconds; machine-dependent)"; done
    else
        echo "reproduction: UNEXPECTED drift -- these differ from the committed"
        echo "copies and do not plot wall-clock, so the repo is shipping figures"
        echo "its own seeded code no longer produces. Inspect, then commit the"
        echo "regenerated files:"
        for f in ${unexpected}; do echo "    ${f}"; done
        echo "(${TIMING_FIGURES} may also differ; that one is expected to.)"
    fi
    echo "=================================================================="
fi
