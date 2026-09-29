"""The README's measured numbers, checked against the logs the runs wrote.

grokking, pinn and diffusion each have this instrument, and each found drifted
numbers on its first run; gp got it in September and found three. This repo
had only ``test_readme_counts.py``, which checks numbers *about* the repo
(figure counts, the one excused PNG) and none of the measurements in its
fifteen result sections. It could not do more: the repo committed no logs, so
there was nothing to check a typed number against except a rerun.

``experiments/common.save_results`` is the first half of the fix: an
experiment writes the quantities its section quotes to ``logs/<name>.json``
beside its figures, and ``reproduce.sh`` reports a log that comes back
different the same way it reports a figure. This file is the second half.

**It covers twelve of the fifteen sections so far** (§§1-12; the §5, §7,
§9 and §10 scripts take about a minute each, §11's about ten seconds, §12's
about twenty).
``NOT_YET`` names the rest, and
``test_every_result_section_is_either_instrumented_or_listed`` fails when a
section is added or renamed, so the gap is stated rather than discovered.

**First run: one drifted number, printed twice.** §2's table and the caption
under the headline figure both gave centered HMC's sd[v] as 2.60. The log
holds 2.5948: ``funnel.py`` prints three decimals (2.595) and that was rounded
again by hand, upward. It is the same double rounding gp's instrument found
three times. Separately, §3's "posterior sds are ~4.3" is now the measured
range (4.1 to 4.5 across mu and the thetas), which a test can hold it to.

**Second slice, §§6 and 8: every table cell clean, one sentence not.** §6 said
centered Gibbs wins per wall-clock second "at ~38k ESS/s". Three reruns on an M4
in September 2026 gave 21.1k to 21.3k. The ordering held (Gibbs
beat HMC's ESS/s by about a third each time), so the claim stays and the
number goes: wall-clock is not logged, so no rate in the README can be
checked, and one that cannot be checked should not be quoted.

**Third slice, §§7 and 9: nothing drifted.** Every cell of their four
tables and the prose numbers around them (§7's "3 to 34" swing and its shared
47.8 ceiling; §9's ~4x, 13% divergences, sd 2.7 vs 3.0) match their logs.
Both scripts' logs came back byte-identical on a second run, with §9's
wall-clock kept out. 30 perturbations of those numbers, 30 caught.

**Fourth slice, §5: one drifted cell.** The calibration table gave the deep
ensemble's mean predictive std on the observed region as 0.11. The log holds
0.1046: ``bnn.py`` prints 0.105 and that was rounded again by hand, upward,
the same double rounding as §2's sd[v]. At 0.10 it sits level with the MAP
ribbon's fixed noise std, which is what the ensemble's band looks like on the
data. Every other cell of both tables, and the prose numbers (7.4x scale
spread, weight R-hat 1.55 / 2.58, prediction R-hat 1.02 / 1.08), match.
The log came back byte-identical on a second run. 39 perturbations, 39 caught.

**Fifth slice, §10: every table cell clean, two sentences overstated.** All
eight ladder cells, the twelve bias cells, the separated-modes table and the
effective-particle counts match ``logs/ais.json``. The prose did not. "Plain
importance sampling wins by 2.6x" was the margin over T = 100 from the tuning
grid, while the ladder table it sits under shows T = 10 at 2.0x and the log
has T = 2 at 1.3x; the sentence now gives both margins. "The jackknife removes
a factor of 3-20" was 3.3 to 23.0; now 3-23. The log came back byte-identical
on a second run and the figure did not move. 51 perturbations, 51 caught.

**Sixth slice, §11: one drifted cell, two sentences overstated.** The
matched-cost table gave SGHMC's ESS per gradient at friction 0.5 as 0.0709.
The run's value is 0.070849: ``sghmc.py`` prints 0.07085 and that was rounded
again by hand, upward, the same double rounding as §§2 and 5. It is now
0.0708. "Raising the friction helps, exactly 2x per doubling" measured 1.97x to
1.99x at the step the claim is made for (2x is the small-step limit), and "the
MAP fit is 43x quieter" held at batch 10 only; at batch 50 it is 32x. The
"300x" was 319x at a friction the section never named; both are now stated.
Every closed-form cell, the gamma = 0 growth, and the BNN noise table match.
The log came back byte-identical on a second run and the figure did not move.

**§12, heavy tails: every cell typed from the output matched, and every
number worked out by hand from rounded cells did not.** The width-ratio column
was divided out of the 3-dp widths, so five of its six cells were off (7.25x
for 7.17x at nu = 2.5, 8.00x for 7.98x at nu = 30); "HMC's ESS per draw falls
180x" is 187x unrounded. Two sentences were stronger than their own tables:
"coverage at or above nominal at every dof" beside cells of 0.947 and 0.949
(within the +/-0.011 two-sigma band over 1,600 replicates, which the section
now says), and "the i.i.d. arm reads ESS/draw = 1.000 at every dof" beside a
0.998. "Three orders of magnitude short" of the exact draws' reach holds at the
Cauchy (1,081x) but is 31x at nu = 1.5; both are now given. The log came back
byte-identical on a second run and the figure did not move.

Pure stdlib plus numpy, so this runs wherever the rest of the suite does. No
matplotlib, no experiment imports.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
README = (ROOT / "README.md").read_text()

# Which result section each committed log backs: log stem -> "### N." prefix.
INSTRUMENTED = {
    "validate_exact": "1.",
    "funnel": "2.",
    "eight_schools": "3.",
    "tempering": "4.",
    "bnn": "5.",
    "external_benchmark": "6.",
    "rank_rhat": "8.",
    "mass_matrix": "7.",
    "nuts_benchmark": "9.",
    "ais": "10.",
    "sghmc": "11.",
    "heavy_tails": "12.",
}

# Sections whose experiments do not write a log yet. Listed, not silent.
NOT_YET = [
    "13.", "14.", "15.",
]


# -- reading the README and the logs -----------------------------------------

def section(number: str) -> str:
    """The text of the `### <number> ...` result section."""
    parts = re.split(r"^### ", README, flags=re.M)
    hits = [p for p in parts if p.startswith(number + " ")]
    assert len(hits) == 1, f"'### {number}' matched {len(hits)} sections"
    return hits[0]


def row(text: str, label: str) -> list[str]:
    """Cells after the label of the one table row whose first cell is `label`.

    Bold markers are dropped and the typographic minus (U+2212) becomes a
    hyphen, so callers compare numbers rather than typography.
    """
    rows = []
    for ln in text.splitlines():
        if not ln.startswith("|"):
            continue
        cs = [c.strip().replace("**", "").replace("−", "-")
              for c in ln.strip().strip("|").split("|")]
        if cs[0] == label:
            rows.append(cs[1:])
    assert len(rows) == 1, f"row {label!r} found {len(rows)} times, expected 1"
    return rows[0]


def number(cell: str) -> str:
    """The leading number of a cell: '2 358' -> '2358', '1.00 (never crossed)'
    -> '1.00'. Digit groups separated by a space are one number."""
    m = re.match(r"-?\d+(?: \d{3})*(?:\.\d+)?", cell.strip())
    assert m, f"no number at the start of {cell!r}"
    return m.group(0).replace(" ", "")


def quoted(text: str, pattern: str) -> str:
    hits = re.findall(pattern, text)
    assert len(hits) == 1, f"{pattern!r} matched {len(hits)} times, expected 1"
    return hits[0]


def log(name: str) -> dict:
    path = LOGS / f"{name}.json"
    assert path.exists(), f"{path} missing: run experiments/{name}.py"
    return json.loads(path.read_text())


def assert_rounds_to(measured: float, printed: str, what: str) -> None:
    """The README's number is what the measurement rounds to, at the precision
    the README chose. A tolerance would accept 0.0165 printed as 0.016; this
    does not."""
    printed = number(printed)
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    assert round(float(measured), decimals) == float(printed), (
        f"{what}: README prints {printed}, the log holds {measured!r}, which "
        f"rounds to {round(float(measured), decimals)} at {decimals} dp"
    )


def assert_thousands(measured: int, printed: str, what: str) -> None:
    """'160k' and '1.5k' style cells."""
    m = re.fullmatch(r"([\d.]+)k", printed.strip())
    assert m, f"{what}: {printed!r} is not an N-k count"
    assert_rounds_to(measured / 1000, m.group(1), what)


# -- bookkeeping --------------------------------------------------------------

def test_every_result_section_is_either_instrumented_or_listed():
    headings = re.findall(r"^### (\d+\.)", README, flags=re.M)
    covered = sorted(INSTRUMENTED.values(), key=float) + NOT_YET
    assert sorted(headings, key=float) == sorted(covered, key=float), (
        "a result section was added, removed or renamed: either instrument it "
        "or add it to NOT_YET"
    )
    assert not set(INSTRUMENTED.values()) & set(NOT_YET)


@pytest.mark.parametrize("name", sorted(INSTRUMENTED))
def test_each_instrumented_section_names_the_script_that_wrote_its_log(name):
    assert f"`experiments/{name}.py`" in section(INSTRUMENTED[name]).splitlines()[0]


# -- Sec. 1: exact-posterior validation (experiments/validate_exact.py) -------

GAUSSIAN_COLS = ["draws", "accept", "max |mean err|", "rel cov err", "tau(x0)",
                 "ESS(x0)", "ESS/1k evals", "R-hat(x0)"]


@pytest.mark.parametrize("sampler", ["RWMH", "Gibbs", "HMC"])
def test_section_1_gaussian_table(sampler):
    rows = {r["sampler"]: r for r in log("validate_exact")["gaussian"]}
    cells = row(section("1."), sampler)
    assert len(cells) == len(GAUSSIAN_COLS)
    r = rows[sampler]
    assert_thousands(r["draws"], cells[0], f"§1 {sampler} draws")
    for col, cell in zip(GAUSSIAN_COLS[1:], cells[1:]):
        assert_rounds_to(r[col], cell, f"§1 {sampler} {col}")


def test_section_1_prose_claims():
    d = log("validate_exact")
    body = section("1.")
    g = {r["sampler"]: r for r in d["gaussian"]}
    # "HMC dominates ($\tau$ 37x smaller than RWMH)"
    ratio = quoted(body, r"\$\\tau\$ (\d+)× smaller than\s+RWMH")
    assert_rounds_to(g["RWMH"]["tau(x0)"] / g["HMC"]["tau(x0)"], ratio,
                     "§1 RWMH/HMC tau ratio")
    # "per density evaluation, exact-conditional Gibbs wins"
    best = max(g.values(), key=lambda r: r["ESS/1k evals"])
    assert best["sampler"] == "Gibbs"
    # "sampled means match the closed form to <= 0.003"
    bound = float(quoted(body, r"closed form to \$\\le ([\d.]+)\$"))
    worst = max(r["max |mean err|"] for r in d["linreg"])
    assert worst <= bound, f"§1 linreg worst mean error {worst:.4f} > {bound}"


# -- Sec. 2: Neal's funnel (experiments/funnel.py) ----------------------------

FUNNEL_ROWS = {"RWMH": "RWMH", "HMC (centered)": "HMC",
               "HMC (non-centered)": "HMC reparam"}
FUNNEL_COLS = ["E[v] (true 0)", "sd[v] (true 3)", "tau(v)", "ESS(v)", "R-hat(v)"]


@pytest.mark.parametrize("label", sorted(FUNNEL_ROWS))
def test_section_2_funnel_table(label):
    rows = {r["sampler"]: r for r in log("funnel")["rows"]}
    r = rows[FUNNEL_ROWS[label]]
    cells = row(section("2."), label)
    assert len(cells) == 7
    assert_thousands(r["draws"], cells[0], f"§2 {label} draws")
    for col, cell in zip(FUNNEL_COLS, cells[1:6]):
        assert_rounds_to(r[col], cell, f"§2 {label} {col}")
    if cells[6] == "—":
        assert r["divergent"] == 0  # RWMH has no trajectory to diverge
    else:
        assert_rounds_to(r["divergent"], cells[6], f"§2 {label} divergent")


def test_section_2_rhat_passes_while_the_bias_is_real():
    """"R-hat = 1.04 while missing the neck entirely": RWMH's R-hat is the one
    quoted, and its sd[v] is far below the exact 3 the reference sampler hits."""
    d = log("funnel")
    rw = {r["sampler"]: r for r in d["rows"]}["RWMH"]
    assert_rounds_to(rw["R-hat(v)"], quoted(section("2."), r"\$\\hat R = ([\d.]+)\$ while"),
                     "§2 RWMH R-hat in prose")
    assert rw["sd[v] (true 3)"] < 2.5 and abs(d["exact_sd_v"] - 3) < 0.02


def test_the_funnel_caption_at_the_top_repeats_section_2():
    """The caption under the headline figure restates §2's table in prose, so
    it drifts with it: the centered-HMC sd read 2.60 in both places."""
    rows = {r["sampler"]: r for r in log("funnel")["rows"]}
    m = re.search(r"never\s+reaches the neck \(sd\$\[v\]\$ = ([\d.]+)\).*?"
                  r"\(sd = ([\d.]+), (\d+)% divergent\).*?"
                  r"\(sd = ([\d.]+), zero divergences\)", README, flags=re.S)
    assert m, "the funnel caption at the top of the README changed shape"
    assert_rounds_to(rows["RWMH"]["sd[v] (true 3)"], m.group(1), "caption RWMH sd")
    h = rows["HMC"]
    assert_rounds_to(h["sd[v] (true 3)"], m.group(2), "caption HMC sd")
    assert_rounds_to(100 * h["divergent"] / h["draws"], m.group(3),
                     "caption HMC divergent %")
    n = rows["HMC reparam"]
    assert_rounds_to(n["sd[v] (true 3)"], m.group(4), "caption non-centered sd")
    assert n["divergent"] == 0


# -- Sec. 3: eight schools (experiments/eight_schools.py) ---------------------

def test_section_3_two_routes_agree():
    d = log("eight_schools")
    body = section("3.")
    assert_rounds_to(d["max_abs_mean_diff"],
                     quoted(body, r"agree to \*\*([\d.]+)\*\*"),
                     "§3 largest posterior-mean difference")
    worst = max(max(r["R-hat"] for r in d[k]["rows"]) for k in ("gibbs", "hmc"))
    bound = float(quoted(body, r"\$\\hat R \\le ([\d.]+)\$ everywhere"))
    assert worst <= bound, f"§3 worst R-hat {worst:.4f} > {bound}"


def test_section_3_posterior_sds_quoted_as_a_range():
    d = log("eight_schools")
    lo, hi = quoted(section("3."), r"posterior sds are ([\d.]+) to ([\d.]+)")
    sds = [r["sd"] for k in ("gibbs", "hmc") for r in d[k]["rows"]
           if r["param"] != "tau"]
    assert_rounds_to(min(sds), lo, "§3 smallest posterior sd")
    assert_rounds_to(max(sds), hi, "§3 largest posterior sd")


def test_section_3_ess_on_mu():
    d = log("eight_schools")
    m = re.search(r"ESS on \$\\mu\$ is ([\d.]+k) from ([\d.]+k) draws vs Gibbs's\s+"
                  r"([\d.]+k) from ([\d.]+k)", section("3."))
    assert m, "§3's ESS-on-mu sentence changed shape"
    mu = {k: next(r for r in d[k]["rows"] if r["param"] == "mu") for k in ("gibbs", "hmc")}
    assert_thousands(mu["hmc"]["ESS"], m.group(1), "§3 HMC ESS(mu)")
    assert_thousands(d["hmc"]["draws"], m.group(2), "§3 HMC draws")
    assert_thousands(mu["gibbs"]["ESS"], m.group(3), "§3 Gibbs ESS(mu)")
    assert_thousands(d["gibbs"]["draws"], m.group(4), "§3 Gibbs draws")


# -- Sec. 4: parallel tempering (experiments/tempering.py) --------------------

def test_section_4_tempering_table():
    d = log("tempering")
    body = section("4.")
    for label, key in [("single random walk", "random_walk"),
                       ("parallel tempering (8 replicas)", "parallel_tempering")]:
        cells = row(body, label)
        assert_rounds_to(d[key]["mean"][0], cells[0], f"§4 {label} E[x0]")
        assert_rounds_to(d[key]["left_frac"], cells[1], f"§4 {label} left-mode frac")
    assert "(never crossed)" in row(body, "single random walk")[1]
    assert d["random_walk"]["left_frac"] == 1.0
    assert "(true 1.8)" in body and "(true 0.35)" in body
    assert_rounds_to(d["true_mean"][0], "1.8", "§4 true E[x0]")


def test_section_4_ladder_and_swap_rates():
    d = log("tempering")
    body = section("4.")
    assert len(d["betas"]) == 8 and "8 replicas" in body
    assert d["betas"][0] == 1.0 and d["betas"][-1] == pytest.approx(0.01)
    # "Swap acceptance holds at ~0.7 across the ladder": every adjacent pair,
    # not just the average.
    claimed = quoted(body, r"Swap acceptance holds at ~([\d.]+) across the ladder")
    for k, rate in enumerate(d["swap_rates"]):
        assert_rounds_to(rate, claimed, f"§4 swap rate between replicas {k} and {k + 1}")
    assert np.all(np.asarray(d["swap_rates"]) > 0)


# -- Sec. 6: external benchmark vs emcee (experiments/external_benchmark.py) --

GAUSS_ROWS = ["RWMH (ours)", "Gibbs (ours)", "HMC (ours)", "emcee (stretch)"]
SCHOOLS_ROWS = {"Gibbs (ours, centered)": "Gibbs (ours)",
                "HMC (ours, non-centered)": "HMC (ours)",
                "emcee (stretch, non-centered)": "emcee (stretch)"}


def _gauss_part(body: str) -> str:
    return body.split("**Eight schools**")[0]


def _schools_part(body: str) -> str:
    return body.split("**Eight schools**")[1]


@pytest.mark.parametrize("label", GAUSS_ROWS)
def test_section_6_correlated_gaussian_table(label):
    r = {x["sampler"]: x for x in log("external_benchmark")["correlated_gaussian"]}[label]
    cells = row(_gauss_part(section("6.")), label)
    assert len(cells) == 5
    assert cells[0] == r["grad"]
    assert_thousands(r["draws"], cells[1], f"§6 {label} draws")
    assert_rounds_to(r["min ESS"], cells[2], f"§6 {label} min ESS")
    assert_rounds_to(r["ESS/1k ev"], cells[3], f"§6 {label} ESS/1k evals")
    assert_rounds_to(r["R-hat"], cells[4], f"§6 {label} R-hat")


@pytest.mark.parametrize("label", sorted(SCHOOLS_ROWS))
def test_section_6_eight_schools_table(label):
    rows = {x["sampler"]: x for x in log("external_benchmark")["eight_schools"]}
    r = rows[SCHOOLS_ROWS[label]]
    cells = row(_schools_part(section("6.")), label)
    assert len(cells) == 5
    assert cells[0] == r["grad"]
    for col, cell in zip(["ESS(mu)", "ESS(tau)", "ESS(tau)/1k ev", "R-hat(tau)"],
                         cells[1:]):
        assert_rounds_to(r[col], cell, f"§6 {label} {col}")


def test_section_6_bold_marks_the_column_winners():
    """The bold cells are claims about who wins a column, so they move when the
    numbers do."""
    d = log("external_benchmark")
    g = {x["sampler"]: x for x in d["correlated_gaussian"]}
    body = _gauss_part(section("6."))
    for col, cell_ix in [("min ESS", 2), ("ESS/1k ev", 3)]:
        best = max(g, key=lambda k: g[k][col])
        bolded = [lab for lab in GAUSS_ROWS
                  if any(ln.startswith(f"| {lab} |") and
                         ln.split("|")[cell_ix + 2].strip().startswith("**")
                         for ln in body.splitlines())]
        assert bolded == [best], f"§6 Gaussian {col}: bold on {bolded}, best is {best}"
    e = {x["sampler"]: x for x in d["eight_schools"]}
    body = _schools_part(section("6."))
    inv = {v: k for k, v in SCHOOLS_ROWS.items()}
    for col, cell_ix in [("ESS(mu)", 1), ("ESS(tau)/1k ev", 3)]:
        best = inv[max(e, key=lambda k: e[k][col])]
        bolded = [lab for lab in SCHOOLS_ROWS
                  if any(ln.startswith(f"| {lab} |") and
                         ln.split("|")[cell_ix + 2].strip().startswith("**")
                         for ln in body.splitlines())]
        assert bolded == [best], f"§6 schools {col}: bold on {bolded}, best is {best}"


def test_section_6_prose_claims():
    d = log("external_benchmark")
    body = section("6.")
    g = {x["sampler"]: x for x in d["correlated_gaussian"]}
    # "(27.9 vs 12.6 — ...)" emcee against RWMH, per evaluation
    em, rw = re.search(r"\(([\d.]+) vs ([\d.]+) —", body).groups()
    assert_rounds_to(g["emcee (stretch)"]["ESS/1k ev"], em, "§6 emcee ESS/1k in prose")
    assert_rounds_to(g["RWMH (ours)"]["ESS/1k ev"], rw, "§6 RWMH ESS/1k in prose")
    # "and even edges HMC's per-eval number"
    assert g["emcee (stretch)"]["ESS/1k ev"] > g["HMC (ours)"]["ESS/1k ev"]
    # "exact-conditional Gibbs wins outright, both per evaluation ..."
    assert max(g, key=lambda k: g[k]["ESS/1k ev"]) == "Gibbs (ours)"
    # No rate in ESS/s: wall-clock is not logged, so none can be checked.
    assert not re.search(r"\d\s*k? ESS/s", body), "§6 quotes an ESS/s figure"

    e = {x["sampler"]: x for x in d["eight_schools"]}
    assert_thousands(e["HMC (ours)"]["ESS(mu)"],
                     quoted(body, r"ESS\(\$\\mu\$\) ~(\d+k)"), "§6 HMC ESS(mu)")
    assert_rounds_to(e["Gibbs (ours)"]["ESS(mu)"] / 100,
                     str(int(quoted(body, r"ESS\(\$\\mu\$\) \(~(\d+)\)")) // 100),
                     "§6 Gibbs ESS(mu) to the hundred")
    h = d["eight_schools_hmc"]
    assert_rounds_to(100 * h["divergent"] / h["draws"],
                     quoted(body, r"~(\d+)% divergences"), "§6 HMC divergence %")
    # "reaches ESS comparable to HMC on the hard tau coordinate": within 10%.
    ratio = e["emcee (stretch)"]["ESS(tau)"] / e["HMC (ours)"]["ESS(tau)"]
    assert 0.9 < ratio < 1.1, f"§6 emcee/HMC ESS(tau) = {ratio:.2f}"


# -- Sec. 8: rank-normalized split-R-hat (experiments/rank_rhat.py) -----------

RANK_ROWS = {"A — mixed $N(0,1)$ (converged)": "A: mixed N(0,1)",
             "B — Cauchy, location shift of 6": "B: Cauchy, shifted location",
             "C — Cauchy, scale $1$ vs $6$, same median": "C: Cauchy, shifted scale"}


@pytest.mark.parametrize("label", sorted(RANK_ROWS))
def test_section_8_table(label):
    d = log("rank_rhat")
    r = {c["case"]: c for c in d["cases"]}[RANK_ROWS[label]]
    cells = row(section("8."), label)
    assert len(cells) == 5
    for col, cell in zip(["classic", "rank_bulk", "rank_folded", "rank_rhat"], cells):
        assert_rounds_to(r[col], cell, f"§8 {label[0]} {col}")
    # "binds" is measured, not the script's label: the larger rank term at the
    # table's two decimals, or a dash when they tie there.
    bulk, folded = round(r["rank_bulk"], 2), round(r["rank_folded"], 2)
    expected = "—" if bulk == folded else ("bulk" if bulk > folded else "folded")
    assert cells[4] == expected, f"§8 {label[0]} binds: {cells[4]!r}, measured {expected!r}"


def test_section_8_verdicts():
    """Each case has a known answer: A converged, B and C not. The classic
    statistic must read ~1.00 on all three (that is the failure shown), and the
    rank statistic must separate them."""
    d = {c["case"][0]: c for c in log("rank_rhat")["cases"]}
    for k in "ABC":
        assert round(d[k]["classic"], 2) == 1.00
    assert round(d["A"]["rank_rhat"], 2) == 1.00
    assert d["B"]["rank_rhat"] > 1.01 and d["C"]["rank_rhat"] > 1.01
    assert round(d["C"]["rank_bulk"], 2) == 1.00  # the location term is blind to C
    assert "SEED = " + str(log("rank_rhat")["seed"]) in section("8.")


# -- Sec. 7: diagonal mass-matrix adaptation (experiments/mass_matrix.py) -----

def _sweep_part(body: str) -> str:
    return body.split("**Eight schools**")[0]


@pytest.mark.parametrize("r", [2, 5, 10, 25, 50])
def test_section_7_anisotropy_sweep_table(r):
    s = {int(x["ratio r"]): x for x in log("mass_matrix")["sweep"]}[r]
    cells = row(_sweep_part(section("7.")), str(r))
    assert len(cells) == 3
    assert_rounds_to(s["ident ESS/keval"], cells[0], f"§7 r={r} identity")
    assert_rounds_to(s["adapt ESS/keval"], cells[1], f"§7 r={r} adapted")
    got, true = re.fullmatch(r"([\d.]+) \((\d+)\)", cells[2]).groups()
    assert_rounds_to(s["inv_mass[1]"], got, f"§7 r={r} recovered M^-1")
    assert int(true) == s["true var[1]"] == r * r


def test_section_7_sweep_bold_marks_the_identity_collapses():
    """The two bold identity cells are the two lowest, the ones the prose's
    "swinging from 3" refers to."""
    sweep = log("mass_matrix")["sweep"]
    body = _sweep_part(section("7."))
    bolded = sorted(int(x["ratio r"]) for x in sweep
                    if f"**{row(body, str(int(x['ratio r'])))[0]}**" in body)
    lowest = sorted(int(x["ratio r"]) for x in
                    sorted(sweep, key=lambda x: x["ident ESS/keval"])[:2])
    assert bolded == lowest


def test_section_7_sweep_prose():
    sweep = log("mass_matrix")["sweep"]
    body = section("7.")
    ident = [x["ident ESS/keval"] for x in sweep]
    adapt = [x["adapt ESS/keval"] for x in sweep]
    lo, hi = re.search(r"swinging from (\d+) to (\d+)", body).groups()
    assert_rounds_to(min(ident), lo, "§7 identity low")
    assert_rounds_to(max(ident), hi, "§7 identity high")
    # "a flat ~30 ESS/1k-grad regardless of scale": every r within 10% of 30,
    # and the identity metric's spread at least five times wider.
    flat = quoted(body, r"flat \$\\approx (\d+)\$ ESS")
    assert all(abs(a / float(flat) - 1) < 0.1 for a in adapt), adapt
    assert (max(ident) - min(ident)) > 5 * (max(adapt) - min(adapt))


ES_COORDS = {"$\\mu$ (wide)": "mu", "$\\log\\tau$ (funnel)": "log tau",
             "$\\eta_1$": "eta_1"}


@pytest.mark.parametrize("label", sorted(ES_COORDS))
def test_section_7_eight_schools_table(label):
    r = {x["param"]: x for x in log("mass_matrix")["eight_schools"]}[ES_COORDS[label]]
    cells = row(section("7.").split("**Eight schools**")[1], label)
    assert len(cells) == 3
    assert_rounds_to(r["ident ESS/keval"], cells[0], f"§7 {label} identity")
    assert_rounds_to(r["adapt ESS/keval"], cells[1], f"§7 {label} adapted")
    assert cells[2].endswith("×")
    assert_rounds_to(r["gain x"], cells[2][:-1], f"§7 {label} gain")


def test_section_7_eight_schools_prose():
    rows = {x["param"]: x for x in log("mass_matrix")["eight_schools"]}
    body = section("7.")
    ceiling = quoted(body, r"shared ceiling of ([\d.]+)")
    for p in ("mu", "eta_1"):
        assert_rounds_to(rows[p]["adapt ESS/keval"], ceiling, f"§7 {p} ceiling")
    assert_rounds_to(rows["log tau"]["gain x"],
                     quoted(body, r"\$\\log\\tau\$ gains only ([\d.]+)×"),
                     "§7 log tau gain in prose")
    assert_rounds_to(rows["log tau"]["gain x"],
                     quoted(body, r"single-seed ([\d.]+) above"),
                     "§7 single-seed log tau gain")


# -- Sec. 9: NUTS vs fixed-L HMC vs RWMH (experiments/nuts_benchmark.py) ------

NUTS_ROWS = {"RWMH": "RWMH", "HMC (fixed $L=20$)": "HMC (fixed L=20)",
             "NUTS": "NUTS"}


def _nuts_tables(body: str) -> tuple[str, str]:
    funnel, schools = body.split("| eight schools (10-dim)")
    return funnel, schools.split("NUTS removes the length knob")[0]


@pytest.mark.parametrize("label", sorted(NUTS_ROWS))
def test_section_9_funnel_table(label):
    r = {x["sampler"]: x for x in log("nuts_benchmark")["funnel_noncentered"]}[NUTS_ROWS[label]]
    cells = row(_nuts_tables(section("9."))[0], label)
    assert len(cells) == 5
    assert cells[0] == ("gradient" if r["grad"] == "yes" else "density")
    assert_rounds_to(r["ESS(v)"], cells[1].replace(",", ""), f"§9 funnel {label} ESS(v)")
    assert_rounds_to(r["min ESS"], cells[2].replace(",", ""), f"§9 funnel {label} min ESS")
    assert_rounds_to(r["ESS(v)/1k ev"], cells[3], f"§9 funnel {label} ESS/1k")
    if cells[4] == "—":
        assert r["depth"] == "nan"  # only NUTS builds a tree
    else:
        assert_rounds_to(r["depth"], cells[4], f"§9 funnel {label} depth")


@pytest.mark.parametrize("label", sorted(NUTS_ROWS))
def test_section_9_eight_schools_table(label):
    r = {x["sampler"]: x for x in log("nuts_benchmark")["eight_schools"]}[NUTS_ROWS[label]]
    cells = row(_nuts_tables(section("9."))[1], label)
    assert len(cells) == 5
    assert cells[0] == ("gradient" if r["grad"] == "yes" else "density")
    assert_rounds_to(r["ESS(tau)"], cells[1].replace(",", ""), f"§9 schools {label} ESS(tau)")
    assert_rounds_to(r["min ESS"], cells[2].replace(",", ""), f"§9 schools {label} min ESS")
    assert_rounds_to(r["ESS(tau)/1k ev"], cells[3].rstrip("\\*"), f"§9 schools {label} ESS/1k")
    assert int(cells[4]) == r["div"]


def test_section_9_bold_winner_is_the_measured_winner():
    d = log("nuts_benchmark")
    for key, col in [("funnel_noncentered", "ESS(v)/1k ev"),
                     ("eight_schools", "ESS(tau)/1k ev")]:
        assert max(d[key], key=lambda x: x[col])["sampler"] == "NUTS"
    body = section("9.")
    assert body.count("| **NUTS** |") == 2


def test_section_9_prose():
    d = log("nuts_benchmark")
    body = section("9.")
    f = {x["sampler"]: x for x in d["funnel_noncentered"]}
    s = {x["sampler"]: x for x in d["eight_schools"]}
    assert_rounds_to(f["NUTS"]["ESS(v)/1k ev"] / f["HMC (fixed L=20)"]["ESS(v)/1k ev"],
                     quoted(body, r"\*\*~(\d+)× the ESS per gradient\*\*"),
                     "§9 NUTS/HMC ESS per gradient")
    assert_rounds_to(f["NUTS"]["depth"], quoted(body, r"mean depth of\s+~(\d+)"),
                     "§9 NUTS mean depth in prose")
    assert_rounds_to(s["RWMH"]["ESS(tau)/1k ev"], quoted(body, r"RWMH's ([\d.]+)\\\*"),
                     "§9 RWMH ESS/1k in prose")
    worst_rw, worst_nuts = re.search(r"ESS\s+is ([\d,]+) against NUTS's ([\d,]+)",
                                     body).groups()
    assert_rounds_to(s["RWMH"]["min ESS"], worst_rw.replace(",", ""), "§9 RWMH min ESS")
    assert_rounds_to(s["NUTS"]["min ESS"], worst_nuts.replace(",", ""), "§9 NUTS min ESS")


def test_section_9_the_centered_funnel_limit():
    lim = log("nuts_benchmark")["centered_funnel_limit"]
    body = section("9.")
    c, n = lim["centered"], lim["non_centered"]
    assert_rounds_to(100 * c["divergent"] / c["draws"],
                     quoted(body, r"— (\d+)% of iterations"), "§9 centered divergence %")
    sd_c, true = re.search(r"\\mathrm\{sd\}\\,([\d.]+)\$ vs the true \$([\d.]+)\$",
                           body).groups()
    assert_rounds_to(c["sd_v"], sd_c, "§9 centered sd[v]")
    assert float(true) == 3.0
    assert n["divergent"] == 0 and "**zero** divergences" in body
    assert_rounds_to(n["sd_v"], quoted(body, r"divergences and \$\\mathrm\{sd\}\\,([\d.]+)\$"),
                     "§9 non-centered sd[v]")


# -- Sec. 5: Bayesian neural network (experiments/bnn.py) ---------------------

CALIB_COLS = ["cover95", "nll", "mean_std"]
CALIB_ROWS = [(m, reg) for m in ("HMC (posterior)", "deep ensemble (5)",
                                 "point estimate (MAP)")
              for reg in ("observed", "gap")]


def _calibration_cells(method: str, region: str) -> list[str]:
    """The calibration table keys a row on two cells, so `row` cannot find it."""
    body = section("5.").split("| method | region |")[1].split("\n\n")[0]
    return row(body.replace(f"| {method} | {region} |", f"| {method}/{region} |")
               .replace(f"| {method} | **{region}** |", f"| {method}/{region} |"),
               f"{method}/{region}")


@pytest.mark.parametrize("method,region", CALIB_ROWS)
def test_section_5_calibration_table(method, region):
    r = {(x["method"], x["region"]): x
         for x in log("bnn")["calibration"]}[(method, region)]
    cells = _calibration_cells(method, region)
    assert len(cells) == len(CALIB_COLS)
    for col, cell in zip(CALIB_COLS, cells):
        assert_rounds_to(r[col], cell, f"§5 {method} {region} {col}")


@pytest.mark.parametrize("label", ["identity", "adapted diagonal"])
def test_section_5_mass_matrix_table(label):
    r = {x["metric"]: x for x in log("bnn")["mass_matrix"]}[label]
    body = section("5.").split("| metric | step size |")[1]
    cells = row(body, label)
    assert len(cells) == 5
    for col, cell in zip(["step", "accept", "ESS_med", "ESS/1k_grad"], cells):
        assert_rounds_to(r[col], cell, f"§5 {label} {col}")
    assert int(cells[4]) == r["diverg"]


def test_section_5_calibration_prose():
    d = log("bnn")
    body = section("5.")
    c = {(x["method"], x["region"]): x for x in d["calibration"]}
    assert d["dim"] == int(quoted(body, r"\$3H\+1 = (\d+)\$ dimensions"))
    assert d["n_test"] == int(quoted(body, r"calibration on (\d+) fresh points"))
    mp = c[("point estimate (MAP)", "gap")]
    assert_rounds_to(mp["cover95"], quoted(body, r"collapses\s+to ([\d.]+)"),
                     "§5 MAP gap coverage in prose")
    assert_rounds_to(mp["nll"], quoted(body, r"NLL blows up to ([\d.]+)"),
                     "§5 MAP gap NLL in prose")
    assert_rounds_to(c[("deep ensemble (5)", "gap")]["cover95"],
                     quoted(body, r"under-covers the gap \(([\d.]+)\)"),
                     "§5 ensemble gap coverage in prose")
    hmc_cov, hmc_nll = re.search(r"stays calibrated \(([\d.]+) / NLL ([\d.]+)\)",
                                 body).groups()
    hg = c[("HMC (posterior)", "gap")]
    assert_rounds_to(hg["cover95"], hmc_cov, "§5 HMC gap coverage in prose")
    assert_rounds_to(hg["nll"], hmc_nll, "§5 HMC gap NLL in prose")
    # "HMC widens the most": the widest gap band of the three
    gap = [x for x in d["calibration"] if x["region"] == "gap"]
    assert max(gap, key=lambda x: x["mean_std"])["method"] == "HMC (posterior)"


def test_section_5_mass_matrix_prose():
    mm = {x["metric"]: x for x in log("bnn")["mass_matrix"]}
    body = section("5.")
    ident, diag = mm["identity"], mm["adapted diagonal"]
    assert_rounds_to(diag["scale_spread"], quoted(body, r"do span ([\d.]+)×"),
                     "§5 adapted scale spread")
    # "It buys nothing" and "The step size does not go *up*"
    assert diag["ESS/1k_grad"] <= ident["ESS/1k_grad"]
    assert diag["step"] <= ident["step"]


def test_section_5_convergence_in_function_space():
    d = log("bnn")
    body = section("5.")
    med, mx = re.search(r"median ([\d.]+) and up to ([\d.]+) across coordinates",
                        body).groups()
    assert_rounds_to(d["weight_rhat"]["median"], med, "§5 weight R-hat median")
    assert_rounds_to(d["weight_rhat"]["max"], mx, "§5 weight R-hat max")
    med, mx = re.search(r"sits at ([\d.]+) \(max ([\d.]+)\)", body).groups()
    assert_rounds_to(d["pred_rhat"]["median"], med, "§5 prediction R-hat median")
    assert_rounds_to(d["pred_rhat"]["max"], mx, "§5 prediction R-hat max")
    # "with ESS in the hundreds"
    assert "ESS in the hundreds" in body
    assert 100 <= d["pred_ess"]["min"] and d["pred_ess"]["median"] < 1000


# -- Sec. 10: annealed importance sampling (experiments/ais.py) ---------------

LADDER_COLS = {"T=1": 1, "T=10": 10, "T=100": 100, "T=500": 500}


@pytest.mark.parametrize("dim", [4, 8])
def test_section_10_ladder_table(dim):
    rows = {r["T"]: r for r in log("ais")[f"ladder_d{dim}"]}
    cells = row(section("10."), f"$d = {dim}$, RMSE of $\\log Z$")
    assert len(cells) == 5
    for (label, t), cell in zip(LADDER_COLS.items(), cells):
        assert_rounds_to(rows[t]["rmse"], cell, f"§10 d={dim} {label} rmse")
    best = min(rows.values(), key=lambda r: r["rmse"])
    assert cells[4] == f"$T = {best['T']}$"


def test_section_10_bold_marks_the_best_ladder():
    body = section("10.")
    for dim in (4, 8):
        cells = row(body, f"$d = {dim}$, RMSE of $\\log Z$")
        raw = [ln for ln in body.splitlines()
               if ln.startswith(f"| $d = {dim}$, RMSE")][0].split("|")[2:6]
        bold = [c for c, r in zip(cells, raw) if "**" in r]
        rows = {r["T"]: r for r in log("ais")[f"ladder_d{dim}"]}
        best = min(rows[t]["rmse"] for t in LADDER_COLS.values())
        assert len(bold) == 1 and float(bold[0]) == round(best, 3)


def test_section_10_ladder_prose():
    d = log("ais")
    body = " ".join(section("10.").split())
    l4 = {r["T"]: r for r in d["ladder_d4"]}
    l8 = {r["T"]: r for r in d["ladder_d8"]}
    assert d["log_z"] == float(quoted(body, r"so \$\\log Z = ([\d.]+)\$ exactly"))
    assert d["budget"] == int(quoted(body, r"budget held at ([\d,]+),").replace(",", ""))
    assert d["n_replicates"] == int(quoted(body, r"over (\d+) replicates:"))
    # "beats every ladder length": T = 1 has the lowest RMSE at d = 4
    assert min(l4.values(), key=lambda r: r["rmse"])["T"] == 1
    t2, ratio2 = re.search(r"\$T = 2\$ scores ([\d.]+), ([\d.]+)× worse",
                           body).groups()
    assert_rounds_to(l4[2]["rmse"], t2, "§10 T=2 rmse")
    assert_rounds_to(l4[2]["rmse"] / l4[1]["rmse"], ratio2, "§10 T=2 margin")
    tuned = [r for r in d["tuning_d4"] if r["T"] > 1]
    assert len(d["tuning_d4"]) == {"eight": 8}[quoted(body, r"(\w+) settings of ladder")]
    best, ratio = re.search(r"best annealed one is ([\d.]+), ([\d.]+)× worse",
                            body).groups()
    best_tuned = min(r["rmse"] for r in tuned)
    assert_rounds_to(best_tuned, best, "§10 best tuned rmse")
    assert_rounds_to(best_tuned / l4[1]["rmse"], ratio, "§10 tuned margin")
    best8 = min(l8.values(), key=lambda r: r["rmse"])
    assert_rounds_to(l8[1]["rmse"] / best8["rmse"],
                     quoted(body, r"annealing is worth ([\d.]+)×"),
                     "§10 d=8 annealing gain")


def test_section_10_effective_particles():
    d = log("ais")
    body = " ".join(section("10.").split())
    l4 = {r["T"]: r for r in d["ladder_d4"]}
    frac1, eff1 = re.search(r"\$T = 1\$ has an ESS fraction of ([\d.]+) .*?"
                            r"particles is \*\*(\d+)\*\* effective", body).groups()
    assert_rounds_to(l4[1]["ESS frac"], frac1, "§10 T=1 ESS fraction")
    assert_rounds_to(l4[1]["eff. particles"], eff1, "§10 T=1 effective particles")
    frac500, eff500 = re.search(r"\$T = 500\$ turns a fraction of ([\d.]+) "
                                r"into \*\*(\d+)\*\*", body).groups()
    assert_rounds_to(l4[500]["ESS frac"], frac500, "§10 T=500 ESS fraction")
    assert_rounds_to(l4[500]["eff. particles"], eff500,
                     "§10 T=500 effective particles")
    assert_rounds_to(d["ladder_d8"][0]["eff. particles"],
                     quoted(body, r"down to \*\*([\d.]+)\*\* effective"),
                     "§10 d=8 T=1 effective particles")
    assert d["ladder_d8"][0]["T"] == 1


@pytest.mark.parametrize("label,key", [("bias (nats)", "bias"),
                                       ("after jackknife", "jack bias")])
def test_section_10_bias_table(label, key):
    rows = log("ais")["bias"]
    body = section("10.")
    ns = [int(n) for n in row(body, "$N$")]
    assert ns == [r["N"] for r in rows]
    cells = row(body, label)
    assert len(cells) == len(rows)
    for r, cell in zip(rows, cells):
        assert_rounds_to(r[key], cell, f"§10 N={r['N']} {key}")


def test_section_10_bias_prose():
    rows = log("ais")["bias"]
    body = " ".join(section("10.").split()).replace("−", "-")
    assert all(r["bias"] < 0 for r in rows)  # "sits *below* the truth"
    assert all(abs(r["jack bias"]) < abs(r["bias"]) for r in rows)
    lo, hi = re.search(r"a factor of (\d+)–(\d+) here", body).groups()
    ratios = [r["bias"] / r["jack bias"] for r in rows]
    assert_rounds_to(min(ratios), lo, "§10 smallest jackknife reduction")
    assert_rounds_to(max(ratios), hi, "§10 largest jackknife reduction")
    first, last = re.search(r"drifts from (-\d+) to (-\d+)", body).groups()
    assert_rounds_to(rows[0]["N*bias"], first, "§10 N*bias at the smallest N")
    assert_rounds_to(rows[-1]["N*bias"], last, "§10 N*bias at the largest N")
    assert rows[-1]["N"] == int(quoted(body, r"real work at \$N = (\d+)\$"))
    assert 3 * log("ais")["n_replicates"] == int(
        quoted(body, r"\((\d+) replicates per row\)"))


MODE_ROWS = ["broad, covers both", "narrow on left mode", "narrow on right mode"]


@pytest.mark.parametrize("start", MODE_ROWS)
def test_section_10_separated_modes_table(start):
    r = {(x["start"], x["T"]): x for x in log("ais")["separated_modes"]}[(start, 200)]
    cells = row(section("10."), start)
    assert len(cells) == 4
    assert_rounds_to(r["log Z"], cells[0], f"§10 {start} log Z")
    assert_rounds_to(r["err"], cells[1], f"§10 {start} error")
    if r["predicted err"] == "--":
        assert cells[2] == "—"
    else:
        assert cells[2].endswith(f"= {r['predicted err']}$")
        assert_rounds_to(r["err"], r["predicted err"], f"§10 {start} exact miss")
    assert_rounds_to(r["ESS frac"], cells[3], f"§10 {start} ESS fraction")


def test_section_10_modes_prose():
    modes = {(x["start"], x["T"]): x for x in log("ais")["separated_modes"]}
    body = " ".join(section("10.").split()).replace("−", "-")
    assert_rounds_to(modes[("broad, covers both", 200)]["err"],
                     quoted(body, r"right anyway \(error (-[\d.]+)\)"),
                     "§10 broad-start error in prose")
    # "both report a *perfect* effective sample size"
    for start in MODE_ROWS[1:]:
        assert round(modes[(start, 200)]["ESS frac"], 3) == 1.0


# -- Sec. 11: SGHMC and its friction (experiments/sghmc.py) --------------------

CLOSED_FORM_ARMS = ["exact gradient", "noisy, uncorrected", "noisy, corrected"]


def _sci(cell: str) -> tuple[str, int]:
    """'$2.1\\times10^{7}$' -> ('2.1', 7)."""
    m = re.fullmatch(r"\$([\d.]+)\\times10\^\{(-?\d+)\}\$", cell.strip())
    assert m, f"{cell!r} is not a $m\\times10^{{e}}$ cell"
    return m.group(1), int(m.group(2))


def assert_sci_rounds_to(measured: float, cell: str, what: str) -> None:
    mantissa, exp = _sci(cell)
    assert_rounds_to(float(measured) / 10.0 ** exp, mantissa, what)


@pytest.mark.parametrize("arm", CLOSED_FORM_ARMS)
def test_section_11_closed_form_table(arm):
    body = section("11.")
    steps = [number(c) for c in row(body, "$h$")]
    rows = {float(r["step"]): r for r in log("sghmc")["closed_form"]
            if r["arm"] == arm}
    cells = row(body, arm)
    assert len(cells) == len(steps)
    for h, cell in zip(steps, cells):
        if cell == "*refused*":
            assert float(h) not in rows, f"§11 {arm} h={h} ran after all"
            continue
        assert_rounds_to(float(rows[float(h)]["predicted"]), cell,
                         f"§11 {arm} h={h}")


def test_section_11_closed_form_prose():
    d = log("sghmc")
    body = " ".join(section("11.").split())
    cells = d["closed_form"]
    assert len(cells) == int(quoted(body, r"closed form, (\d+) cells, three arms"))
    n_chains, n_draws = re.search(r"(\d+) chains × (\d+) draws\)\. The table",
                                  body).groups()
    assert (int(n_chains), int(n_draws)) == (d["n_chains"], d["n_samples"])
    worst = max(abs(float(r["observed"]) / float(r["predicted"]) - 1)
                for r in cells)
    assert_rounds_to(100 * worst, quoted(body, r"within \*\*([\d.]+)%\*\* of its"),
                     "§11 worst sampled-vs-closed-form cell")
    # "The corrected row is the exact-gradient row *identically*"
    exact = {r["step"]: r["predicted"] for r in cells if r["arm"] == "exact gradient"}
    for r in cells:
        if r["arm"] == "noisy, corrected":
            assert r["predicted"] == exact[r["step"]]
    # the cap 2 gamma / Vhat, and that the one step past it was refused
    cap = 2 * d["friction"] / d["noise_var"]
    assert_rounds_to(cap, quoted(body, r"caps the step at \$2\\gamma/\\hat V = ([\d.]+)\$"),
                     "§11 friction cap")
    ran = {r["step"] for r in cells if r["arm"] == "noisy, corrected"}
    refused = set(exact) - ran
    assert refused and all(float(h) > cap for h in refused)
    assert all(float(h) <= cap for h in ran)


def test_section_11_error_orders():
    o = log("sghmc")["orders"]
    body = " ".join(section("11.").split())
    disc = [float(e) for e in o["discretization_error"]]
    noisy = [float(e) for e in o["uncorrected_error"]]
    steps = o["steps"]
    assert steps[-1] == float(quoted(body, r"by \$h = ([\d.]+)\$ the second"))
    assert_rounds_to(noisy[-1] / disc[-1],
                     quoted(body, r"the second is \*\*(\d+)× the first\*\*"),
                     "§11 miscorrection over discretization")
    g, v = re.search(r"\(at \$\\gamma = (\d+)\$, \$V = (\d+)\$\)", body).groups()
    assert (float(g), float(v)) == (o["noisy_friction"], o["noise_var"])
    # "over four halvings the discretization term falls 4x each time and the
    # miscorrection term only 2x": the last four halvings, to 0.0125
    halvings = zip(range(len(steps) - 5, len(steps) - 1),
                   range(len(steps) - 4, len(steps)))
    for i, j in halvings:
        assert steps[i] == 2 * steps[j]
        assert disc[i] / disc[j] == pytest.approx(4.0, rel=0.07)
        assert noisy[i] / noisy[j] == pytest.approx(2.0, rel=0.05)
    f = o["friction_sweep"]
    errs = [float(e) for e in f["uncorrected_error"]]
    ratios = [a / b for a, b in zip(errs, errs[1:])]
    lo, hi = re.search(r"cuts the miscorrection term by ([\d.]+)× to ([\d.]+)×",
                       body).groups()
    assert f["step"] == float(quoted(body, r"at \$h = ([\d.]+)\$ each doubling"))
    assert f["frictions"][0] == float(quoted(body, r"from \$\\gamma = ([\d.]+)\$ to"))
    assert f["frictions"][-1] == float(quoted(body, r"\$ to (\d+) cuts"))
    assert_rounds_to(min(ratios), lo, "§11 smallest friction-doubling gain")
    assert_rounds_to(max(ratios), hi, "§11 largest friction-doubling gain")


def test_section_11_no_friction_growth():
    rows = log("sghmc")["no_friction"]
    body = " ".join(section("11.").split())
    printed = re.search(r"steps — ([\d., ]+) against a target variance of 1, "
                        r"at 1k through 16k steps", body).group(1)
    printed = [p.strip() for p in printed.split(",") if p.strip()]
    assert len(printed) == len(rows)
    assert [r["steps"] for r in rows] == [1000 * 2 ** i for i in range(len(rows))]
    for r, p in zip(rows, printed):
        assert_rounds_to(float(r["var"]), p, f"§11 gamma=0 after {r['steps']}")


MATCHED_COLS = ["-", "0.25", "0.5", "1", "2", "4"]


def test_section_11_matched_cost_table():
    body = section("11.")
    rows = {r["friction"]: r for r in log("sghmc")["matched_cost"]}
    header = [ln for ln in body.splitlines() if ln.startswith("| | SGLD")]
    assert len(header) == 1
    frictions = ["-"] + re.findall(r"\$\\gamma\{=\}([\d.]+)\$", header[0])
    assert frictions == MATCHED_COLS
    steps, per_grad, vs = (row(body, "step"), row(body, "ESS / gradient"),
                           row(body, "vs SGLD"))
    sgld = float(rows["-"]["ess"]) / rows["-"]["grad_evals"]
    for f, s, e, x in zip(frictions, steps, per_grad, vs):
        r = rows[f]
        pg = float(r["ess"]) / r["grad_evals"]
        assert_rounds_to(float(r["step"]), s, f"§11 step at friction {f}")
        assert_rounds_to(pg, e, f"§11 ESS/gradient at friction {f}")
        assert x.endswith("×")
        assert_rounds_to(pg / sgld, x[:-1], f"§11 gain over SGLD at friction {f}")


def test_section_11_bold_marks_the_best_friction():
    body = section("11.")
    raw = [ln for ln in body.splitlines()
           if ln.startswith("| ESS / gradient")][0].split("|")[2:-1]
    bold = [i for i, c in enumerate(raw) if "**" in c]
    rows = {r["friction"]: r for r in log("sghmc")["matched_cost"]}
    best = max(MATCHED_COLS,
               key=lambda f: float(rows[f]["ess"]) / rows[f]["grad_evals"])
    assert bold == [MATCHED_COLS.index(best)]


def test_section_11_matched_cost_prose():
    d = log("sghmc")
    body = " ".join(section("11.").split())
    rows = d["matched_cost"]
    lo, hi = re.search(r"measured variances \(([\d.]+) to ([\d.]+)\)", body).groups()
    vars_ = [float(r["observed_var"]) for r in rows]
    assert_rounds_to(min(vars_), lo, "§11 lowest matched-cost variance")
    assert_rounds_to(max(vars_), hi, "§11 highest matched-cost variance")
    assert_rounds_to(1 + d["bias"], quoted(body, r"targeted ([\d.]+) within"),
                     "§11 targeted variance")
    # "all agree ... within their own Monte Carlo error": two standard errors
    for r in rows:
        z = (float(r["observed_var"]) - (1 + d["bias"])) / float(r["var_stderr"])
        assert abs(z) < 2, (r["friction"], z)
    n_chains, n_draws = re.search(r"(\d+) chains × ([\d,]+) draws:", body).groups()
    n_draws = int(n_draws.replace(",", ""))
    # one gradient per step, warmup included (5000 steps)
    for r in rows:
        assert r["grad_evals"] == int(n_chains) * (n_draws + 5000)
    lo, hi = re.search(r"Momentum is worth ([\d.]+)–([\d.]+)×", body).groups()
    sgld = float(rows[0]["ess"]) / rows[0]["grad_evals"]
    gains = [float(r["ess"]) / r["grad_evals"] / sgld for r in rows[1:]]
    assert_rounds_to(min(gains), lo, "§11 smallest momentum gain")
    assert_rounds_to(max(gains), hi, "§11 largest momentum gain")


BNN_ROWS = [("prior draw", 10), ("prior draw", 50), ("MAP fit", 10), ("MAP fit", 50)]


@pytest.mark.parametrize("where,batch", BNN_ROWS)
def test_section_11_bnn_table(where, batch):
    r = {(x["where"], x["batch"]): x for x in log("sghmc")["bnn_noise"]}[(where, batch)]
    body = section("11.")
    hits = [ln for ln in body.splitlines()
            if ln.startswith(f"| {where} | {batch} |")]
    assert len(hits) == 1
    cells = [c.strip() for c in hits[0].strip().strip("|").split("|")][2:]
    assert len(cells) == 3
    for key, cell in zip(["worst_coord_var", "max_step_gamma1",
                          "max_step_any_gamma"], cells):
        assert_sci_rounds_to(float(r[key]), cell, f"§11 {where} b={batch} {key}")


def test_section_11_bnn_prose():
    rows = log("sghmc")["bnn_noise"]
    body = " ".join(section("11.").split())
    n, dim = re.search(r"BNN posterior \((\d+) points, (\d+) weights\)", body).groups()
    assert all((r["n_data"], r["dim"]) == (int(n), int(dim)) for r in rows)
    worst = {(r["where"], r["batch"]): float(r["worst_coord_var"]) for r in rows}
    lo, b_lo, hi, b_hi = re.search(
        r"the MAP fit is (\d+)× \(batch (\d+)\) to (\d+)× \(batch (\d+)\) quieter",
        body).groups()
    ratios = {b: worst[("prior draw", b)] / worst[("MAP fit", b)]
              for b in {r["batch"] for r in rows}}
    assert_rounds_to(ratios[int(b_lo)], lo, "§11 quieter at the large batch")
    assert_rounds_to(ratios[int(b_hi)], hi, "§11 quieter at the small batch")
    assert min(ratios.values()) == ratios[int(b_lo)]
    assert max(ratios.values()) == ratios[int(b_hi)]


# -- Sec. 12: heavy tails (experiments/heavy_tails.py) ------------------------

def _dof_label(dof: float) -> str:
    """The README's first-column spelling: 1.0, 1.25, 1.5, 2.5, 5.0, 30."""
    return "30" if dof == 30.0 else f"{dof:g}" if dof % 1 else f"{dof:.1f}"


def _table(header: str) -> dict[str, list[str]]:
    """The §12 table whose header row starts with `header`, as label -> cells.

    §12's three tables share their first column (the dof), so ``row`` cannot
    tell them apart; this reads one table at a time.
    """
    lines = section("12.").splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith(header)]
    assert len(starts) == 1, f"table {header!r} found {len(starts)} times"
    out = {}
    for ln in lines[starts[0] + 2:]:
        if not ln.startswith("|"):
            break
        cs = [c.strip().replace("**", "").replace("−", "-")
              for c in ln.strip().strip("|").split("|")]
        out[cs[0]] = cs[1:]
    return out


RATE_TABLE = "| $\\nu$ | mean, measured"
COVER_TABLE = "| $\\nu$ | coverage, n=250"
ARMS_TABLE = "| $\\nu$ | ESS/draw: iid"


@pytest.mark.parametrize("dof", [1.0, 1.25, 1.5, 2.5, 5.0, 30.0])
def test_section_12_rate_table(dof):
    r = {x["dof"]: x for x in log("heavy_tails")["rates"]}[dof]
    cells = _table(RATE_TABLE)[_dof_label(dof)]
    assert len(cells) == 5
    for key, cell in zip(["mean rate", "predicted", "P(|X|<=1) rate",
                          "sd rate", "sd predicted"], cells):
        assert_rounds_to(r[key], cell.lstrip("+"), f"§12 dof={dof} {key}")


@pytest.mark.parametrize("dof", [1.0, 1.25, 1.5, 2.5, 5.0, 30.0])
def test_section_12_coverage_table(dof):
    c = {x["dof"]: x for x in log("heavy_tails")["coverage"]}[dof]
    cells = _table(COVER_TABLE)[_dof_label(dof)]
    assert len(cells) == 5
    for key, cell in zip(["cover n=250", "width n=250", "cover n=16000",
                          "width n=16000"], cells):
        assert_rounds_to(c[key], cell, f"§12 dof={dof} {key}")
    assert_rounds_to(c["width n=250"] / c["width n=16000"], cells[4].rstrip("×"),
                     f"§12 dof={dof} width ratio")


def test_section_12_coverage_prose():
    d = log("heavy_tails")
    body = " ".join(section("12.").split())
    covers = [c[k] for c in d["coverage"] for k in ("cover n=250", "cover n=16000")]
    assert_rounds_to(min(covers), quoted(body, r"the lowest cell is ([\d.]+),"),
                     "§12 lowest coverage")
    reps = int(quoted(body, r"over ([\d,]+) replicates a cell").replace(",", ""))
    assert reps == 4 * d["n_replicates"]   # coverage_sweep draws 4x the replicates
    band = 2 * (0.95 * 0.05 / reps) ** 0.5
    assert_rounds_to(band, quoted(body, r"carries about ±([\d.]+) at two"),
                     "§12 two-sigma coverage band")
    # "at nominal or within Monte Carlo error of it at every dof"
    assert all(c >= 0.95 - band for c in covers)
    # "64x the data buys a 2% narrower interval, against the 7.98x ..."
    ns = d["ns"]
    assert ns[-1] // ns[0] == int(quoted(body, r"\*\*(\d+)× the data buys"))
    cov = {c["dof"]: c for c in d["coverage"]}
    shrink = {k: v["width n=250"] / v["width n=16000"] for k, v in cov.items()}
    assert_rounds_to(100 * (1 - 1 / shrink[1.0]),
                     quoted(body, r"buys a (\d+)% narrower interval"),
                     "§12 Cauchy width gain")
    assert_rounds_to(shrink[30.0], quoted(body, r"against the ([\d.]+)× a light"),
                     "§12 light-tailed width ratio")
    assert int(quoted(body, r"\$\\sqrt\{(\d+)\} = \d+\$ is the limit")) == ns[-1] // ns[0]
    assert int(quoted(body, r"\$\\sqrt\{\d+\} = (\d+)\$ is the limit")) ** 2 == ns[-1] // ns[0]
    header = section("12.").splitlines()
    header = [ln for ln in header if ln.startswith(COVER_TABLE)][0]
    assert re.findall(r"n=([\d,]+)", header) == [f"{ns[0]}", f"{ns[-1]:,}"]


@pytest.mark.parametrize("dof", [1.0, 1.5, 5.0, 30.0])
def test_section_12_sampler_table(dof):
    arms = {x["arm"]: x for x in log("heavy_tails")["arms"] if x["dof"] == dof}
    table = _table(ARMS_TABLE)
    assert sorted(table, key=float) == ["1.0", "1.5", "5.0", "30"]
    cells = table[_dof_label(dof)]
    assert len(cells) == 6
    for arm, cell in zip(["iid", "rwm", "hmc"], cells[:3]):
        assert_rounds_to(arms[arm]["ess/draw"], cell, f"§12 dof={dof} {arm} ESS/draw")
    for arm, cell in zip(["iid", "rwm", "hmc"], cells[3:]):
        assert arms[arm]["max|x|"] == int(cell.replace(",", "")), (dof, arm)


def test_section_12_sampler_prose():
    d = log("heavy_tails")
    body = " ".join(section("12.").split())
    arms = {(a["dof"], a["arm"]): a for a in d["arms"]}
    n, chains = re.search(r"At \$n = ([\d{},]+) \\times (\d+)\$ chains", body).groups()
    assert int(re.sub(r"\D", "", n)) == d["mcmc_n"] and int(chains) == d["mcmc_chains"]
    light = float(quoted(body, r"\*\* between \$\\nu = (\d+)\$ and the Cauchy"))
    assert light == max(d["dofs"])
    assert_rounds_to(arms[(light, "hmc")]["ess/draw"] / arms[(1.0, "hmc")]["ess/draw"],
                     quoted(body, r"falls \*\*(\d+)×\*\* between"), "§12 HMC ESS fall")
    # "(the table's rounded cells make it look like 180)"
    table = _table(ARMS_TABLE)
    looks = float(table["30"][2]) / float(table["1.0"][2])
    assert round(looks, -1) == int(quoted(body, r"make it look like (\d+)\)"))

    def reach(dof):
        return arms[(dof, "iid")]["max|x|"] / max(arms[(dof, a)]["max|x|"]
                                                  for a in ("rwm", "hmc"))

    cauchy = int(quoted(body, r"three orders of magnitude \(([\d,]+)×\)").replace(",", ""))
    assert round(reach(1.0)) == cauchy and 1000 <= cauchy < 10000
    assert round(reach(1.5)) == int(quoted(body, r"at \$\\nu = 1\.5\$ the shortfall is (\d+)×"))
    lo, hi = re.search(r"reads ESS/draw of ([\d.]+) to ([\d.]+) at every", body).groups()
    iid = [a["ess/draw"] for a in d["arms"] if a["arm"] == "iid"]
    assert_rounds_to(min(iid), lo, "§12 lowest iid ESS/draw")
    assert_rounds_to(max(iid), hi, "§12 highest iid ESS/draw")
    # "their error on the bounded functional is <= 0.004 at every dof"
    worst = max(a["P(|X|<=1) err"] for a in d["arms"] if a["arm"] != "iid")
    bound = quoted(body, r"bounded functional is ≤ ([\d.]+) at every")
    assert worst <= float(bound)
    assert_rounds_to(worst, bound, "§12 worst sampler error, bounded functional")
    # "coverage 0.996-1.000 in the low-dof MCMC cells" (dof <= 2.5)
    lo, hi = re.search(r"coverage ([\d.]+)–([\d.]+) in the low-dof", body).groups()
    low = [a["cover"] for a in d["arms"] if a["arm"] != "iid" and a["dof"] <= 2.5]
    assert_rounds_to(min(low), lo, "§12 low-dof MCMC coverage, lowest")
    assert_rounds_to(max(low), hi, "§12 low-dof MCMC coverage, highest")
    band = (0.95 * 0.05 / d["mcmc_chains"]) ** 0.5
    assert_rounds_to(band, quoted(body, r"carries about ±([\d.]+), so"),
                     "§12 256-replicate coverage se")
    assert int(quoted(body, r"table is over (\d+) replicates")) == d["mcmc_chains"]
    # "its sample mean at nu = 1.5 is converging at n^{-1/3} and at nu = 1 is
    # not converging at all": the generalized-CLT exponents the log carries
    rates = {r["dof"]: r["predicted"] for r in d["rates"]}
    nu, num, den = re.search(r"sample mean at \$\\nu = ([\d.]+)\$ is converging "
                             r"at \$n\^\{-(\d+)/(\d+)\}\$", body).groups()
    assert rates[float(nu)] == pytest.approx(-int(num) / int(den))
    assert rates[float(quoted(body, r"and at \$\\nu = ([\d.]+)\$ is not converging"))] == 0


def test_section_12_bounded_functional_holds_root_n():
    """'The bounded functional holds n^-1/2 at every nu including the Cauchy.'"""
    rates = [r["P(|X|<=1) rate"] for r in log("heavy_tails")["rates"]]
    assert all(abs(r + 0.5) < 0.05 for r in rates), rates
