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

**It covers nine of the fifteen sections so far** (§§1-9; the §5, §7
and §9 scripts take about a minute each).
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
}

# Sections whose experiments do not write a log yet. Listed, not silent.
NOT_YET = [
    "10.", "11.", "12.", "13.", "14.", "15.",
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
