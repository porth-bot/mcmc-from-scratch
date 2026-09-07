"""The counts the README states about this repository, checked against it.

Every other number here is a measurement of a sampler, and the discipline is
that measurements come from seeded code the reader can rerun. Three claims are
not measurements at all -- how many figures ship, how many of them are expected
to come back byte-for-byte, and which single file is allowed to differ -- and
those are the ones nothing was recomputing.

That is not hypothetical. gp-from-scratch's Reproduce block quoted 340 tests
against a suite that collected 366; it drifted because every instrument in
that repo recomputes measurements from logs and none of them looked at a
number about the repo itself. This module is the small guard against the same
thing here: the README's "25 of the 26" against what git actually tracks, and
the one file it excuses against the whitelist ``reproduce.sh`` applies when it
reports drift. Those two lists disagreeing is how a figure would quietly
acquire permission to change.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
REPRODUCE = (ROOT / "reproduce.sh").read_text()


def tracked_figures() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "figures/*.png"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return out.stdout.split()


def test_the_readme_counts_the_figures_that_are_actually_committed():
    m = re.search(r"regenerates (\d+) of the (\d+) committed PNGs", README)
    assert m, "the README no longer states how many PNGs reproduce byte-for-byte"
    exact, total = int(m.group(1)), int(m.group(2))
    tracked = tracked_figures()
    assert total == len(tracked), (
        f"README claims {total} committed figures, git tracks {len(tracked)}"
    )
    excused = re.findall(r'TIMING_FIGURES="([^"]*)"', REPRODUCE)
    assert len(excused) == 1, "reproduce.sh no longer whitelists the wall-clock figure"
    assert exact == total - len(excused[0].split()), (
        "the README's exact/total split no longer matches the number of files "
        "reproduce.sh excuses"
    )


def test_the_figure_the_readme_excuses_is_the_one_reproduce_sh_excuses():
    whitelisted = re.search(r'TIMING_FIGURES="([^"]*)"', REPRODUCE).group(1).split()
    assert whitelisted, "the whitelist is empty; the README still names a file"
    for path in whitelisted:
        assert path in tracked_figures(), f"{path} is whitelisted but not committed"
        name = Path(path).name
        assert f"`{name}`" in README, (
            f"reproduce.sh excuses {name}, which the README does not name as "
            "machine-dependent"
        )


def test_the_script_states_the_figure_count_it_will_report():
    m = re.search(r"all (\d+) committed figures came back", REPRODUCE)
    assert m, "reproduce.sh no longer reports a figure count"
    assert int(m.group(1)) == len(tracked_figures())
