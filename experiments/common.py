"""Shared plotting setup and small helpers for the experiment scripts."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def savefig(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.abspath(os.path.join(FIGDIR, name))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def print_table(rows, cols):
    """Minimal aligned table printer: rows = list of dicts, cols = keys."""
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    widths = {c: max(len(c), *(len(fmt(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.rjust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(fmt(r[c]).rjust(widths[c]) for c in cols))
