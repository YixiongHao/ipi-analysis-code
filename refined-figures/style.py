"""Shared matplotlib style for presentation-ready figures.

Conventions (per user request 2026-08-07):
- Short, concise title stating what the graph is; big enough to stand alone on a slide.
- One-line dataset annotation (bare minimum description) below the title or axes.
- Legend sized relative to the plot so it stays readable without dominating.
- No footnotes, no text overlap.
"""

import matplotlib as mpl

TITLE_SIZE = 22
SUBTITLE_SIZE = 13.5
LABEL_SIZE = 15
TICK_SIZE = 13
LEGEND_SIZE = 13
ANNOT_SIZE = 12

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]


def apply():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": TICK_SIZE,
        "axes.titlesize": LABEL_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
    })


def title_and_dataset(fig, title, dataset_line, title_y=0.985, sub_y=None):
    """Big slide-ready suptitle + one-line dataset annotation under it."""
    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold", y=title_y)
    if dataset_line:
        if sub_y is None:
            sub_y = title_y - 0.055
        fig.text(0.5, sub_y, dataset_line, ha="center", va="top",
                 fontsize=SUBTITLE_SIZE, color="#444444")
