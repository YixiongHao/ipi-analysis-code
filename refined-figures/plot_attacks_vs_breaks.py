"""Refined: the TOP HALF of unique-attacks-tried vs breaks-achieved (red-teamer panels only).

Original: planning/user_attack_analysis/fig_attacks_vs_breaks.png (2x3; top row = red-teamers,
bottom row = per-model break-rate bars). Only the top row is rebuilt here, per request.

Source data: planning/user_attack_analysis/summary.json (`per_user`), written by
analyze_users_attacks.py from the graded-submission scans.
Run: python plot_attacks_vs_breaks.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import style

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "planning" / "user_attack_analysis" / "summary.json"

CORPORA = [("2025", "2025 (Q4)", "#2a78d6"),
           ("q1", "2026 Q1", "#eb6834"),
           ("q2", "2026 Q2", "#1baf7a")]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#e4e3dd"
XLIM = (1, 2.2e4)          # shared across panels so the diagonals mean the same thing
YLIM = (0.26, 3.0e3)
ZERO = 0.55                # log axis cannot show y=0; park never-broke users on a rug line


def main():
    style.apply()
    d = json.load(open(SRC))

    fig, axes = plt.subplots(1, 3, figsize=(17.4, 6.6), facecolor=SURFACE)
    for (tag, label, col), ax in zip(CORPORA, axes):
        pu = d["per_user"][tag]
        s = d["summary"][tag]
        xs = [u["uniq_attacks"] for u in pu.values()]
        ys = [u["breaks"] or ZERO for u in pu.values()]
        nzero = sum(1 for u in pu.values() if not u["breaks"])

        ax.set_facecolor(SURFACE)
        for frac, ls in ((1, "-"), (0.1, (0, (5, 3))), (0.01, (0, (1.5, 2.5)))):
            ax.plot(XLIM, [XLIM[0] * frac, XLIM[1] * frac], ls=ls, color=MUTED, lw=1.1,
                    alpha=0.6, zorder=1)
            # label each diagonal where it leaves the panel (top edge, else right edge)
            if XLIM[1] * frac > YLIM[1]:
                x, y, va, ha, dy = YLIM[1] / frac, YLIM[1], "bottom", "center", 3
            else:
                x, y, va, ha, dy = XLIM[1], XLIM[1] * frac, "bottom", "right", 3
            ax.annotate(f"{100*frac:g}%", (x, y), textcoords="offset points",
                        xytext=(0, dy), fontsize=10.5, color=MUTED, va=va, ha=ha)

        ax.axhline(ZERO, color=MUTED, lw=0.8, alpha=0.45, zorder=1)
        ax.scatter(xs, ys, s=32, color=col, alpha=0.55, edgecolor=SURFACE, linewidth=0.7,
                   zorder=3)
        ax.annotate(f"never broke anything — {100*nzero/len(pu):.0f}% of red-teamers",
                    (XLIM[0] * 1.35, ZERO), textcoords="offset points", xytext=(0, -6),
                    fontsize=10, color=MUTED, va="top")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect("equal", adjustable="box")   # equal decades => diagonals read at 45 deg
        ax.set_title(f"{label}   ·   {s['users']} red-teamers, {s['models']} models",
                     fontsize=style.LABEL_SIZE, color=INK, loc="left", pad=10)
        ax.set_xlabel("unique attacks tried", color=INK2)
        if ax is axes[0]:
            ax.set_ylabel("breaks achieved", color=INK2)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2)

    style.title_and_dataset(
        fig,
        "Attacks tried vs breaks landed, per red-teamer",
        "one dot per red-teamer · graded arena submissions · diagonals = 100% / 10% / 1% of "
        "attacks tried that broke a model",
        title_y=0.965, sub_y=0.905,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.10, wspace=0.16)
    out = HERE / "attacks_vs_breaks_users.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
