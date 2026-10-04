"""Refined: how much effort sits between a red-teamer's successive first-time behavior breaks.

Original: planning/strategy_time_trends/q3_user_effort/plot_gaps.py -> fig_user_effort_gaps.png
(3 rows: pooled median / within-user / slope forest).

This rebuild keeps ONLY the within-user row and stacks the three corpora vertically, in flat
panels, so the whole figure is a slide-shaped rectangle. Each gap is divided by that red-teamer's
own median gap, so every person is centred on 1.0 and level differences between people cancel --
the pooled row it replaces was dominated by between-user selection (76% of the 2025 slope, 82% of
Q1's), and the slope forest is a table-shaped result better read from the REPORT.

Source data: planning/strategy_time_trends/q3_user_effort/trends.json
Run: python plot_next_break_cost.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import style

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "planning" / "strategy_time_trends" / "q3_user_effort"

#          label,       tag,   colour,    behaviors, red-teamers
CORPORA = [("2025 (Q4)", "2025", "#2a78d6", 41, 165),
           ("2026 Q1", "q1", "#eb6834", 72, 142),
           ("2026 Q2", "q2", "#1baf7a", 9, 125)]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#e4e3dd"
KCAP = 12
DASH = (0, (4, 2.5))
SERIES = [("records", "all records", "-", 1.0, 6),
          ("records_open", "unbroken-behavior records only", DASH, 0.85, 4.5)]


def main():
    style.apply()
    T = json.load(open(SRC / "trends.json"))

    fig, axes = plt.subplots(3, 1, figsize=(14.6, 8.8), facecolor=SURFACE, sharex=True)
    fig.subplots_adjust(left=0.088, right=0.985, top=0.815, bottom=0.088, hspace=0.30)

    for (label, tag, col, nbeh, nusers), ax in zip(CORPORA, axes):
        ax.set_facecolor(SURFACE)
        ax.axhline(1.0, color=MUTED, lw=1.0, alpha=0.7, zorder=1)
        for metric, slab, ls, al, ms in SERIES:
            wc = T[f"{tag}|dedup|{metric}|within_curve"]
            ks = sorted(int(k) for k in wc)
            m = [wc[str(k)]["median_rel"] for k in ks]
            if metric == "records":
                ax.fill_between(ks, [wc[str(k)]["ci"][0] for k in ks],
                                [wc[str(k)]["ci"][1] for k in ks], color=col, alpha=0.15,
                                linewidth=0)
            ax.plot(ks, m, color=col, lw=2.4, ls=ls, alpha=al, marker="o", ms=ms, zorder=3,
                    label=slab)

        ax.set_xlim(0.4, KCAP + 0.7)
        ax.set_ylim(0, 2.75)
        ax.set_yticks([0, 1, 2])
        ax.text(0.65, 2.34, label, fontsize=14, color=INK, weight="bold", va="center")
        ax.text(0.65, 1.90, f"{nbeh} behaviors · {nusers} red-teamers", fontsize=11,
                color=INK2, va="center")
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=12)

    # parked in the bottom panel: Q2 ends at k=9, so its right-hand side is the only free space
    axes[-1].annotate("1.0 = that red-teamer's own median gap", (KCAP + 0.6, 1.0),
                      textcoords="offset points", xytext=(-4, 9), fontsize=11.5, color=MUTED,
                      ha="right")
    axes[0].legend(fontsize=12, frameon=False, labelcolor=INK, loc="upper center",
                   bbox_to_anchor=(0.60, 1.06), ncol=2, handlelength=2.4, columnspacing=1.8)
    axes[-1].set_xlabel("k — unique behaviors broken so far", color=INK2, fontsize=13)
    fig.text(0.016, 0.45, "gap ÷ that red-teamer's own median gap\n"
                          "median across red-teamers (95% CI)", rotation=90, va="center",
             ha="center", fontsize=13, color=INK2)

    style.title_and_dataset(
        fig,
        "How much effort does the next break cost?",
        # the "each gap ÷ own median gap" half is dropped: the y-axis label already says it
        "chat records + unsuccessful submissions · arena IPI corpora",
        title_y=0.972, sub_y=0.912,
    )
    out = HERE / "next_break_cost.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
