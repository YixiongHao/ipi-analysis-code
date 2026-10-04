"""Refined: per-behavior CoT-suppression flip rate vs implied downstream reversibility.

Strip plot — each dot is one behavior's directional (pos->neg) flip rate; the horizontal
bar in each column is the across-behavior mean. Two monitors per level.

Population: baseline-positive records, policy head, new (post-2026-07-13) Cygnal.
Per-behavior implied-reversibility from the corrected red-team labels.
Source: classifier-defenses/cot_suppression/prefix_ablation/redteam_judge_join.json
  (n_pos>=20 per monitor; flip = fired at baseline, not with CoT surfaced).

Run: python plot_reversibility_gradient.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import style

HERE = Path(__file__).resolve().parent
SRC = (HERE.parent / "classifier-defenses" / "cot_suppression" / "prefix_ablation"
       / "redteam_judge_join.json")

LEVELS = [2, 3, 4, 5]
LABELS = ["simple\nundo", "recoverable\nw/ effort", "largely\nirreversible", "fully\nirreversible"]
MIN_POS = 20
#         label, per-behavior rate key, baseline-positive key, colour, x-offset
SERIES = [
    ("Cygnal-8b (trained detector)", "cygnal_flip_rate", "n_pos",       style.PALETTE[0], -0.19),
    ("Qwen3.5-9b judge",             "judge_flip_rate",   "judge_n_pos", style.PALETTE[1], +0.19),
]


def main():
    style.apply()
    J = json.load(open(SRC))
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    half = 0.15  # dot-cloud + mean-bar half-width

    for label, ratek, nposk, colour, dx in SERIES:
        means = []  # (cx, mean) per level, for the connecting trend line
        for li, lvl in enumerate(LEVELS):
            vals = [r[ratek] * 100 for r in J
                    if r["implied_reversibility"] == lvl
                    and r.get(nposk, 0) >= MIN_POS and r.get(ratek) is not None]
            if not vals:
                continue
            cx = li + dx
            jitter = rng.uniform(-half * 0.8, half * 0.8, size=len(vals)) if len(vals) > 1 else np.zeros(1)
            ax.scatter(cx + jitter, vals, s=34, color=colour, alpha=0.55,
                       edgecolor="white", linewidth=0.5, zorder=2)
            m = float(np.mean(vals))
            means.append((cx, m))
            ax.plot([cx - half, cx + half], [m, m], color=colour, lw=3, solid_capstyle="round", zorder=4)
            ax.annotate(f"{m:.1f}", (cx, m), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=style.ANNOT_SIZE, color=colour, fontweight="bold")
        # connect the column means to show the reversibility trend
        if len(means) > 1:
            mx, my = zip(*means)
            ax.plot(mx, my, color=colour, lw=1.4, ls=(0, (4, 3)), alpha=0.7, zorder=3)

    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels(LABELS)
    ax.set_xlim(-0.55, len(LEVELS) - 0.45)
    ax.set_ylim(0, 25)
    ax.set_ylabel("CoT-suppression flip rate  (pos->neg, %)")
    ax.set_xlabel("implied (downstream-consequence) reversibility")
    ax.set_box_aspect(1)

    handles = [Line2D([0], [0], marker="o", linestyle="none", color=c, alpha=0.7,
                      markeredgecolor="white", markersize=9, label=lab)
               for lab, *_ , c, _ in SERIES]
    ax.legend(handles=handles, loc="upper right", frameon=False)

    style.title_and_dataset(
        fig,
        "raw pos -> neg flip rate vs reversibility",
        "policy head · 9,271 records · Spearman(reversibility, flip):  Cygnal -0.31 / judge -0.41",
        title_y=0.965, sub_y=0.912,
    )
    fig.subplots_adjust(top=0.86, left=0.13, right=0.96, bottom=0.11)
    out = HERE / "reversibility_gradient.png"
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main()
