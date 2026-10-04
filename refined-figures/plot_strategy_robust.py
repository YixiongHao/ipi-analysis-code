"""Refined: the bottom-row 'what holds up on robust targets' panels of strategy-by-robustness.

Original: planning/user_attack_analysis/strategy_by_robustness/fig_strategy_by_robustness_no_concealment.png
(bottom row — each corpus's own steepest gradients, recoloured per panel).

This refinement keeps only the bottom-row question — share of each model's breaking attacks,
models ordered most-robust to least — and highlights the SAME three strategies in every panel
with ONE fixed colour each, so recurrence reads across corpora. The three are the strategies that
hold up on the robust end (top_negative in both 2025 and Q1): Fake user/assistant msgs, Forge tool
outputs, Embed in legit context. Each carries its own per-corpus Spearman ρ(ASR, share), so Q2's
divergence (only Fake-user still holds) is visible rather than hidden. Colours match the source
figure's top-row identity. Every other category is drawn faint grey.

Source data: strategy_by_robustness_no_concealment.json
Run: python plot_strategy_robust.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import style

HERE = Path(__file__).resolve().parent
SRC = (HERE.parent / "planning" / "user_attack_analysis" / "strategy_by_robustness"
       / "strategy_by_robustness_no_concealment.json")

CORPORA = [("ipi_2025", "2025 (q4)"), ("ipi_2026_q1", "2026 Q1"), ("ipi_2026_q2", "2026 Q2")]
# strategy key -> (fixed colour [= source top-row identity], short label)
HL = [
    ("Fake User and Assistant Messages", "#1baf7a", "Fake user/assistant msgs"),
    ("Embed in Legitimate Context",      "#eda100", "Embed in legit context"),
    ("Forge Tool or Service Outputs",    "#e87ba4", "Forge tool outputs"),
]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#e4e3dd"
MODEL_SHORT = [("gdm-eval-model-", "gdm-"), ("deepseek-v3.1-terminus", "deepseek-v3.1"),
               ("qwen3-vl-235b-a22b-instruct", "qwen3-vl-235b"), ("qwen3.5-397b-a17b", "qwen3.5-397b"),
               ("mistral-large-2512", "mistral-large"), ("kimi-k2-thinking", "kimi-k2-think"),
               ("Meta-SecAlign-70B", "SecAlign-70B"), ("minimax-m2.5", "minimax-m2.5")]
YMAX = 74


def stagger(ys, gap):
    order = np.argsort(ys)
    out = np.array(ys, float)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def setup(ax, models, show_y):
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, YMAX)
    ax.set_xlim(-0.35, len(models) - 1 + 0.35)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.set_xticks(range(len(models)))
    labs = []
    for m in models:
        s = m["short"]
        for a, b in MODEL_SHORT:
            s = s.replace(a, b)
        labs.append(f"{s}\n{100*m['asr']:.1f}%  n={m['n_attacks']}")
    ax.set_xticklabels(labs, rotation=42, ha="right", fontsize=7.2, color=INK2,
                       rotation_mode="anchor")
    if show_y:
        ax.set_ylabel("share of the model's breaking attacks", fontsize=10, color=INK2)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")


def annotate_ends(ax, x, ends, labels, colors):
    ys = stagger(ends, YMAX * 0.055)
    xr = len(x) - 1
    for y0, y1, lab, col in zip(ends, ys, labels, colors):
        ax.plot([xr + 0.10, xr + 0.30], [y0, y1], color=col, lw=0.8, alpha=0.55,
                clip_on=False, zorder=3)
        ax.plot([xr + 0.34], [y1], marker="o", ms=5, color=col, clip_on=False, zorder=4)
        ax.text(xr + 0.46, y1, lab, fontsize=8.0, color=INK, va="center", clip_on=False)


def main():
    d = json.load(open(SRC))
    hl_keys = {k for k, _, _ in HL}
    rest = [k for k in d["design"]["mean_share_all_categories"] if k not in hl_keys]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 7.2), facecolor=SURFACE)
    for i, ((tag, label), ax) in enumerate(zip(CORPORA, axes)):
        e = d["corpora"][tag]
        models = [m for m in e["models"] if m["plotted"]]
        x = np.arange(len(models))
        setup(ax, models, i == 0)

        for k in rest:                      # faint grey background
            ys = [100 * m["share"].get(k, 0.0) for m in models]
            ax.plot(x, ys, color=MUTED, lw=0.9, alpha=0.28, zorder=1)

        ends, labs, cols, rhos = [], [], [], []
        for k, col, short in HL:
            ys = [100 * m["share"].get(k, 0.0) for m in models]
            ci = [m["share_ci"].get(k, [0, 0]) for m in models]
            ax.fill_between(x, [100 * c[0] for c in ci], [100 * c[1] for c in ci],
                            color=col, alpha=0.10, lw=0, zorder=2)
            ax.plot(x, ys, color=col, lw=2.6, marker="o", ms=5, mec=SURFACE, mew=1.2, zorder=4)
            ends.append(ys[-1])
            cols.append(col)
            labs.append(short)                       # right edge: strategy name only
            rhos.append(e["eligibility"].get(k, {}).get("rho"))
        # names only on the last panel — the three colours are identical across panels, so the
        # shared identity is read once; the left panels keep the width for the crammed x-axis.
        if i == len(CORPORA) - 1:
            annotate_ends(ax, x, ends, labs, cols)

        # panel-specific Spearman ρ, colour-keyed to the same three lines (values differ per panel)
        ax.text(0.975, 0.985, "ρ (ASR, share)", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=MUTED)
        for j, (col, rho) in enumerate(zip(cols, rhos)):
            if rho is None:
                continue
            ax.text(0.975, 0.905 - j * 0.078, f"ρ {rho:+.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9.5, color=col, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.0))

        ttl = (f"{label}   ·   {e['n_models']} models, {e['n_attacks_total']:,} attacks, "
               f"{e['n_behaviors']} behaviors")
        ax.set_title(ttl, fontsize=11, color=INK, loc="left", pad=8)

    style.title_and_dataset(
        fig,
        "Three strategies that hold up on the most robust targets",
        "share of each model's breaking attacks · models ordered most-robust → least (native ASR) · "
        "ρ = Spearman(ASR, share)",
        title_y=0.975, sub_y=0.930,
    )
    fig.subplots_adjust(left=0.055, right=0.880, top=0.800, bottom=0.235, wspace=0.16)
    out = HERE / "strategy_robust_holds.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
