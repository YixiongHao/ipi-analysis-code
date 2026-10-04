#!/usr/bin/env python3
"""Figure: CoT-vs-no-CoT effect on three monitors over the SAME 9,271 CoT-bearing store records.

A — the two CONDITIONAL rates (base-rate free): P(suppress | fired without CoT) and
    P(raise | silent without CoT). This is the comparison that needs no counterfactual.
B — the raw net effect, and the net each monitor WOULD show at the other's baseline firing rate,
    which is what shows the raw sign is largely a base-rate artifact for the two LLM monitors.

  python llm_judge/plot_cot_effect.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SHORT = {"judge Qwen3.5-9B (temp 0)": "LLM judge\nQwen3.5-9B",
         "Cygnal cygnal-code-8b": "Cygnal\ncode-8b (new)",
         "Granite-Guardian-4.1-8b": "Granite\nGuardian 4.1-8b"}
COL = {"judge Qwen3.5-9B (temp 0)": "#1a5276", "Cygnal cygnal-code-8b": "#c0392b",
       "Granite-Guardian-4.1-8b": "#d4a05a"}


def main():
    s = json.load(open(os.path.join(HERE, "cot_effect_summary.json")))
    names = [k for k in SHORT if k in s]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.9),
                                  gridspec_kw={"width_ratios": [1.15, 1]})

    w = 0.34
    for i, k in enumerate(names):
        d = s[k]
        nneg = d["n"] - d["basepos"]
        ps = d["pos->neg"] / max(1, d["basepos"])
        pr = d["neg->pos"] / max(1, nneg)
        ax.bar(i - w / 2, ps, width=w, color=COL[k], zorder=3)
        ax.bar(i + w / 2, pr, width=w, color=COL[k], alpha=0.42, zorder=3,
               edgecolor=COL[k], linewidth=1.2)
        ax.text(i - w / 2, ps + 0.008, f"{ps:.1%}", ha="center", fontsize=9)
        ax.text(i + w / 2, pr + 0.008, f"{pr:.1%}", ha="center", fontsize=9)
        ax.text(i - w / 2, -0.028, f"{d['pos->neg']}/{d['basepos']}", ha="center", fontsize=7.4,
                color="#666")
        ax.text(i + w / 2, -0.028, f"{d['neg->pos']}/{nneg}", ha="center", fontsize=7.4, color="#666")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([SHORT[k] for k in names], fontsize=9)
    ax.set_ylabel("share of that arm's records that flip")
    ax.set_ylim(-0.045, 0.52)
    ax.set_title("A · base-rate-free: what CoT does to each verdict class\n"
                 "solid = SUPPRESSED a detection   ·   hollow = RAISED one", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.text(0.015, 0.965,
            "the judge and Cygnal react the same way — CoT raises ~30% of their\n"
            "misses and suppresses ≤10% of their catches. Granite is inverted.",
            transform=ax.transAxes, fontsize=7.8, color="#555", va="top")

    labels = ["raw\n(own base rate)", "if base+ were\n93.8% (judge's)", "if base+ were\n64.9% (Cygnal's)"]
    w2 = 0.26
    for i, k in enumerate(names):
        d = s[k]
        n, bp = d["n"], d["basepos"]
        ps = d["pos->neg"] / max(1, bp)
        pr = d["neg->pos"] / max(1, n - bp)
        vals = [d["neg->pos"] - d["pos->neg"]]
        for base in (0.938, 0.649):
            vals.append(int(round(pr * n * (1 - base) - ps * n * base)))
        for j, v in enumerate(vals):
            ax2.bar(j + (i - 1) * w2, v, width=w2, color=COL[k], zorder=3,
                    alpha=1.0 if j == 0 else 0.5,
                    label=SHORT[k].replace("\n", " ") if j == 0 else None)
            ax2.text(j + (i - 1) * w2, v + (90 if v >= 0 else -230), f"{v:+d}", ha="center",
                     fontsize=7.6, color="#333")
    ax2.axhline(0, color="#333", lw=1)
    ax2.set_ylim(-3950, 1050)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("net change in records flagged")
    ax2.set_title("B · the raw sign is mostly a base-rate artifact\n"
                  "(re-weighting is illustrative, not a prediction)", fontsize=10.5)
    ax2.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax2.grid(axis="y", alpha=0.25, zorder=0)

    fig.suptitle("Surfacing the agent's own CoT to the monitor — same manipulation, same 9,271 "
                 "CoT-bearing records, three monitors", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(HERE, "fig_cot_effect.png")
    fig.savefig(out, dpi=170)
    print("->", out)


if __name__ == "__main__":
    main()
