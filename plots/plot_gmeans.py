#!/usr/bin/env python
"""Presentation bars: StackOne vs the field, G-means at default untuned thresholds.

Metric per https://yongzx.substack.com/p/cot-monitorability-why-g-means-and :
    G-means = sqrt(TPR x TNR) = sqrt(TPR x (1 - FPR))
Prevalence-free (unlike F1, which drags in the positive/negative mix through
precision), so it compares monitors fairly at their *shipped default thresholds*
without tuning. One G-means per benign FP dataset (the TNR side).

Inputs = plots/scores.json (per-event default-config decisions; ipi head —
StackOne has no policy head). TPR over the attack store's injection-site tool
outputs; FPR over each benign set's tool outputs.

Figures (one PNG per slide; StackOne + Cygnal in color, others muted gray):
    g_block_rate.png                       TPR @ default threshold
    g_fpr_swe_rebench.png / g_fpr_toolathlon.png    FPR per benign set
    g_gmeans_swe_rebench.png / g_gmeans_toolathlon.png   G-means per benign set
Numbers table: gmeans_summary.md
"""
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "plots"
scores = json.load(open(f"{HERE}/scores.json"))

SHORT = {"protectai-v2": "ProtectAI", "promptguard2-86m": "PromptGuard",
         "datasentinel-mistral7b": "DataSentinel", "cygnal-code-8b": "Cygnal",
         "granite-guardian-4.1-8b": "Granite", "granite-guardian-4.1-8b-think": "Granite-think",
         "stackone-defender-tier2": "StackOne", "llmjudge-gpt-5.6-luna": "LLM-judge",
         "llmjudge-gemini-3-flash": "Gemini-judge", "llmjudge-qwen3.5-2b": "Qwen3.5-2B", "llmjudge-qwen3.5-9b": "Qwen3.5-9B"}
# StackOne/Cygnal keep their deck hues (op_*/roc_* figures); the rest are a muted
# wash — identity comes from the axis label on every bar, never from color alone.
HILITE = {"stackone-defender-tier2": "#e87ba4", "cygnal-code-8b": "#008300",
          "llmjudge-gpt-5.6-luna": "#111111",
          "llmjudge-gemini-3-flash": "#8a5a2b"}
MUTE = "#c9c7c1"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
DS_LABEL = {"swe-rebench": "SWE-rebench (coding)", "toolathlon": "Toolathlon (admin)"}

FOOTER_POS = ("Positives: attack store — 26,466 recorded IPI attacks over 3 Arena corpora "
              "(ipi_2025 · Q1'26 · Q2'26) → ≈29.9k injection-site tool outputs (ipi head).")
FOOTER_NEG = {
    "swe-rebench": "Negatives: SWE-rebench (coding) — 50 clean OpenHands trajectories, "
                   "2,811 tool outputs (all detectors on the full 50).",
    "toolathlon": "Negatives: Toolathlon (admin) — 108 clean office/admin agent rollouts "
                  "(5-model pool), 4,286 tool outputs.",
}

# ── metrics at default thresholds (per-event decisions) ────────────────────────
rows = {}
for d, e in scores["ipi"].items():
    tpr = float(np.mean(e["pos_dec"]))
    fpr = {ds: float(np.mean(neg["dec"])) for ds, neg in e["neg"].items()}
    rows[d] = {"tpr": tpr, "fpr": fpr,
               "gmeans": {ds: math.sqrt(tpr * (1.0 - f)) for ds, f in fpr.items()}}


# ── bar figure ─────────────────────────────────────────────────────────────────
def bar_fig(vals, title, sub, out, footer, ylim, pct=True, high_is_good=True):
    order = sorted(vals, key=vals.get, reverse=high_is_good)
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.21)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, d in enumerate(order):
        hi = d in HILITE
        ax.bar(i, vals[d], width=0.62, color=HILITE.get(d, MUTE),
               edgecolor=SURFACE, linewidth=1.5, zorder=3)
        lab = f"{vals[d]*100:.1f}%" if pct else f"{vals[d]:.3f}"
        ax.text(i, vals[d] + ylim * 0.015, lab, ha="center", va="bottom",
                fontsize=10.5 if hi else 9.5, color=INK if hi else MUTED,
                fontweight="bold" if hi else "normal")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SHORT[d] for d in order],
                       fontsize=9.5, color=INK, rotation=20, ha="right")
    for tick, d in zip(ax.get_xticklabels(), order):
        if d in HILITE:
            tick.set_fontweight("bold")
        else:
            tick.set_color(MUTED)
    ax.set_ylim(0, ylim)
    ax.set_title(title, color=INK, fontsize=13.5, pad=22)
    ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha="center", va="bottom",
            color=MUTED, fontsize=9.5)
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    if pct:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    fig.text(0.5, 0.02, footer, ha="center", va="bottom", color=MUTED,
             fontsize=7.8, linespacing=1.5)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)
    plt.close(fig)


SUB_DEFAULT = "per-event decisions at each detector's shipped default threshold (untuned) — ipi head"

bar_fig({d: r["tpr"] for d, r in rows.items()},
        "Attack block rate (TPR)", SUB_DEFAULT,
        f"{HERE}/g_block_rate.png", FOOTER_POS, ylim=1.0)

fpr_ylim = 0.8  # shared across all FP figures so the sets stay comparable
bar_fig({d: (r["fpr"]["swe-rebench"] + r["fpr"]["toolathlon"]) / 2 for d, r in rows.items()},
        "False-positive rate — aggregate (both benign sets, weighted equally)",
        "mean of the two per-set FPRs, default untuned thresholds — ipi head  (lower is better)",
        f"{HERE}/g_fpr_aggregate.png",
        FOOTER_NEG["swe-rebench"] + "\n" + FOOTER_NEG["toolathlon"]
        + "\nAggregate = unweighted mean of the two per-set FPRs (datasets weighted equally, "
        "not pooled by event count).",
        ylim=fpr_ylim, high_is_good=False)
for ds in ("swe-rebench", "toolathlon"):
    bar_fig({d: r["fpr"][ds] for d, r in rows.items()},
            f"False-positive rate — {DS_LABEL[ds]}", SUB_DEFAULT + "  (lower is better)",
            f"{HERE}/g_fpr_{ds.replace('-', '_')}.png", FOOTER_NEG[ds],
            ylim=fpr_ylim, high_is_good=False)
    bar_fig({d: r["gmeans"][ds] for d, r in rows.items()},
            f"G-means — {DS_LABEL[ds]}",
            "G-means = √(TPR × (1 − FPR)), prevalence-free, at default untuned thresholds",
            f"{HERE}/g_gmeans_{ds.replace('-', '_')}.png",
            FOOTER_POS + "\n" + FOOTER_NEG[ds], ylim=1.0, pct=False)

# ── numbers table ──────────────────────────────────────────────────────────────
with open(f"{HERE}/gmeans_summary.md", "w") as f:
    f.write("# G-means at default untuned thresholds\n\n"
            "G-means = sqrt(TPR x (1 - FPR)) "
            "(https://yongzx.substack.com/p/cot-monitorability-why-g-means-and): "
            "prevalence-free, so it compares monitors at their shipped default configs "
            "without threshold tuning. Per-event decisions from `scores.json` "
            "(same convention as op_*/roc_* figures).\n\n"
            "## ipi head (injection detection)\n\n"
            "| defense | TPR | FPR SWE | FPR Tool | G-means SWE | G-means Tool |\n"
            "|---|---|---|---|---|---|\n")
    for d in sorted(rows, key=lambda d: rows[d]["gmeans"]["toolathlon"], reverse=True):
        r = rows[d]
        f.write(f"| {SHORT[d]} | {r['tpr']*100:.1f}% | {r['fpr']['swe-rebench']*100:.1f}% | "
                f"{r['fpr']['toolathlon']*100:.1f}% | {r['gmeans']['swe-rebench']:.3f} | "
                f"{r['gmeans']['toolathlon']:.3f} |\n")
    f.write("\n## policy head (harmful-action detection; only the LLM monitors have one)\n\n"
            "| defense | TPR | FPR SWE | FPR Tool | G-means SWE | G-means Tool |\n"
            "|---|---|---|---|---|---|\n")
    for d, e in scores["policy"].items():
        tpr = float(np.mean(e["pos_dec"]))
        fpr = {ds: float(np.mean(neg["dec"])) for ds, neg in e["neg"].items()}
        f.write(f"| {SHORT[d]} | {tpr*100:.1f}% | {fpr['swe-rebench']*100:.1f}% | "
                f"{fpr['toolathlon']*100:.1f}% | {math.sqrt(tpr*(1-fpr['swe-rebench'])):.3f} | "
                f"{math.sqrt(tpr*(1-fpr['toolathlon'])):.3f} |\n")
print("wrote", f"{HERE}/gmeans_summary.md")
