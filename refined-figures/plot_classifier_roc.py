"""Refined: classifier-defense ROC + operating points, one figure per head.

Source data: plots/scores.json (built by plots/extract_scores.py).
Positives = attack store injection sites / harmful actions; negatives = the two
benign FP datasets (SWE-rebench coding, Toolathlon admin), one panel each.

Run: python plot_classifier_roc.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import style

HERE = Path(__file__).resolve().parent
scores = json.loads((HERE.parent / "plots" / "scores.json").read_text())

# Two families, two visual channels (lightness + weight):
#   Specialized / trained defenses -> LIGHT tints, thick strokes.
#   LLM-judge baselines            -> DARK shades, thin strokes.
# Lightness (not hue temperature) carries the family; hue only separates members
# within a family, so no two members of one family share a hue neighbourhood.
DEF_COLOR = {
    # specialized defenses (light)
    "protectai-v2": "#f09a86",                    # light salmon
    "promptguard2-86m": "#f2b44e",                # light amber
    "datasentinel-mistral7b": "#aecb5f",          # light olive-green
    "granite-guardian-4.1-8b": "#7cc79c",         # light green
    "granite-guardian-4.1-8b-think": "#82c8d8",   # light cyan (appendix only)
    "stackone-defender-tier2": "#e9a0c8",         # light pink
    "cygnal-code-8b": "#bfa8e2",                  # light violet
    # LLM-judge baselines (dark)
    "llmjudge-gpt-5.6-luna": "#123a5e",           # dark navy
    "llmjudge-gemini-3-flash": "#3c2a6b",         # dark indigo (appendix only)
    "llmjudge-qwen3.5-2b": "#0e4a42",             # dark teal
    "llmjudge-qwen3.5-9b": "#5c2148",             # dark plum (appendix only)
}
DEF_LABEL = {
    "protectai-v2": "ProtectAI-v2", "promptguard2-86m": "PromptGuard2-86M",
    "datasentinel-mistral7b": "DataSentinel", "cygnal-code-8b": "Cygnal-8b",
    "granite-guardian-4.1-8b": "Granite-4.1-8b", "granite-guardian-4.1-8b-think": "Granite-think ⚠",
    "stackone-defender-tier2": "StackOne-T2", "llmjudge-gpt-5.6-luna": "GPT-5.6-Luna judge",
    "llmjudge-gemini-3-flash": "Gemini-3-Flash judge",
    "llmjudge-qwen3.5-2b": "Qwen3.5-2B judge", "llmjudge-qwen3.5-9b": "Qwen3.5-9B judge",
}
ORDER = list(DEF_COLOR)
UNRELIABLE = "granite-guardian-4.1-8b-think"
CYGNAL = "cygnal-code-8b"

# LLM-judge baselines vs specialized defenses -> lightness + line-weight channel.
def is_baseline(d):
    return d.startswith("llmjudge-")


DEF_CURVE_ALPHA, DEF_LW, DEF_MARK_ALPHA = 0.95, 3.0, 1.0      # defenses: light, thick
BASE_CURVE_ALPHA, BASE_LW, BASE_MARK_ALPHA = 0.85, 1.9, 0.95  # baselines: dark, thin

# A light fill needs a dark rim to read against the white panel; a dark fill
# needs the opposite. Marker edges therefore follow the family, not one colour.
LIGHT_EDGE, DARK_EDGE = "#4a4a4a", "white"


def edge(d):
    return DARK_EDGE if is_baseline(d) else LIGHT_EDGE


DATASETS = ["swe-rebench", "toolathlon"]
DS_TITLE = {"swe-rebench": "SWE-rebench (coding agent)",
            "toolathlon": "Toolathlon (tool-use agent)"}

TITLE = {"ipi": "Input Head — Detection vs. False Positives",
         "policy": "Output Head — Detection vs. False Positives"}
DATASET_LINE = {
    "ipi": "Positives: 26,466 arena IPI attacks (≈29.9k injection-site tool outputs)  ·  "
           "Negatives: benign agent tool outputs, one dataset per panel  ·  "
           "LLM judges scored on a stratified 1/5 subset",
    "policy": "Positives: 26,146 harmful-action tool calls from 26,466 arena IPI attacks  ·  "
              "Negatives: benign agent actions, one dataset per panel  ·  "
              "LLM judges scored on a stratified 1/5 subset",
}


def roc(pos, neg):
    pos = np.sort(np.asarray(pos, float)); neg = np.sort(np.asarray(neg, float))
    thr = np.unique(np.concatenate([pos, neg]))
    thr = np.concatenate([[thr[-1] + 1e-9], thr[::-1], [thr[0] - 1e-9]])
    tpr = 1.0 - np.searchsorted(pos, thr, "left") / len(pos)
    fpr = 1.0 - np.searchsorted(neg, thr, "left") / len(neg)
    o = np.argsort(fpr, kind="stable")
    return fpr[o], tpr[o], float(np.trapezoid(tpr[o], fpr[o]))


def panel(ax, head, ds):
    ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), color="#c9c7c1", lw=1.4, zorder=1)
    for d in ORDER:
        e = scores[head].get(d)
        if not e or ds not in e["neg"]:
            continue
        color = DEF_COLOR[d]
        base = is_baseline(d)
        calpha = BASE_CURVE_ALPHA if base else DEF_CURVE_ALPHA
        clw = BASE_LW if base else DEF_LW
        malpha = BASE_MARK_ALPHA if base else DEF_MARK_ALPHA
        # Defenses drawn on top of the faded baselines.
        zbump = 0 if base else 2
        if e["kind"] == "float" and e["pos_score"] and e["neg"][ds]["score"]:
            fpr, tpr, _ = roc(e["pos_score"], e["neg"][ds]["score"])
            ax.plot(fpr, tpr, color=color, lw=clw, alpha=calpha,
                    ls="--" if d == UNRELIABLE else "-", zorder=3 + zbump)
        star = d == CYGNAL
        ax.scatter([np.mean(e["neg"][ds]["dec"])], [np.mean(e["pos_dec"])],
                   color=color, marker="*" if star else "o",
                   s=520 if star else 150, alpha=malpha,
                   edgecolor=edge(d), linewidth=1.4, zorder=(7 if star else 6) + zbump)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_box_aspect(1)  # ROC axes share a 0-1 range -> keep the panels square
    ax.set_title(DS_TITLE[ds], fontsize=style.LABEL_SIZE, pad=10)


def legend_handle(d):
    """Legend swatch for detector d.

    The judge baselines never draw a curve (they are sampled, so they contribute a
    single operating point), and Cygnal returns hard labels -> also a point. Giving
    those a line swatch invites the reader to hunt for a curve that is not there, so
    the dark judge family and the Cygnal star are shown as markers instead.
    """
    star = d == CYGNAL
    if is_baseline(d) or star:
        return Line2D([0], [0], color=DEF_COLOR[d], lw=0,
                      marker="*" if star else "o", markersize=20 if star else 11,
                      markeredgecolor=edge(d), markeredgewidth=1.2,
                      alpha=1.0 if star else BASE_MARK_ALPHA, label=DEF_LABEL[d])
    return Line2D([0], [0], color=DEF_COLOR[d], lw=3.6,
                  ls="--" if d == UNRELIABLE else "-", alpha=1.0, label=DEF_LABEL[d])


def fig_head(head, out_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4), sharex=True, sharey=True)
    for ax, ds in zip(axes, DATASETS):
        panel(ax, head, ds)
        ax.set_xlabel("False-positive rate (benign flagged)")
    axes[0].set_ylabel("True-positive rate (attack detected)")

    # Defenses first (light, thick), then the dark baselines, so the legend
    # reads as two groups.
    handles = []
    for d in sorted((d for d in ORDER if d in scores[head]), key=is_baseline):
        handles.append(legend_handle(d))
    axes[0].legend(handles=handles, loc="lower right", fontsize=style.LEGEND_SIZE - 1,
                   frameon=True, facecolor="white", edgecolor="#c9c7c1",
                   framealpha=0.95, labelspacing=0.35, borderpad=0.6)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.09, wspace=0.06)
    style.title_and_dataset(fig, TITLE[head], None)
    out = HERE / out_name
    fig.savefig(out)
    print(f"wrote {out}")


def _op_point(head, d, ds):
    """Mean operating point (fpr, tpr) of detector d's `head` on dataset ds, or None."""
    e = scores.get(head, {}).get(d)
    if not e or ds not in e["neg"]:
        return None
    return float(np.mean(e["neg"][ds]["dec"])), float(np.mean(e["pos_dec"]))


def panel_combined(ax, ds):
    """Both heads on one panel: input head as ROC curve + circle; output head as an ✕ operating
    point, joined to the same detector's IPI circle by a thin same-colour line."""
    ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), color="#c9c7c1", lw=1.4, zorder=1)
    for d in ORDER:
        if d not in scores.get("ipi", {}) and d not in scores.get("policy", {}):
            continue
        color = DEF_COLOR[d]
        base = is_baseline(d)
        calpha = BASE_CURVE_ALPHA if base else DEF_CURVE_ALPHA
        clw = BASE_LW if base else DEF_LW
        malpha = BASE_MARK_ALPHA if base else DEF_MARK_ALPHA
        zb = 0 if base else 2
        # IPI-head ROC curve (float detectors only)
        ei = scores.get("ipi", {}).get(d)
        if ei and ds in ei["neg"] and ei["kind"] == "float" and ei["pos_score"] and ei["neg"][ds]["score"]:
            fpr, tpr, _ = roc(ei["pos_score"], ei["neg"][ds]["score"])
            ax.plot(fpr, tpr, color=color, lw=clw, alpha=calpha,
                    ls="--" if d == UNRELIABLE else "-", zorder=3 + zb)
        pi, pp = _op_point("ipi", d, ds), _op_point("policy", d, ds)
        if pi and pp:   # thin connector between the two heads of the same detector
            ax.plot([pi[0], pp[0]], [pi[1], pp[1]], color=color, lw=1.0,
                    alpha=(0.40 if base else 0.45), zorder=4 + zb)
        if pi:
            ax.scatter([pi[0]], [pi[1]], color=color, marker="o", s=140, alpha=malpha,
                       edgecolor=edge(d), linewidth=1.3, zorder=6 + zb)
        if pp:
            ax.scatter([pp[0]], [pp[1]], color=color, marker="X", s=170, alpha=malpha,
                       edgecolor=edge(d), linewidth=1.2, zorder=6 + zb)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_box_aspect(1)
    ax.set_title(DS_TITLE[ds], fontsize=style.LABEL_SIZE, pad=10)


def fig_combined(out_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4), sharex=True, sharey=True)
    for ax, ds in zip(axes, DATASETS):
        panel_combined(ax, ds)
        ax.set_xlabel("False-positive rate (benign flagged)")
    axes[0].set_ylabel("True-positive rate (attack detected)")

    present = [d for d in ORDER if d in scores.get("ipi", {}) or d in scores.get("policy", {})]
    color_handles = []
    for d in sorted(present, key=is_baseline):
        color_handles.append(legend_handle(d))
    leg1 = axes[0].legend(handles=color_handles, loc="lower right",
                          fontsize=style.LEGEND_SIZE - 1, frameon=True, facecolor="white",
                          edgecolor="#c9c7c1", framealpha=0.95, labelspacing=0.35, borderpad=0.6)
    axes[0].add_artist(leg1)
    # marker legend: which head is which
    mark_handles = [
        Line2D([0], [0], color="#555555", marker="o", lw=0, markersize=10,
               markeredgecolor="white", label="Input head"),
        Line2D([0], [0], color="#555555", marker="X", lw=0, markersize=11,
               markeredgecolor="white", label="Output head"),
    ]
    # marker legend (which head is which): horizontal, centered just below both panels
    fig.legend(handles=mark_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.005), fontsize=style.LEGEND_SIZE - 1,
               frameon=True, facecolor="white", edgecolor="#c9c7c1", framealpha=0.95,
               columnspacing=1.8, handletextpad=0.4, borderpad=0.5)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.155, wspace=0.06)
    style.title_and_dataset(fig, "Detection vs. False Positives — input & output heads", None)
    out = HERE / out_name
    fig.savefig(out)
    print(f"wrote {out}")


def main():
    style.apply()
    fig_head("ipi", "roc_ipi_head.png")
    fig_head("policy", "roc_policy_head.png")
    fig_combined("roc_combined.png")


if __name__ == "__main__":
    main()
