#!/usr/bin/env python
"""Cleaned-up detector plots (NotInject excluded). Two deliverables x two heads:

Part 1 — default-threshold OPERATING POINTS (dots only), all defenses, both FP datasets:
    op_ipi.png, op_policy.png
Part 2 — per-FP-dataset ROC curves (faint) with the Part-1 operating-point dots overlaid:
    roc_ipi_by_dataset.png, roc_policy_by_dataset.png   (one panel per FP dataset)

color = defense; marker = FP dataset (● AgentDojo benign, ◆ SWE-rebench).
Float detectors get an AUC curve; binary detectors (DataSentinel, Cygnal) are op-point only.
Granite-think's curve is dashed + ⚠ (prob unreliable in think mode); its op-point (label) is fine.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = "plots"
scores = json.load(open(f"{HERE}/scores.json"))

DEF_COLOR = {
    "protectai-v2": "#2a78d6", "promptguard2-86m": "#1baf7a", "datasentinel-mistral7b": "#eda100",
    "cygnal-code-8b": "#008300", "granite-guardian-4.1-8b": "#4a3aa7",
    "granite-guardian-4.1-8b-think": "#e34948", "stackone-defender-tier2": "#e87ba4",
    "llmjudge-gpt-5.6-luna": "#111111", "llmjudge-gemini-3-flash": "#8a5a2b", "llmjudge-qwen3.5-2b": "#8f8b84", "llmjudge-qwen3.5-9b": "#00a0a8",
}
DEF_LABEL = {
    "protectai-v2": "ProtectAI-v2", "promptguard2-86m": "PromptGuard2-86M",
    "datasentinel-mistral7b": "DataSentinel", "cygnal-code-8b": "Cygnal-8b",
    "granite-guardian-4.1-8b": "Granite-4.1-8b", "granite-guardian-4.1-8b-think": "Granite-think",
    "stackone-defender-tier2": "StackOne-T2", "llmjudge-gpt-5.6-luna": "GPT-5.6-Luna judge",
    "llmjudge-gemini-3-flash": "Gemini-3-Flash judge",
    "llmjudge-qwen3.5-2b": "Qwen3.5-2B judge (local)",
    "llmjudge-qwen3.5-9b": "Qwen3.5-9B judge",
}
SHORT = {"protectai-v2": "ProtectAI", "promptguard2-86m": "PromptGuard",
         "datasentinel-mistral7b": "DataSentinel", "cygnal-code-8b": "Cygnal",
         "granite-guardian-4.1-8b": "Granite", "granite-guardian-4.1-8b-think": "Granite-thk",
         "stackone-defender-tier2": "StackOne", "llmjudge-gpt-5.6-luna": "LLM-judge", "llmjudge-gemini-3-flash": "Gemini-judge", "llmjudge-qwen3.5-2b": "Qwen3.5-2B", "llmjudge-qwen3.5-9b": "Qwen3.5-9B"}
DS_MARKER = {"swe-rebench": "D", "toolathlon": "^"}
DS_LABEL = {"swe-rebench": "SWE-rebench (coding)", "toolathlon": "Toolathlon (admin)"}
ORDER = list(DEF_COLOR)
UNRELIABLE = "granite-guardian-4.1-8b-think"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"

# Dataset footer (so every figure is self-contained). Positives = the attack store;
# negatives = the two realistic benign FP datasets. Counts verified from scores.json.
FOOTER = {
    "ipi": [
        "Positives: attack store — 26,466 recorded IPI attacks over 3 Arena corpora "
        "(ipi_2025 · Q1'26 · Q2'26; 90 behaviors) → ≈29.9k injection-site tool outputs.",
        "Benign negatives (injection-free, clean tool outputs):  "
        "◆ SWE-rebench (coding) — 50 clean OpenHands trajectories on real GitHub issues, 2,811 tool outputs;",
        "▲ Toolathlon (admin) — 108 clean office/admin agent rollouts (5-model pool), 4,286 tool outputs.",
        "(AgentDojo benign dropped — its environment fills contain phishing-flavored fixtures ⇒ not clean negatives.)",
        "LLM judges (GPT-5.6-Luna, Gemini-3-Flash, Qwen3.5-9B, local Qwen3.5-2B) ran on seed-0 stratified",
        "1/5 subsets of all three sets (5,987 sites; 544 / 835 benign) — rates comparable, counts not.",
    ],
    "policy": [
        "Positives: attack store — 26,466 recorded IPI attacks over 3 Arena corpora "
        "(ipi_2025 · Q1'26 · Q2'26; 90 behaviors) → 26,146 decisive harmful-action tool calls.",
        "Benign negatives:  ◆ SWE-rebench (coding) — 2,861 actions;  "
        "▲ Toolathlon (admin) — 2,461 agent actions.",
        "(AgentDojo benign dropped — phishing-flavored environment fixtures ⇒ not clean negatives.)",
        "LLM judges (GPT-5.6-Luna, Gemini-3-Flash, Qwen3.5-9B, local Qwen3.5-2B) ran on seed-0 stratified",
        "1/5 subsets of all three sets (5,227 actions; 554 / 479 benign) — rates comparable, counts not.",
    ],
}


def add_footer(fig, head):
    fig.text(0.5, 0.015, "\n".join(FOOTER[head]), ha="center", va="bottom",
             color=MUTED, fontsize=7.8, linespacing=1.5)


def roc(pos, neg):
    pos = np.sort(np.asarray(pos, float)); neg = np.sort(np.asarray(neg, float))
    thr = np.unique(np.concatenate([pos, neg]))
    thr = np.concatenate([[thr[-1] + 1e-9], thr[::-1], [thr[0] - 1e-9]])
    tpr = 1.0 - np.searchsorted(pos, thr, "left") / len(pos)
    fpr = 1.0 - np.searchsorted(neg, thr, "left") / len(neg)
    o = np.argsort(fpr, kind="stable")
    return fpr[o], tpr[o], float(np.trapezoid(tpr[o], fpr[o]))


def op(pos_dec, neg_dec):
    return float(np.mean(neg_dec)), float(np.mean(pos_dec))


def style_ax(ax, title):
    ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), color=GRID, lw=1.4, zorder=1)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False-positive rate  (benign flagged)", color=INK, fontsize=11)
    ax.set_ylabel("True-positive rate  (attack detected)", color=INK, fontsize=11)
    ax.set_title(title, color=INK, fontsize=12, pad=8)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)


def defense_legend(ax, head, loc="lower right", anchor=None, caveat=True):
    # caveat=True (ROC figures): think curve dashed + ⚠. caveat=False (op-point figures):
    # only reliable label-dots are drawn there, so no dashed / no ⚠.
    def flag(d):
        return caveat and d == UNRELIABLE
    handles = [Line2D([0], [0], color=DEF_COLOR[d], lw=3, ls="--" if flag(d) else "-",
                      label=DEF_LABEL[d] + (" ⚠" if flag(d) else ""))
               for d in ORDER if d in scores[head]]
    kw = dict(title="defense (color)", fontsize=9, title_fontsize=9, frameon=True,
              facecolor="white", edgecolor=GRID)
    if anchor:
        return ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, **kw)
    return ax.legend(handles=handles, loc=loc, **kw)


# ---------- Part 1: operating-point scatter (dots only, both datasets) ----------
def fig_operating(head, out):
    fig, ax = plt.subplots(figsize=(9.2, 8.1))
    fig.subplots_adjust(left=0.09, right=0.72, top=0.93, bottom=0.23)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    style_ax(ax, f"Default-threshold operating points — {'injection (ipi)' if head=='ipi' else 'policy'} head")
    # open (outlined) markers so coincident ●/◆ of one defense both stay visible
    for d in ORDER:
        if d not in scores[head]:
            continue
        e = scores[head][d]
        for ds, neg in e["neg"].items():
            fx, ty = op(e["pos_dec"], neg["dec"])
            ax.scatter([fx], [ty], facecolors="none", edgecolors=DEF_COLOR[d],
                       marker=DS_MARKER[ds], s=170, linewidth=2.4, zorder=5)
    leg1 = defense_legend(ax, head, loc="upper left", anchor=(1.02, 1.0), caveat=False)
    ax.add_artist(leg1)
    ds_used = [ds for ds in DS_MARKER if any(ds in scores[head][d]["neg"] for d in scores[head])]
    mk = [Line2D([0], [0], marker=DS_MARKER[ds], lw=0, markersize=11, markerfacecolor="none",
                 markeredgecolor=MUTED, markeredgewidth=2.0, label=DS_LABEL[ds]) for ds in ds_used]
    ax.legend(handles=mk, title="FP dataset (marker)", loc="upper left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, title_fontsize=9, frameon=True, facecolor="white", edgecolor=GRID)
    add_footer(fig, head)
    fig.savefig(out, dpi=150, facecolor=SURFACE); print("wrote", out)
    plt.close(fig)


# ---------- Part 2: per-dataset ROC (faint) + operating-point dots ----------
def panel(ax, head, ds):
    style_ax(ax, DS_LABEL[ds])
    auc_rows = []
    for d in ORDER:
        if d not in scores[head] or ds not in scores[head][d]["neg"]:
            continue
        e = scores[head][d]; neg = e["neg"][ds]; color = DEF_COLOR[d]
        if e["kind"] == "float" and e["pos_score"] and neg["score"]:
            fpr, tpr, auc = roc(e["pos_score"], neg["score"])
            ax.plot(fpr, tpr, color=color, lw=2.0, alpha=0.45,
                    ls="--" if d == UNRELIABLE else "-", zorder=3)
            auc_rows.append((d, f"{auc:.3f}"))
        fx, ty = op(e["pos_dec"], neg["dec"])
        ax.scatter([fx], [ty], color=color, marker=DS_MARKER[ds], s=130,
                   edgecolor="white", linewidth=1.2, zorder=6)
    # per-panel AUC box (float detectors only)
    if auc_rows:
        lines = ["AUC (faint curve)"]
        for d, a in auc_rows:
            lines.append(f"  {SHORT[d] + (' ⚠' if d == UNRELIABLE else ''):<12} {a}")
        binaries = [SHORT[d] for d in ORDER if d in scores[head]
                    and ds in scores[head][d]["neg"] and scores[head][d]["kind"] == "binary"]
        if binaries:
            lines += ["", "dot only: " + ", ".join(binaries)]
        # the dot is each detector's default decision; only sits ON the curve when that
        # decision is a threshold on the plotted score (encoders). Granite's decision is a
        # yes/no label, so its dot can sit off its prob-swept curve (badly for think ⚠).
        lines += ["", "● = default-threshold decision", "  (Granite = yes/no label →", "   may sit off its prob curve)"]
        ax.text(0.97, 0.03, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.4, family="monospace", color=MUTED,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=GRID, lw=0.8))


def fig_roc_by_dataset(head, out):
    ds_used = [ds for ds in DS_MARKER if any(ds in scores[head][d]["neg"] for d in scores[head])]
    fig, axes = plt.subplots(1, len(ds_used), figsize=(7.4 * len(ds_used), 8.0), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.06, right=0.85, top=0.92, bottom=0.22, wspace=0.12)
    fig.patch.set_facecolor(SURFACE)
    for ax, ds in zip(axes, ds_used):
        ax.set_facecolor(SURFACE)
        panel(ax, head, ds)
    fig.suptitle(f"Detection vs over-defense per FP dataset — {'injection (ipi)' if head=='ipi' else 'policy'} head "
                 f"(faint = ROC curve, dot = default-threshold operating point)", fontsize=12.5, color=INK)
    leg = defense_legend(axes[-1], head, loc="upper left", anchor=(1.03, 1.0))
    axes[-1].add_artist(leg)
    add_footer(fig, head)
    fig.savefig(out, dpi=150, facecolor=SURFACE); print("wrote", out)
    plt.close(fig)


fig_operating("ipi", f"{HERE}/op_ipi.png")
fig_operating("policy", f"{HERE}/op_policy.png")
fig_roc_by_dataset("ipi", f"{HERE}/roc_ipi_by_dataset.png")
fig_roc_by_dataset("policy", f"{HERE}/roc_policy_by_dataset.png")
