#!/usr/bin/env python
"""Does sweeping the AUX threshold (main head fixed) buy anything — across the 2 clean FP sets?

The multi-head rule blocks iff `main >= main_thr AND aux < aux_thr`; the aux head is a
"directive targets a human reader" veto (high aux => rescued). Question, PER benign FP set:
at a FIXED main threshold, is sweeping aux_thr a USEFUL false-positive lever — does it drop
FP faster than detection, and beat simply raising main?

Two CLEAN FP sets (per-site tool outputs, any flag = FP), one panel each:
  SWE-rebench coding (fp_rebench)
  Toolathlon admin (fp_toolathlon)
(AgentDojo benign dropped — phishy environment fixtures + StackOne train set.)

Key contrast this figure surfaces: the aux veto lowers FP on SWE (default multi-head FP < single-head)
but BACKFIRES on Toolathlon (multi-head FP > single-head) — on admin content benign FPs do NOT have
systematically higher aux, so the veto rescues few of them.

Post-hoc over saved per-site main + aux_of_max (no re-run). Positives shared (attack store).
Outputs: aux_sweep.png, aux_sweep.json
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CD = "classifier-defenses"
HERE = os.path.dirname(os.path.abspath(__file__))
TR = "after_tool_response"

FPSETS = {
    "SWE-rebench": {"dir": "fp_rebench/results",    "contam": False},
    "Toolathlon":  {"dir": "fp_toolathlon/results", "contam": False},
}
C_MAIN, C_A05, C_A064, C_CYG = "#7a7a7a", "#c0392b", "#e07b1a", "#008300"


def load(path):
    m, a = [], []
    for l in open(path):
        r = json.loads(l)
        for e in r["events"]:
            if e.get("stage") != TR:
                continue
            m.append(e.get("positive_prob", 0.0))
            a.append(e.get("aux_of_max"))
    m = np.array(m)
    a = np.array([np.nan if x is None else x for x in a], dtype=float)
    return m, a


def cyg_fp(path):
    n = f = 0
    for l in open(path):
        for e in json.loads(l)["events"]:
            if e.get("stage") != TR:
                continue
            n += 1; f += 1 if e.get("ipi") else 0
    return f / n if n else np.nan


pm, pa = load(f"{CD}/results_ipi/stackone-defender-tier2.jsonl")          # positives (shared)
cyg_pm = [json.loads(l) for l in open(f"{CD}/results_ipi/cygnal-code-8b.jsonl")]
cyg_tpr = np.mean([1 if e.get("ipi") else 0 for r in cyg_pm for e in r["events"]
                   if e.get("stage") == TR])


def mannwhitney_auc(pos, neg):
    v = np.r_[pos, neg]
    ranks = v.argsort().argsort().astype(float) + 1
    R = ranks[:len(pos)].sum()
    return (R - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main_sweep(nm):
    thr = np.sort(np.unique(np.r_[pm, nm, 0.0, 1.0]))
    return np.array([(nm >= t).mean() for t in thr]), np.array([(pm >= t).mean() for t in thr])


def aux_sweep(main_thr, nm, na):
    pa_eff = np.where(np.isnan(pa), np.inf, pa)
    na_eff = np.where(np.isnan(na), np.inf, na)
    pass_p = pm >= main_thr
    pass_n = nm >= main_thr
    a_grid = np.linspace(0, 1.0001, 400)
    fpr = np.array([((pass_n) & (na_eff < a)).mean() for a in a_grid])
    tpr = np.array([((pass_p) & (pa_eff < a)).mean() for a in a_grid])
    return fpr, tpr


# ================================ 1x3 figure ================================
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd7"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, len(FPSETS), figsize=(5.7 * len(FPSETS), 5.8), sharey=True)
out = {"per_fp_set": {}}
TARGETS = [0.01, 0.02, 0.03, 0.05, 0.08]


def tpr_at_fpr(fpr_c, tpr_c, targets):
    return {t: (float(tpr_c[fpr_c <= t + 1e-12].max()) if (fpr_c <= t + 1e-12).any() else 0.0)
            for t in targets}


for ax, (name, cfg) in zip(axes, FPSETS.items()):
    nm, na = load(f"{CD}/{cfg['dir']}/stackone-defender-tier2.jsonl")
    m_fpr, m_tpr = main_sweep(nm)
    a05f, a05t = aux_sweep(0.5, nm, na)
    a064f, a064t = aux_sweep(0.64, nm, na)
    ax.plot(m_fpr, m_tpr, "-", color=C_MAIN, lw=1.8, zorder=2)
    ax.plot(a05f, a05t, "-", color=C_A05, lw=2.4, zorder=3)
    ax.plot(a064f, a064t, "-", color=C_A064, lw=2.4, zorder=3)
    # default op-points
    single = ((nm >= 0.64).mean(), (pm >= 0.64).mean())
    multi = (((nm >= 0.5) & ((na < 0.64) | np.isnan(na))).mean(),
             ((pm >= 0.5) & ((pa < 0.64) | np.isnan(pa))).mean())
    ax.scatter(*single, marker="o", s=110, color=C_A05, edgecolor="white", zorder=5)
    ax.scatter(*multi, marker="s", s=120, color=C_A064, edgecolor="white", zorder=5)
    cf = cyg_fp(f"{CD}/{cfg['dir']}/cygnal-code-8b.jsonl")
    ax.scatter([cf], [cyg_tpr], marker="D", s=120, color=C_CYG, edgecolor="white", zorder=5)
    # separation stats
    tp = pa[(pm >= 0.5) & ~np.isnan(pa)]
    fp = na[(nm >= 0.5) & ~np.isnan(na)]
    sepauc = float(mannwhitney_auc(fp, tp)) if len(fp) and len(tp) else float("nan")
    helps = multi[0] < single[0]
    title = name + (" ⚠" if cfg["contam"] else "")
    ax.set_title(f"{title}\naux veto {'lowers' if helps else 'RAISES'} FP: "
                 f"single {single[0]*100:.1f}% → multi {multi[0]*100:.1f}%",
                 fontweight="bold", fontsize=10.5,
                 color=INK if helps else "#c0392b")
    ax.text(0.97, 0.03, f"aux FP>TP sep AUC {sepauc:.2f}\n(>0.5 ⇒ veto rescues FPs)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.6, color=MUTED,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=GRID, lw=0.7))
    ax.set_xlabel(f"FPR — {name} benign (per-site)")
    ax.set_xlim(-0.004, 0.14); ax.set_ylim(0, 0.85)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    out["per_fp_set"][name] = {
        "op_single_fpr_tpr": [round(single[0], 4), round(single[1], 4)],
        "op_multi_fpr_tpr": [round(multi[0], 4), round(multi[1], 4)],
        "aux_veto_lowers_fp": bool(helps),
        "aux_fp>tp_sep_auc": round(sepauc, 3),
        "tpr_at_matched_fpr": {
            "main_sweep": {str(k): round(v, 3) for k, v in tpr_at_fpr(m_fpr, m_tpr, TARGETS).items()},
            "aux_sweep_main0.5": {str(k): round(v, 3) for k, v in tpr_at_fpr(a05f, a05t, TARGETS).items()},
        }}

axes[0].set_ylabel("Detection — store injection sites (per-site TPR)")
handles = [Line2D([0], [0], color=C_MAIN, lw=2, label="main-threshold sweep (single-head ROC)"),
           Line2D([0], [0], color=C_A05, lw=2.4, label="aux sweep @ main≥0.5"),
           Line2D([0], [0], color=C_A064, lw=2.4, label="aux sweep @ main≥0.64"),
           Line2D([0], [0], marker="o", lw=0, color=C_A05, markeredgecolor="white", label="single-head @0.64 (default)"),
           Line2D([0], [0], marker="s", lw=0, color=C_A064, markeredgecolor="white", label="multi-head default"),
           Line2D([0], [0], marker="D", lw=0, color=C_CYG, markeredgecolor="white", label="Cygnal ipi (hard label)")]
fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.4, framealpha=0.96,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("StackOne Defender aux-threshold veto — a real FP lever on SWE-rebench, "
             "but it BACKFIRES on Toolathlon admin", fontweight="bold", fontsize=12.5)
fig.text(0.5, 0.055, "Post-hoc over saved per-site main+aux (no re-run). Positives = attack store "
         "(shared). AgentDojo benign excluded (phishy fixtures + StackOne train set).",
         ha="center", fontsize=7.4, color=MUTED)
fig.subplots_adjust(bottom=0.20, top=0.86, wspace=0.08)
fig.savefig(f"{HERE}/aux_sweep.png", dpi=150, bbox_inches="tight")
plt.close(fig)

out["note"] = ("aux sweep at fixed main = block iff main>=M AND aux<a, swept over saved scores, "
               "per FP set. The multi-head veto lowers FP only where benign FPs have higher aux than "
               "injections (sep AUC > 0.5); on Toolathlon they do not, so the default multi-head rule "
               "raises FP vs single-head.")
json.dump(out, open(f"{HERE}/aux_sweep.json", "w"), indent=2)
print(json.dumps(out, indent=2))
