#!/usr/bin/env python
"""StackOne Defender Tier 2 vs Cygnal (ipi head) — detection-vs-FP, one panel per clean FP set.

Positives  = attack store injection sites (after_tool_response @ containsIPI turns),
             classifier-defenses/results_ipi/<defense>.jsonl (shared across FP sets).
Negatives  = 2 CLEAN benign FP sets (any flag = FP), per-site tool outputs, one PANEL each:
             SWE-rebench coding (fp_rebench)     — Defender 2,811 / Cygnal 1,479 sites
             Toolathlon admin  (fp_toolathlon)   — 4,286 sites
(AgentDojo benign is DROPPED: its environment fills carry phishing-flavored fixtures ⇒ not clean
negatives, and it is StackOne's training set ⇒ contaminated. Both reasons exclude it.)

Defender is a FLOAT detector: we SWEEP its single-head threshold over saved per-site `positive_prob`
(main score) to trace an ROC per FP set (post-hoc, no re-run). Two shipped operating points: single-head
@0.64 and the multi-head rule (main>=0.5 AND aux<0.64). Cygnal returns HARD LABELS only -> 1 point/set.

Outputs (this dir): defender_vs_cygnal_roc.png (1x2, one panel per FP set), detection_by_corpus.png,
summary.json
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
DEF = "stackone-defender-tier2"
CYG = "cygnal-code-8b"
TR = "after_tool_response"

FPSETS = {
    "SWE-rebench (coding)": {"dir": "fp_rebench/results"},
    "Toolathlon (admin)":   {"dir": "fp_toolathlon/results"},
}

corpus_of = {json.loads(l)["attack_id"]: json.loads(l)["corpus"]
             for l in open(f"{CD}/store/attacks.jsonl")}


def load_sites(path, kind):
    out = []
    if not os.path.exists(path):
        return out
    for l in open(path):
        r = json.loads(l)
        c = corpus_of.get(r["attack_id"], "benign")
        for e in r["events"]:
            if e.get("stage") != TR:
                continue
            if kind == "defender":
                out.append((c, e.get("positive_prob", 0.0), e.get("blocked_multihead", False)))
            else:
                out.append((c, 1 if e.get("ipi") else 0, None))
    return out


# ---- positives (shared) ----
def_pos = load_sites(f"{CD}/results_ipi/{DEF}.jsonl", "defender")
cyg_pos = load_sites(f"{CD}/results_ipi/{CYG}.jsonl", "cygnal")
def_pos_s = np.array([s for _, s, _ in def_pos])
def_pos_mh = np.array([mh for _, _, mh in def_pos])
cyg_pos_tpr = float(np.mean([s for _, s, _ in cyg_pos]))
_trap = getattr(np, "trapezoid", getattr(np, "trapz", None))

# ---- per-FP-set sweep + operating points ----
perset = {}
for name, cfg in FPSETS.items():
    def_neg = load_sites(f"{CD}/{cfg['dir']}/{DEF}.jsonl", "defender")
    cyg_neg = load_sites(f"{CD}/{cfg['dir']}/{CYG}.jsonl", "cygnal")
    neg_s = np.array([s for _, s, _ in def_neg])
    neg_mh = np.array([mh for _, _, mh in def_neg])
    thr = np.sort(np.unique(np.r_[def_pos_s, neg_s, 0.0, 1.0]))
    tpr = np.array([(def_pos_s >= t).mean() for t in thr])
    fpr = np.array([(neg_s >= t).mean() for t in thr])
    order = np.argsort(fpr)
    perset[name] = {
        "fpr": fpr, "tpr": tpr, "auc": float(_trap(tpr[order], fpr[order])),
        "n_neg": len(neg_s), "n_cyg_neg": len(cyg_neg),
        "op_single": (float((neg_s >= 0.64).mean()), float((def_pos_s >= 0.64).mean())),
        "op_multi":  (float(neg_mh.mean()),          float(def_pos_mh.mean())),
        "op_cygnal": (float(np.mean([s for _, s, _ in cyg_neg])) if cyg_neg else np.nan, cyg_pos_tpr),
    }

# ================================ figure 1: ROC, 1x2 per FP set ================================
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd7"
C_CURVE, C_SINGLE, C_MULTI, C_CYG = "#c0392b", "#c0392b", "#e07b1a", "#008300"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2), sharey=True)
for ax, (name, p) in zip(axes, perset.items()):
    ax.plot(p["fpr"], p["tpr"], "-", color=C_CURVE, lw=2.2, zorder=3,
            label=f"Defender single-head sweep (AUC={p['auc']:.3f})")
    ax.plot([0, 1], [0, 1], ":", color=GRID, lw=1, zorder=1)
    for op_key, m, col, lab in [
            ("op_single", "o", C_SINGLE, "Defender single-head @0.64 (default)"),
            ("op_multi", "s", C_MULTI, "Defender multi-head (main≥0.5 ∧ aux<0.64)"),
            ("op_cygnal", "D", C_CYG, "Cygnal ipi (hard label — 1 point)")]:
        x, y = p[op_key]
        if np.isnan(x):
            continue
        ax.scatter([x], [y], marker=m, s=140, color=col, edgecolor="white", lw=1.2, zorder=5, label=lab)
        ax.annotate(f"  ({x*100:.1f}%, {y*100:.1f}%)", (x, y), fontsize=8.2, color=col,
                    xytext=(6, -3), textcoords="offset points")
    ax.set_title(name, fontweight="bold", fontsize=12)
    ax.set_xlabel("False-positive rate — benign tool outputs (per-site)")
    ax.set_xlim(-0.006, 0.20); ax.set_ylim(0, 1.0)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.legend(loc="lower right", fontsize=8.4, framealpha=0.95)
axes[0].set_ylabel("Detection — attack-store injection sites (per-site TPR)")
fig.suptitle("StackOne Defender Tier 2 vs Cygnal — detection vs false-positive tradeoff, per clean FP set",
             fontweight="bold", fontsize=13)
foot = (f"Positives: {len(def_pos_s):,} store injection-site tool outputs (3 Arena corpora). "
        f"Negatives (per-site): SWE-rebench Defender {perset['SWE-rebench (coding)']['n_neg']:,} / "
        f"Cygnal {perset['SWE-rebench (coding)']['n_cyg_neg']:,} · "
        f"Toolathlon {perset['Toolathlon (admin)']['n_neg']:,}.  "
        f"Defender curve = post-hoc threshold sweep over saved scores (no re-run); Cygnal = hard label, "
        f"1 point/set.  AgentDojo benign excluded (phishy fixtures + StackOne train set).")
fig.text(0.5, 0.02, foot, ha="center", va="bottom", fontsize=7.0, color=MUTED)
fig.subplots_adjust(bottom=0.17, top=0.88, wspace=0.06)
fig.savefig(f"{HERE}/defender_vs_cygnal_roc.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ======================= figure 2: detection by corpus (positives only, unchanged) =======================
CORP = ["ipi_2025", "ipi_2026_q1", "ipi_2026_q2"]
CLAB = {"ipi_2025": "2025", "ipi_2026_q1": "Q1'26", "ipi_2026_q2": "Q2'26"}
C_DEF = "#c0392b"


def det_by_corpus(sites, is_defender, mh=False):
    d = {}
    for c in CORP:
        if is_defender:
            vals = [(1 if (m if mh else s >= 0.64) else 0) for cc, s, m in sites if cc == c]
        else:
            vals = [s for cc, s, _ in sites if cc == c]
        d[c] = float(np.mean(vals)) if vals else 0.0
    return d


det = {"Defender single@0.64": det_by_corpus(def_pos, True, mh=False),
       "Defender multi-head":  det_by_corpus(def_pos, True, mh=True),
       "Cygnal ipi":           det_by_corpus(cyg_pos, False)}
fig, ax = plt.subplots(figsize=(7.4, 4.8))
colr = {"Defender single@0.64": C_DEF, "Defender multi-head": "#e07b1a", "Cygnal ipi": C_CYG}
x = np.arange(len(CORP)); w = 0.26
for i, s in enumerate(det):
    vals = [det[s][c] * 100 for c in CORP]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=s, color=colr[s], edgecolor="white", lw=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([CLAB[c] for c in CORP])
ax.set_ylabel("Per-site injection detection (%)")
ax.set_title("Injection detection by corpus — Defender vs Cygnal", fontweight="bold")
ax.set_ylim(0, 100); ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7)
ax.legend(fontsize=9, loc="upper right")
fig.text(0.5, 0.005, "Positives only (attack store). Higher = better. Cygnal decays 2025→Q2; "
         "Defender multi-head holds up better on Q1/Q2.", ha="center", fontsize=6.8, color=MUTED)
fig.subplots_adjust(bottom=0.13)
fig.savefig(f"{HERE}/detection_by_corpus.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ================================ summary.json ================================
summary = {
    "positives_store_sites": len(def_pos_s),
    "per_fp_set": {
        n: {"n_neg_defender": perset[n]["n_neg"], "n_neg_cygnal": perset[n]["n_cyg_neg"],
            "defender_auc": round(perset[n]["auc"], 4),
            "op_defender_single@0.64_fpr_tpr": [round(perset[n]["op_single"][0], 4), round(perset[n]["op_single"][1], 4)],
            "op_defender_multihead_fpr_tpr": [round(perset[n]["op_multi"][0], 4), round(perset[n]["op_multi"][1], 4)],
            "op_cygnal_fpr_tpr": [round(perset[n]["op_cygnal"][0], 4), round(perset[n]["op_cygnal"][1], 4)]}
        for n in FPSETS},
    "detection_by_corpus": {s: {c: round(det[s][c], 4) for c in CORP} for s in det},
    "note": "One ROC panel per clean FP set (SWE-rebench coding, Toolathlon admin). Defender ROC = "
            "post-hoc single-head threshold sweep over saved per-site scores; Cygnal = fixed hard-label "
            "operating point. AgentDojo benign excluded (phishing-flavored fixtures + StackOne train set).",
}
json.dump(summary, open(f"{HERE}/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
