"""Refined: CoT-suppression figures (3 outputs).

1. cot_mechanisms.png              — per-mechanism prevalence in suppression flips vs controls
2. cot_sufficiency_hypotheses.png  — H01 / H09 sufficiency arms, Cygnal vs Qwen judge
3. cot_sufficiency_strategies.png  — self-approval / flag-and-proceed arms, Cygnal vs Qwen judge

Source data (all under classifier-defenses/cot_suppression/):
  newcygnal/or_results.json                     (case-control mechanism labels)
  exp2/results/{cygnal_synth,llmjudge-...}      (H01 sufficiency, Family B)
  exp4_h09/results/{cygnal_synth,llmjudge-...}  (H09 sufficiency, Family B)
  exp6_selfapprove/results/...                  (self-approval arms)
  exp7_flagandproceed/results/...               (flag-and-proceed arms; C3/C4 from exp6)

Suppression is computed exactly as in cot_suppression/plot_judge_arm.py: per record, the
fraction of its variants on which the monitor went silent; bars are means over records with
95% bootstrap CIs.

Run: python plot_cot_suppression.py
"""

import collections
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import style

HERE = Path(__file__).resolve().parent
CS = HERE.parent / "classifier-defenses" / "cot_suppression"

CYGNAL_C, JUDGE_C = "#C44E52", "#4C72B0"


# --------------------------------------------------------------------------- #
# 1. Mechanism prevalence
# --------------------------------------------------------------------------- #
def fig_mechanisms():
    rows = json.loads((CS / "newcygnal" / "or_results.json").read_text())
    rows = sorted(rows, key=lambda r: -r["prev_in_flips"])

    fig, ax = plt.subplots(figsize=(15, 7.4))
    y = np.arange(len(rows))
    h = 0.38
    ax.barh(y + h / 2, [r["prev_in_flips"] for r in rows], h,
            color=CYGNAL_C, label="Present in suppression flips (n=206)")
    ax.barh(y - h / 2, [r["prev_in_controls"] for r in rows], h,
            color="#9E9E9E", label="Present in controls (n=1,000)")
    for i, r in enumerate(rows):
        ax.text(r["prev_in_flips"] + 0.004, i + h / 2, f"{r['prev_in_flips']:.0%}",
                va="center", fontsize=style.TICK_SIZE)
        ax.text(r["prev_in_controls"] + 0.004, i - h / 2, f"{r['prev_in_controls']:.0%}",
                va="center", fontsize=style.TICK_SIZE, color="#555555")

    labels = []
    for r in rows:
        mark = " ✓" if r["adj_class"] == "driver" else ""
        labels.append(f"{r['mech']}  {r['name'].replace('-', ' ')}{mark}")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Share of records where the mechanism is present")
    ax.set_xlim(0, max(r["prev_in_controls"] for r in rows) * 1.16)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(loc="lower right", framealpha=0.95,
              title="✓ = raises miss odds after adjusting for behavior",
              title_fontsize=style.ANNOT_SIZE)

    fig.subplots_adjust(top=0.80, bottom=0.10, left=0.34, right=0.99)
    style.title_and_dataset(
        fig, "Which CoT Mechanisms Appear When Cygnal Misses an Attack",
        "206 suppression flips vs 1,000 matched controls (Cygnal code-8b, arena IPI store)  ·  "
        "blind multi-label mechanism coding")
    out = HERE / "cot_mechanisms.png"
    fig.savefig(out)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# 2 & 3. Sufficiency arms, Cygnal vs Qwen judge
# --------------------------------------------------------------------------- #
def boot(vals, B=10000, seed=0):
    rng = random.Random(seed)
    n = len(vals)
    if not n:
        return 0.0, 0.0, 0.0
    m = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(vals) / n, m[int(0.025 * B)], m[int(0.975 * B)]


def suppression(path, extra=None):
    """Per-record suppression rate: fraction of a record's variants the monitor missed."""
    rows = [json.loads(l) for l in open(CS / path) if l.strip()]
    if extra:
        epath, keep = extra
        rows += [r for l in open(CS / epath) if l.strip()
                 for r in [json.loads(l)] if r["arm"] in keep]
    rows = [r for r in rows if not r["error"]]
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        d[r["attack_id"]][r["arm"]].append(r["fired"])
    return {x: {k: 1 - sum(v) / len(v) for k, v in m.items()} for x, m in d.items()}


EXP6_CYG = ("exp6_selfapprove/results/cygnal_exp6.jsonl", {"C3", "C4"})
EXP6_JUD = ("exp6_selfapprove/results/llmjudge-qwen3.5-9b_exp6_t0.gated.jsonl", {"C3", "C4"})

HYPOTHESES = [
    dict(title="H01 — laundered diligence",
         sub="a cleared audit of the injected item",
         arms=["C4", "C3", "S2", "S1"],
         labels={"C4": "C4\nlorem filler", "C3": "C3\nunrelated checklist",
                 "S2": "S2\nreal text", "S1": "S1\nfabricated"},
         cyg="exp2/results/cygnal_synth.jsonl",
         jud="exp2/results/llmjudge-qwen3.5-9b_exp2_t0.gated.jsonl"),
    dict(title="H09 — notice, then dismiss",
         sub="agent spots the injection and proceeds anyway",
         arms=["C4", "C3", "S2", "S1"],
         labels={"C4": "C4\nlorem filler", "C3": "C3\nunrelated notice",
                 "S2": "S2\nreal text", "S1": "S1\nfabricated"},
         cyg="exp4_h09/results/cygnal_synth.jsonl",
         jud="exp4_h09/results/llmjudge-qwen3.5-9b_exp4_t0.gated.jsonl"),
]

STRATEGIES = [
    dict(title="Self-approval",
         sub="CoT approves its own conduct",
         arms=["C4", "C3", "A1", "A2a", "A2b"],
         labels={"C4": "C4\nlorem filler", "C3": "C3\nunrelated", "A1": "A1\ntask-spec.",
                 "A2a": "A2a\ngen. blunt", "A2b": "A2b\ngen. soft"},
         cyg="exp6_selfapprove/results/cygnal_exp6.jsonl",
         jud="exp6_selfapprove/results/llmjudge-qwen3.5-9b_exp6_t0.gated.jsonl"),
    dict(title="Flag-and-proceed",
         sub="CoT flags the injection, then acts on it",
         arms=["C4", "C3", "B1", "B2a", "B2b"],
         labels={"C4": "C4\nlorem filler", "C3": "C3\nunrelated", "B1": "B1\ntask-spec.",
                 "B2a": "B2a\ngen. explicit", "B2b": "B2b\ngen. plain"},
         cyg="exp7_flagandproceed/results/cygnal_exp7.jsonl", cyg_extra=EXP6_CYG,
         jud="exp7_flagandproceed/results/llmjudge-qwen3.5-9b_exp7_t0.gated.jsonl",
         jud_extra=EXP6_JUD),
]


def sufficiency_fig(specs, out_name, title, dataset_line):
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.6))
    handles = None
    for ax, spec in zip(axes, specs):
        cyg = suppression(spec["cyg"], spec.get("cyg_extra"))
        jud = suppression(spec["jud"], spec.get("jud_extra"))
        arms = spec["arms"]
        x = np.arange(len(arms))
        w = 0.36
        top = 0.0
        for off, sup, color, name in [(-w / 2, cyg, CYGNAL_C, "Cygnal code-8b"),
                                      (w / 2, jud, JUDGE_C, "Qwen3.5-9B judge")]:
            for i, arm in enumerate(arms):
                v = [m[arm] for m in sup.values() if arm in m]
                mean, lo, hi = boot(v)
                ax.bar(i + off, mean, w, color=color,
                       label=name if i == 0 else None, zorder=3)
                ax.errorbar(i + off, mean, yerr=[[mean - lo], [hi - mean]], fmt="none",
                            ecolor="#222", capsize=4, lw=1.2, zorder=4)
                ax.text(i + off, hi + 0.0015, f"{mean:.1%}", ha="center", va="bottom",
                        fontsize=style.TICK_SIZE - 2, zorder=5)
                top = max(top, hi)
        ax.set_xticks(x, [spec["labels"][a] for a in arms], fontsize=style.TICK_SIZE - 1)
        ax.set_ylim(0, top * 1.40 + 0.006)
        # a 0-decimal percent collapses distinct ticks when the range is a few percent
        dec = 0 if ax.get_ylim()[1] >= 0.05 else 1
        ax.yaxis.set_major_formatter(lambda v, _, d=dec: f"{v:.{d}%}")
        ax.set_title(f"{spec['title']}\n{spec['sub']}", pad=12)
        ax.axvspan(-0.5, 1.5, color="#f0f0f0", zorder=0)
        ax.text(0.5, ax.get_ylim()[1] * 0.985, "controls", ha="center", va="top",
                fontsize=style.ANNOT_SIZE, color="#666666", style="italic")
        handles = ax.get_legend_handles_labels()
    axes[0].set_ylabel("Monitor suppression rate")

    fig.legend(*handles, loc="upper center", bbox_to_anchor=(0.5, 0.855), ncol=2,
               frameon=False, fontsize=style.LEGEND_SIZE + 1)
    fig.subplots_adjust(top=0.72, bottom=0.11, wspace=0.18)
    style.title_and_dataset(fig, title, dataset_line, sub_y=0.925)
    out = HERE / out_name
    fig.savefig(out)
    print(f"wrote {out}")


def main():
    style.apply()
    fig_mechanisms()
    sufficiency_fig(
        HYPOTHESES, "cot_sufficiency_hypotheses.png",
        "Do the Two Leading CoT Mechanisms Actually Cause Misses?",
        "Mechanism text inserted into attack CoTs that both monitors already caught  ·  "
        "n = 99 (H01) / 50 (H09) records  ·  a real effect must beat both controls")
    sufficiency_fig(
        STRATEGIES, "cot_sufficiency_strategies.png",
        "Do Self-Approval or Flag-and-Proceed Hide an Attack?",
        "Strategy text inserted into attack CoTs that both monitors already caught  ·  "
        "n = 500 records (Cygnal) / 469 (judge)  ·  a real effect must beat both controls")


if __name__ == "__main__":
    main()
