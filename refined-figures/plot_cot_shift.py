"""Refined: operating-point shift caused by surfacing the agent's CoT to the monitor.

True-positive vs false-positive shift in detection space for all three monitors.

Both axes are the POLICY head (judging the agent's action), so they compose:
  detection : classifier-defenses/llm_judge/cot_effect_summary.json
              9,271 CoT-bearing attack records, CoT-blind vs CoT-surfaced
  benign FP : classifier-defenses/cot_suppression/fp_cot_exposure/fp_cot_summary.json
              108 benign Toolathlon trajectories -> 2,290 policy action events

All three monitors run the same paired records through the same event enumeration
(events_common._events) and the same CoT injector (run_cygnal_cot.inject_cot); the judge
arm is llmjudge-qwen3.5-9b-t0, the same instrument as the detection side.

Run: python plot_cot_shift.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import style

HERE = Path(__file__).resolve().parent
CD = HERE.parent / "classifier-defenses"
DET = CD / "llm_judge" / "cot_effect_summary.json"
FP = CD / "cot_suppression" / "fp_cot_exposure" / "fp_cot_summary.json"

# colours match the AUC figures (plots/plot_roc.py DEF_COLOR) for cross-figure consistency
#            label, detection key, benign-FP key, colour, label offset, ha
MONITORS = [
    ("Cygnal code-8b", "Cygnal cygnal-code-8b", "cygnal-code-8b", "#008300",
     (14, 10), "left"),
    ("Granite Guardian 4.1-8b", "Granite-Guardian-4.1-8b", "granite-guardian-4.1-8b",
     "#4a3aa7", (8, 20), "right"),
    ("Qwen3.5-9B judge", "judge Qwen3.5-9B (temp 0)", "llmjudge-qwen3.5-9b-t0",
     "#00a0a8", (-6, 20), "right"),
]


def detection_rates(d):
    """(CoT-blind, CoT-surfaced) share of attack records flagged."""
    return d["basepos"] / d["n"], (d["pos->pos"] + d["neg->pos"]) / d["n"]


def main():
    style.apply()
    det = json.loads(DET.read_text())
    fp = json.loads(FP.read_text())

    # Broken x-axis: the interesting operating points all sit below ~11% FP, but the axis
    # really runs to 100% — the narrow right panel shows that endpoint so the zoom is explicit.
    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(15.0, 10.0), sharey=True,
        gridspec_kw={"width_ratios": [11, 1], "wspace": 0.05})

    xmax = 0.0
    for label, dkey, fkey, color, (dx, dy), ha in MONITORS:
        tp0, tp1 = detection_rates(det[dkey])
        pe = fp[fkey]["policy_events"]
        fp0, fp1 = pe["base_fp"] / pe["n"], pe["cot_fp"] / pe["n"]
        xmax = max(xmax, fp0, fp1)
        ax.annotate("", xy=(fp1, tp1), xytext=(fp0, tp0),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.32,head_length=0.7",
                                    color=color, lw=2.8, shrinkA=8, shrinkB=8))
        ax.scatter([fp0], [tp0], s=200, facecolor="white", edgecolor=color,
                   linewidth=2.6, zorder=5)
        ax.scatter([fp1], [tp1], s=200, color=color, edgecolor="white",
                   linewidth=1.6, zorder=5)
        ax.annotate(label, (fp0, tp0), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=style.LABEL_SIZE, color=color, fontweight="bold")

    ax.set_ylabel("Attacks detected")
    ax.set_xlim(-0.006, xmax * 1.20 + 0.012)
    ax.set_ylim(0.08, 1.0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    # right stub: the true 100% end of the false-positive axis (no data lives out here)
    axr.set_xlim(0.985, 1.004)
    axr.set_xticks([1.0], ["100%"])
    axr.spines["left"].set_visible(False)
    axr.tick_params(axis="y", length=0)

    # break marks on the facing edges
    dk = dict(marker=[(-1, -0.55), (1, 0.55)], markersize=11, linestyle="none",
              color="#444444", mec="#444444", mew=1.2, clip_on=False)
    ax.plot([1, 1], [0, 1], transform=ax.transAxes, **dk)
    axr.plot([0, 0], [0, 1], transform=axr.transAxes, **dk)

    ax.scatter([], [], s=180, facecolor="white", edgecolor="#555555", linewidth=2.4,
               label="CoT hidden from monitor")
    ax.scatter([], [], s=180, color="#555555", edgecolor="white", label="CoT surfaced")
    ax.legend(loc="lower left", framealpha=0.95)

    # up-and-left = more attacks caught at fewer false positives
    ax.annotate("better", xy=(0.012, 0.985), xytext=(0.135, 0.90),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="left", va="center", fontsize=style.ANNOT_SIZE + 1,
                color="#777777", style="italic",
                arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.65",
                                color="#999999", lw=2.0))

    fig.subplots_adjust(top=0.91, bottom=0.10, left=0.11, right=0.975)
    fig.text(0.545, 0.028, "False positives on benign agent actions",
             ha="center", fontsize=style.LABEL_SIZE)
    style.title_and_dataset(
        fig, "Effect of Surfacing CoT on Monitor Operating Points", None)
    out = HERE / "cot_shift_operating_point.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
