"""Title-less appendix figure: the Anthropic frontier in the open->frontier transfer study.

Anthropic is kept out of the main figure because it has no measurable frontier at the hard end
(Claude broke 0/986 in 2026) and no de-confounded signal where breaks do exist. This figure shows
why, in two panels:

  Left  P(Claude breaks): unfiltered whole-panel rate (grey open circle) vs the screened rate
        (purple = open frontier, green = open <40B), over the 6 half-year snapshots. 2024 blank
        (every H1-era Claude retired from the API); 2025 H1 = claude-sonnet-4, 2025 H2 =
        claude-opus-4.5; 2026 = claude-fable-5 / claude-opus-5, both 0 breaks (fable-5 additionally
        refuses 58% outright via its content filter -- a refusal, not resistance).
  Right behavior-stratified AUC for the two halves that have any breaks, with the 0.5 chance line:
        the crude uplift is positive (3-4.6x) but the AUC sits at/below chance, so once behavior is
        held fixed an open-model break says nothing about whether Claude breaks.

Reads planning/open_to_frontier_transfer/results_panel_gemini.json. Title-less (title in caption).

Run:  python plot_open_frontier_transfer_anthropic_paper.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
RF = BASE.parent
REPO = RF.parent
sys.path.insert(0, str(RF))
import style  # noqa: E402

RESULTS = REPO / "planning/open_to_frontier_transfer/results_panel_gemini.json"
OUT = BASE / "open_frontier_transfer_anthropic.png"

HALVES = ["2024H1", "2024H2", "2025H1", "2025H2", "2026H1", "2026H2"]
HLAB = ["2024\nH1", "2024\nH2", "2025\nH1", "2025\nH2", "2026\nH1", "2026\nH2"]
USABLE = ["2025H1", "2025H2"]
PROF = [("open-frontier", "Open frontier", style.PALETTE[4]),   # purple
        ("open-small",    "Open <40B",     style.PALETTE[2])]    # green
UNF = style.PALETTE[7]   # grey


def cell(d, h, prof):
    return d["halves"].get(h, {}).get(f"{prof}->anthropic")


def main():
    style.apply()
    d = json.loads(RESULTS.read_text())

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.2),
                                   gridspec_kw={"width_ratios": [1.7, 1.0]})

    # ---- Left: P(break) dumbbell over 6 halves ----
    axL.set_title("Break rate", fontsize=style.LABEL_SIZE + 1, fontweight="bold", pad=8)
    for i, h in enumerate(HALVES):
        pa = None
        for prof, _, _ in PROF:
            c = cell(d, h, prof)
            if c and c["p_all"] is not None:
                pa = c["p_all"]; break
        if pa is None:
            continue
        axL.scatter([i], [pa], s=64, facecolor="white", edgecolor=UNF, lw=2.0, zorder=5)
        for j, (prof, _, col) in enumerate(PROF):
            c = cell(d, h, prof)
            if not c or c["p_screened"] is None:
                continue
            xo = i + (-0.13 if j == 0 else 0.13)
            ps = c["p_screened"]
            lo, hi = c["ci_screened"]
            axL.vlines(xo, lo, hi, color=col, lw=1.6, alpha=0.35, zorder=2)
            axL.plot([xo, xo], [pa, ps], color=col, lw=2.2, alpha=0.5, zorder=3)
            axL.scatter([xo], [ps], s=62, color=col, edgecolor="white", lw=1.3, zorder=6)
    # 2026 annotations: zero breaks, and the fable-5 content-filter share
    axL.annotate("0 breaks\n(58% refused)", (4, 0.0), textcoords="offset points",
                 xytext=(0, 20), ha="center", fontsize=style.ANNOT_SIZE - 1, color="#7a7a7a")
    axL.annotate("0 breaks", (5, 0.0), textcoords="offset points",
                 xytext=(0, 20), ha="center", fontsize=style.ANNOT_SIZE - 1, color="#7a7a7a")
    axL.set_xticks(range(len(HALVES)))
    axL.set_xticklabels(HLAB)
    axL.set_xlim(-0.55, len(HALVES) - 0.45)
    axL.set_ylim(bottom=0)
    axL.set_ylabel("P(Claude breaks)")

    handles = [Line2D([], [], color=col, lw=0, marker="o", ms=9, mec="white", label=lab)
               for _, lab, col in PROF]
    handles.append(Line2D([], [], color=UNF, lw=0, marker="o", ms=9, mfc="white", mew=2.0,
                          label="unfiltered whole panel"))

    # ---- Right: behavior-stratified AUC for the two halves with breaks ----
    axR.set_title("Screen quality (de-confounded)", fontsize=style.LABEL_SIZE + 1,
                  fontweight="bold", pad=8)
    axR.axhline(0.5, color="#b3403a", lw=1.3, ls=(0, (4, 3)), zorder=1)
    axR.text(len(USABLE) - 0.5, 0.505, "chance", fontsize=style.ANNOT_SIZE - 1,
             color="#b3403a", va="bottom", ha="right")
    for j, (prof, lab, col) in enumerate(PROF):
        xs, ys = [], []
        for i, h in enumerate(USABLE):
            c = cell(d, h, prof)
            if not c:
                continue
            a = c["auc_stratified"]
            if a != a:   # NaN
                continue
            xs.append(i); ys.append(a)
        if not xs:
            continue
        axR.plot(xs, ys, color=col, lw=2.4, marker="o", ms=9, mec="white", zorder=3)
        for x, y in zip(xs, ys):
            dy = 11 if j == 1 else -18   # open<40B above, open-frontier below (they nearly touch)
            axR.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, dy),
                         ha="center", fontsize=style.ANNOT_SIZE - 1, color=col)
    axR.set_xticks(range(len(USABLE)))
    axR.set_xticklabels([HLAB[HALVES.index(h)] for h in USABLE])
    axR.set_xlim(-0.4, len(USABLE) - 0.6)
    axR.set_ylim(0.3, 0.8)
    axR.set_ylabel("behavior-stratified AUC")

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=style.LEGEND_SIZE, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.4, columnspacing=1.6)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
