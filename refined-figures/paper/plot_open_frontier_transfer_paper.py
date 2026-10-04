"""Title-less paper figure: open-weight models as attack proxies for the frontier.

2x2 grid. Columns = frontier vendor (OpenAI | Gemini), shared time axis:
  Row 1  P(frontier model breaks): unfiltered whole-panel rate (grey open circle) vs the rate
         among attacks an open model broke on its one pass (purple = open frontier, green = open
         <40B). The gap is the screening uplift; Wilson 95% CIs on the filtered points.
  Row 2  cost-adjusted transfer effectiveness -- how many times CHEAPER per frontier break an
         attacker who screens first is, at each half's own prices. ratio = uplift * pi*c_f /
         (c_o + pi*c_f): the raw uplift discounted by the screen's own bill (it runs on every
         attack, only pi reach the frontier). Below the 1x rule the screen costs MORE than
         firing the whole panel at the frontier.

Reads planning/open_to_frontier_transfer/results_panel_gemini.json (one analyze_panel.py run over
the xfer_ stores; carries openai, anthropic and gemini cells) + prices_at_time.py. Title-less by
construction (title in the LaTeX caption); vendor names kept as column titles, per convention.

Run:  python plot_open_frontier_transfer_paper.py
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
sys.path.insert(0, str(REPO / "planning/open_to_frontier_transfer"))
from prices_at_time import rollout_cost, cost_ratio  # noqa: E402

RESULTS = REPO / "planning/open_to_frontier_transfer/results_panel_gemini.json"
OUT = BASE / "open_frontier_transfer.png"

HALVES = ["2024H1", "2024H2", "2025H1", "2025H2", "2026H1", "2026H2"]
HLAB = ["2024\nH1", "2024\nH2", "2025\nH1", "2025\nH2", "2026\nH1", "2026\nH2"]
PROF = [("open-frontier", "Open frontier", style.PALETTE[4]),   # purple
        ("open-small",    "Open <40B",     style.PALETTE[2])]    # green
UNF = style.PALETTE[7]   # grey
VENDORS = [("openai", "OpenAI frontier"), ("gemini", "Gemini frontier")]


def cell(d, h, prof, vend):
    return d["halves"].get(h, {}).get(f"{prof}->{vend}")


def main():
    style.apply()
    d = json.loads(RESULTS.read_text())

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.2), sharex="col", sharey="row",
                             gridspec_kw={"height_ratios": [1.0, 1.0]})

    # ---- Row 1: P(break) dumbbell ----
    for col, (vend, vtitle) in enumerate(VENDORS):
        ax = axes[0][col]
        ax.set_title(vtitle, fontsize=style.LABEL_SIZE + 2, fontweight="bold", pad=8)
        for i, h in enumerate(HALVES):
            pa = None
            for prof, _, _ in PROF:
                c = cell(d, h, prof, vend)
                if c and c["p_all"] is not None:
                    pa = c["p_all"]; break
            if pa is not None:
                ax.scatter([i], [pa], s=64, facecolor="white", edgecolor=UNF, lw=2.0, zorder=5)
            for j, (prof, _, col_c) in enumerate(PROF):
                c = cell(d, h, prof, vend)
                if not c or c["p_screened"] is None:
                    continue
                xo = i + (-0.13 if j == 0 else 0.13)
                ps = c["p_screened"]
                lo, hi = c["ci_screened"]
                ax.vlines(xo, lo, hi, color=col_c, lw=1.6, alpha=0.35, zorder=2)
                if pa is not None:
                    ax.plot([xo, xo], [pa, ps], color=col_c, lw=2.2, alpha=0.5, zorder=3)
                ax.scatter([xo], [ps], s=62, color=col_c, edgecolor="white", lw=1.3, zorder=6)
        ax.set_ylim(bottom=0)
        ax.set_xlim(-0.55, len(HALVES) - 0.45)
    axes[0][0].set_ylabel("P(frontier model breaks)")

    # ---- Row 2: cost-adjusted (x cheaper per break) ----
    for col, (vend, _) in enumerate(VENDORS):
        ax = axes[1][col]
        ax.axhline(1.0, color="#b3403a", lw=1.3, ls=(0, (4, 3)), zorder=1)
        for j, (prof, _, col_c) in enumerate(PROF):
            xs, ys = [], []
            for i, h in enumerate(HALVES):
                c = cell(d, h, prof, vend)
                if not c or not c["uplift"]:
                    continue
                co, cf = rollout_cost(h, prof), rollout_cost(h, vend)
                if co is None or cf is None:
                    continue
                r = cost_ratio(c["uplift"], c["screen_rate"], co, cf)
                if r is None:
                    continue
                xs.append(i); ys.append(r)
            if not xs:
                continue
            ax.plot(xs, ys, color=col_c, lw=2.4, marker="o", ms=8, zorder=3)
            for x, y in zip(xs, ys):
                # open<40B (green) always above; open-frontier (purple) below, but above when it
                # sits low (near 0) so the label clears the x-axis.
                dy = 11 if (j == 1 or y < 0.7) else -17
                ax.annotate(f"{y:.1f}×", (x, y), textcoords="offset points", xytext=(0, dy),
                            ha="center", fontsize=style.ANNOT_SIZE - 1, color=col_c, zorder=7)
        ax.set_xticks(range(len(HALVES)))
        ax.set_xticklabels(HLAB)
        ax.set_xlim(-0.55, len(HALVES) - 0.45)
        ax.set_ylim(0, max(4.8, ax.get_ylim()[1]))
    axes[1][0].set_ylabel("cost-adjusted transfer\n(× cheaper per break)")
    axes[1][1].text(len(HALVES) - 0.5, 1.03, "break-even", fontsize=style.ANNOT_SIZE - 1,
                    color="#b3403a", va="bottom", ha="right")

    handles = [Line2D([], [], color=col_c, lw=2.4, marker="o", ms=9, mec="white",
                      label=lab) for _, lab, col_c in PROF]
    handles.append(Line2D([], [], color=UNF, lw=0, marker="o", ms=9, mfc="white", mew=2.0,
                          label="unfiltered whole panel"))

    fig.tight_layout(rect=(0, 0.053, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=style.LEGEND_SIZE, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.4, columnspacing=1.6)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
