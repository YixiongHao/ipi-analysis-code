"""Refined: open->frontier transfer lift (500-attack panel), title/subtitle stripped.

Same three-row figure as planning/open_to_frontier_transfer/plot_panel.py::fig_lift
(top = P(frontier break) dumbbells, middle = uplift ratio, bottom = cost-adjusted uplift;
one column per frontier vendor), but with the suptitle and the explanatory paragraph
removed — only the panels and their per-facet headings ("OpenAI frontier" / "Anthropic
frontier") remain. Reads the committed 500-attack panel results.

Run: python plot_transfer_lift.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "planning" / "open_to_frontier_transfer"
sys.path.insert(0, str(SRC))
from plot_panel import (HALVES, HALF_LABEL, MODELS, PROF, VEND, INK, MUTED,
                        MIN_SCREENED, MIN_PAIRED, _style, _cells)  # noqa: E402
from prices_at_time import rollout_cost, cost_ratio  # noqa: E402

RESULTS = SRC / "results_panel_500.json"
OUT = HERE / "transfer_lift.png"


def fig_lift(d, out):
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.2), sharex="col",
                             gridspec_kw={"height_ratios": [2.3, 1.05, 1.05], "hspace": 0.13})
    axes[1][1].sharey(axes[1][0])
    axes[2][1].sharey(axes[2][0])
    halves = [h for h in HALVES if h in d["halves"]]

    for col_i, vend in enumerate(VEND):
        top, bot, cst = axes[0][col_i], axes[1][col_i], axes[2][col_i]
        _style(top); _style(bot); _style(cst)

        for hi, h in enumerate(halves):
            for prof, cell in _cells(d, h, vend):
                if cell["p_screened"] is None or cell["p_all"] is None:
                    continue
                pi = list(PROF).index(prof)
                x = hi + (pi - 0.5) * 0.30
                col = PROF[prof][0]
                thin = cell["screened"] < MIN_SCREENED
                lo, hi_ = cell["ci_screened"]
                top.vlines(x, lo, hi_, color=col, lw=1.2, alpha=0.45, zorder=2)
                top.plot([x, x], [cell["p_all"], cell["p_screened"]],
                         color=col, lw=2.0, alpha=0.55, zorder=3, solid_capstyle="round")
                top.scatter([x], [cell["p_all"]], s=46, facecolor="white",
                            edgecolor=col, lw=1.8, zorder=4)
                top.scatter([x], [cell["p_screened"]], s=58, zorder=5,
                            facecolor="white" if thin else col, edgecolor=col,
                            lw=1.8 if thin else 0)
                mdl = MODELS.get((h, prof))
                if mdl:
                    txt = f"{mdl}  n={cell['screened']}" if thin else mdl
                    top.text(x, hi_ + 0.015, txt, ha="center", va="bottom", rotation=90,
                             fontsize=6.8, color=col)

        # --- uplift ratio ---------------------------------------------------------------
        bot.axhline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        series = {}
        for prof in PROF:
            pts = []
            for hi, h in enumerate(halves):
                c = d["halves"].get(h, {}).get(f"{prof}->{vend}")
                if c and c["n"] >= MIN_PAIRED and c.get("uplift"):
                    pts.append((hi, c["uplift"], c["screened"] < MIN_SCREENED))
            if pts:
                series[prof] = pts
        any_pt = bool(series)
        other_at = {prof: {x: y for x, y, _ in pts} for prof, pts in series.items()}
        for prof, pts in series.items():
            col = PROF[prof][0]
            bot.plot([x for x, _, _ in pts], [y for _, y, _ in pts], color=col, lw=2.0, zorder=3)
            for x, y, t in pts:
                bot.scatter([x], [y], s=44, zorder=4, facecolor="white" if t else col,
                            edgecolor=col, lw=1.8 if t else 0)
                rival = [other_at[q].get(x) for q in series if q != prof]
                rival = [v for v in rival if v is not None]
                above = (not rival) or y >= max(rival)
                bot.annotate(f"{y:.1f}×", (x, y), textcoords="offset points",
                             xytext=(0, 9 if above else -16),
                             ha="center", fontsize=7.5, color=col)
        if not any_pt:
            bot.text(0.5, 0.5, "no breaks on either side — uplift undefined", ha="center",
                     va="center", transform=bot.transAxes, fontsize=9, color=MUTED, style="italic")
            bot.set_yticks([])
        else:
            bot.set_ylim(bottom=0)

        cst.set_xticks(range(len(halves)))
        cst.set_xticklabels([f"{HALF_LABEL[h]}\n{MODELS.get((h, vend)) or '—'}" for h in halves],
                            fontsize=8)
        # --- cost-adjusted: how many x cheaper PER BREAK, at that half's prices --------
        cst.axhline(1.0, color="#b3403a", lw=1.1, ls=(0, (4, 3)), zorder=2)
        cser = {}
        for prof in PROF:
            pts = []
            for hi, h in enumerate(halves):
                c = d["halves"].get(h, {}).get(f"{prof}->{vend}")
                if not (c and c["n"] >= MIN_PAIRED and c.get("uplift")):
                    continue
                arm = d["arms"].get(f"{h}/{prof}")
                co, cf = rollout_cost(h, prof), rollout_cost(h, vend)
                if not arm or co is None or cf is None:
                    continue
                R = cost_ratio(c["uplift"], arm["asr"], co, cf)
                if R:
                    pts.append((hi, R, c["screened"] < MIN_SCREENED))
            if pts:
                cser[prof] = pts
        ymax = max((y for v in cser.values() for _, y, _ in v), default=1)
        for prof, pts in cser.items():
            col = PROF[prof][0]
            cst.plot([x for x, _, _ in pts], [y for _, y, _ in pts], color=col, lw=2.0, zorder=3)
            for x, y, t in pts:
                cst.scatter([x], [y], s=44, zorder=4, facecolor="white" if t else col,
                            edgecolor=col, lw=1.8 if t else 0)
                rival = [dict((a, b) for a, b, _ in v).get(x)
                         for q, v in cser.items() if q != prof]
                rival = [v for v in rival if v is not None]
                above = (not rival) or y >= max(rival)
                if y < 0.18 * ymax:
                    above = True
                cst.annotate(f"{y:.1f}×", (x, y), textcoords="offset points",
                             xytext=(0, 9 if above else -16), ha="center",
                             fontsize=7.5, color=col)
        if not cser:
            cst.text(0.5, 0.5, "no uplift to price", ha="center", va="center",
                     transform=cst.transAxes, fontsize=9, color=MUTED, style="italic")
            cst.set_yticks([])
        else:
            cst.set_ylim(bottom=0)

        top.set_title(f"{VEND[vend]} frontier", fontsize=13, color=INK, loc="left", pad=10)
        top.set_ylim(0, top.get_ylim()[1] * 1.42)
        if col_i == 0:
            top.set_ylabel("P(frontier model breaks)", fontsize=10, color=MUTED)
            bot.set_ylabel("uplift\n(× unfiltered ASR)", fontsize=9.5, color=MUTED)
            cst.set_ylabel("cost-adjusted\n(× cheaper per break)", fontsize=9.5, color=MUTED)

    handles = [Line2D([], [], color=c, lw=2.2, marker="o", ms=7, label=f"{lab} — filtered")
               for c, lab in PROF.values()]
    handles += [Line2D([], [], color=MUTED, lw=0, marker="o", ms=7, mfc="white",
                       mec=MUTED, mew=1.6, label="unfiltered panel")]
    axes[0][1].legend(handles=handles, frameon=False, fontsize=9, loc="upper right",
                      labelcolor=INK)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.955, bottom=0.055, wspace=0.13)
    fig.savefig(out, dpi=180, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    d = json.loads(RESULTS.read_text())
    fig_lift(d, OUT)
