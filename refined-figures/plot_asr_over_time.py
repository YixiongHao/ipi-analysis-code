"""Refined: IPI attack success rate over 11 quarters, two frontier lineages x four attack arms.

Original: strategy_frontier_panel/asr_over_time_v2.png (src/plot_v2.py).

Rebuild changes:
  * one figure-level legend instead of a boxed legend duplicated in both panels
  * shared y-axis, so "the two frontiers converge by 2026Q3" is readable off the figure
  * model names become a clean horizontal row under each panel's own axis (the original
    overprinted them, rotated, on top of the plot area at y=0)
  * 2024Q1/Q2 open is shaded and labelled -- it is BLANK, not zero, and a bare gap reads as zero
  * claude-fable-5's content-filter rate is annotated in place, since its ~1% is a refusal rate
  * the 4-line footer is dropped

Source data: strategy_frontier_panel/asr_over_time_v2.json
Detail + caveats: strategy_frontier_panel/ASR_OVER_TIME.md (RESULTS v2) and QUIRKS.md.
Default reads the committed aggregate. `--by-behavior` re-derives every point as the mean of the
24 per-behavior break rates (equal behavior weight) from the per-record stores, with a
behavior-CLUSTER bootstrap CI -- the interval that matches that estimand, and the one the pooled
Wilson bar deliberately excludes.

Source data: strategy_frontier_panel/asr_over_time_v2.json (default) or
  system-defenses/results/sh24_<modelkey>__<arm>/run-*.jsonl (--by-behavior).
Detail + caveats: strategy_frontier_panel/ASR_OVER_TIME.md (RESULTS v2) and QUIRKS.md.
Run: python plot_asr_over_time.py [--by-behavior]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import style

HERE = Path(__file__).resolve().parent
PANEL = HERE.parent / "strategy_frontier_panel"
SRC = PANEL / "asr_over_time_v2.json"

ARMS = [("baseline_control", "Generic control", "#4a4a48", "o", "-"),
        ("fake_user_assistant", "Fake user/assistant msgs", "#c1272d", "s", "-"),
        ("fake_social_proof", "Fake social proof", "#1f6fb4", "^", "-"),
        ("fake_cot", "Fake chain-of-thought", "#2a8a3e", "D", "-")]
PANELS = [("closed", "Proprietary frontier"), ("open", "Open-weight frontier")]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#e4e3dd"
# The source flags 7 points under one undifferentiated marker, which over-warns on the two
# serving-stack differences and under-warns on the points whose MEANING changes. Split in two:
#   SUBST  -> the model or its serving differs from the quarter's intended pick. Marked "*".
#   the rest are annotated in place, because each says something different about the point:
#     2024Q4+2025Q1 closed  one o1 run plotted at two quarters (verified byte-identical)
#     2026Q2 closed         claude-fable-5 refuses rather than resists (filter rate annotated)
# 2025Q4 closed is ALSO non-standard -- gemini-3-flash is both target and shared judge -- but its
# marker is off the figure by request.
SUBST = {("2024Q4", "closed"),   # o1 served natively (OpenAI 1P), not Azure-via-OpenRouter
         ("2025Q1", "open"),     # DeepSeek-R1 org-blocked; V3-0324 (non-reasoning) substituted
         ("2026Q1", "closed"),   # gpt-5.4 for gpt-5.4-pro (ECI tie)
         ("2026Q3", "open")}     # kimi-k3 served at mxfp4, not fp8
SHORT = {"gpt-4o-2024-05-13": "gpt-4o (05-13)", "gpt-4o-2024-08-06": "gpt-4o (08-06)",
         "gemini-3-flash-preview": "gemini-3-flash", "qwen-2.5-72b-instruct": "qwen-2.5-72b",
         "deepseek-chat": "ds-chat", "deepseek-chat-v3-0324": "ds-chat-v3-0324",
         "deepseek-r1-0528": "ds-r1-0528", "deepseek-v3.2": "ds-v3.2",
         "qwen3-235b-a22b-thinking-2507": "qwen3-235b-think"}
YMAX = 84
N_BOOT = 2000


def by_behavior_rows():
    """Re-derive every point with EQUAL weight per behavior, from the per-record stores.

    Reuses plot_v2's own scorable/dedup/arm-membership rules so the only difference from the
    published figure is the weighting. CI is a behavior-cluster bootstrap: resample the 24
    behaviors with replacement and recompute the macro-average. That is wider than the pooled
    Wilson bar because it carries between-behavior heterogeneity, which is the dominant variance
    source in this corpus and which the Wilson-over-strings interval leaves out by construction.
    """
    sys.path.insert(0, str(PANEL / "src"))
    import plot_v2 as P

    cfg = json.loads((PANEL / "data" / "models_by_quarter.json").read_text())
    meta = {(r["quarter"], r["side"], r["arm"]): r for r in json.load(open(SRC))}
    rng = np.random.default_rng(0)
    out = []
    for q in cfg["quarters"]:
        for side in ("closed", "open"):
            slug = q.get(side)
            if not slug:
                continue
            for arm, *_ in ARMS:
                d = P.SD / f"sh24_{P.modelkey(slug)}__{arm}"
                files = sorted(d.glob("run-*.jsonl"))
                if not files:
                    continue
                want = P.arm_hashes(arm)
                recs = [json.loads(l) for f in files for l in f.open()]
                best = {}
                for r in recs:
                    if r.get("attack_hash") not in want:
                        continue
                    h = r["attack_hash"]
                    if h not in best or P.scorable(r) or not P.scorable(best[h]):
                        best[h] = r
                per = defaultdict(lambda: [0, 0])
                for r in best.values():
                    if P.scorable(r):
                        c = per[r["behavior_id"]]
                        c[0] += bool(r.get("is_break"))
                        c[1] += 1
                if not per:
                    continue
                cell = np.array(list(per.values()), dtype=float)
                macro = float((cell[:, 0] / cell[:, 1]).mean())
                idx = rng.integers(0, len(cell), (N_BOOT, len(cell)))
                boot = (cell[idx, 0] / cell[idx, 1]).mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                m = meta.get((q["quarter"], side, arm), {})
                out.append(dict(quarter=q["quarter"], side=side, model=slug, arm=arm,
                                asr=macro, ci_lo=float(lo), ci_hi=float(hi),
                                flag=m.get("flag"), n_behaviors=len(per),
                                content_filtered=m.get("content_filtered", 0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-behavior", action="store_true",
                    help="equal weight per behavior + behavior-cluster bootstrap CI")
    args = ap.parse_args()

    style.apply()
    rows = by_behavior_rows() if args.by_behavior else json.load(open(SRC))
    quarters = sorted({r["quarter"] for r in rows})
    xi = {q: i for i, q in enumerate(quarters)}

    fig, axes = plt.subplots(2, 1, figsize=(15.4, 10.4), facecolor=SURFACE, sharex=True,
                             sharey=True)
    fig.subplots_adjust(left=0.058, right=0.988, top=0.845, bottom=0.105, hspace=0.30)

    for (side, ptitle), ax in zip(PANELS, axes):
        sub = [r for r in rows if r["side"] == side]
        ax.set_facecolor(SURFACE)

        # blank != zero. Such models existed; none released by 2024-06-30 is still REACHABLE with
        # tool support (delisted, data-policy-blocked, or no tool endpoint) -- QUIRKS.md section 15.
        present = {r["quarter"] for r in sub}
        missing = [q for q in quarters if q not in present]
        if missing:
            ax.axvspan(-0.6, max(xi[q] for q in missing) + 0.5, color=GRID, alpha=0.55, zorder=0)
            ax.text((min(xi[q] for q in missing) + max(xi[q] for q in missing)) / 2, YMAX * 0.5,
                    "no open model of this era\nis still servable with tools", fontsize=11.5,
                    color=INK2, ha="center", va="center", style="italic")

        for arm, label, col, mk, ls in ARMS:
            pts = sorted((r for r in sub if r["arm"] == arm), key=lambda r: xi[r["quarter"]])
            if not pts:
                continue
            xs = [xi[r["quarter"]] for r in pts]
            ys = [100 * r["asr"] for r in pts]
            lo = [100 * (r["asr"] - r["ci_lo"]) for r in pts]
            hi = [100 * (r["ci_hi"] - r["asr"]) for r in pts]
            ax.errorbar(xs, ys, yerr=[lo, hi], color=col, marker=mk, ls=ls, lw=2.0, ms=6.5,
                        capsize=3, elinewidth=1.1, alpha=0.92, zorder=3,
                        label=label if ax is axes[0] else None)

        ax.set_title(ptitle, fontsize=style.LABEL_SIZE, color=INK, loc="left", pad=10)
        ax.set_ylabel("attack success rate (%)", color=INK2, fontsize=13)
        ax.set_xlim(-0.6, len(quarters) - 0.4)
        ax.set_ylim(0, YMAX)
        ax.grid(True, color=GRID, lw=0.8, zorder=1)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=12)

        # model row: one horizontal label per quarter, under this panel's own axis
        by_q = {r["quarter"]: r for r in sub if r["arm"] == "baseline_control"}
        for q, i in xi.items():
            r = by_q.get(q)
            if not r:
                continue
            name = r["model"].split("/")[-1]
            name = SHORT.get(name, name) + ("*" if (q, side) in SUBST else "")
            ax.annotate(name, (i, 0), xytext=(0, -9), textcoords="offset points", ha="center",
                        va="top", fontsize=9, color=INK2, annotation_clip=False)

    # One o1 run sits at both 2024Q4 and 2025Q1 -- identical counts in all four arms. Left unmarked
    # it reads as a flat two-quarter segment, i.e. a trend that was never measured.
    a, b = xi["2024Q4"], xi["2025Q1"]
    ax0 = axes[0]
    ax0.plot([a, a, b, b], [74, 76.5, 76.5, 74], color=MUTED, lw=1.0, zorder=4)
    ax0.text((a + b) / 2, 78, "one o1 run, plotted at both quarters", fontsize=10.5,
             color=INK2, ha="center", va="bottom")

    # claude-fable-5's ASR floor is a refusal rate, not a robustness rate -- say so in place
    fab = next(r for r in rows if r["quarter"] == "2026Q2" and r["side"] == "closed"
               and r["arm"] == "baseline_control")
    axes[0].annotate(f"{fab['content_filtered']}% content-filtered —\nthis is a refusal rate",
                     (xi["2026Q2"], 100 * fab["asr"]), xytext=(len(quarters) - 0.55, 70),
                     textcoords="data", ha="right", va="center", fontsize=11, color=INK2,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9, shrinkA=6, shrinkB=5))

    axes[-1].set_xticks(range(len(quarters)))
    axes[-1].set_xticklabels(quarters, fontsize=12.5, color=INK)
    axes[-1].tick_params(axis="x", pad=26)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=13, frameon=False, labelcolor=INK, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 0.905), handlelength=2.2, columnspacing=2.6)

    sub = ("4 attack arms × 100 strings replayed on every model · 24 behaviors shared by all arms · "
           "n=1 per string · bars = Wilson 95% CI · * = model or serving differs from that "
           "quarter's intended pick")
    if args.by_behavior:
        sub = ("equal weight per behavior (mean of the 24 per-behavior rates) · same 4 arms × 100 "
               "strings · bars = behavior-cluster bootstrap 95% CI · * = model or serving "
               "substitution")
    style.title_and_dataset(
        fig, "IPI attack success against each quarter's frontier model", sub,
        title_y=0.975, sub_y=0.930,
    )
    out = HERE / ("asr_over_time_by_behavior.png" if args.by_behavior else "asr_over_time.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
