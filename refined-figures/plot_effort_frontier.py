"""Refined: effort vs breaks per model, with the per-model effort frontier.

Original: planning/user_attack_analysis/effort_frontier.py -> fig_effort_frontier.png.
The data build below is a verbatim port of that script's `build()` (same corpora, same
chat/submission id namespaces, same >=80%-of-models frontier eligibility) so the numbers are
identical; only the drawing is rebuilt.

Source data: planning/user_attack_analysis/scan/*.tsv.gz (uncommitted ~40 MB intermediates;
rebuild with scan_ids.py + q1_span_ids.py + fix_q2_chat_ids.py per that REPORT's Reproduce block).
Run: python plot_effort_frontier.py
"""
import datetime
import gzip
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import style

HERE = Path(__file__).resolve().parent
SCAN = HERE.parent / "planning" / "user_attack_analysis" / "scan"

#          label,       submissions,  chats,             submission-id remap
CORPORA = [("2025 (Q4)", "2025__subs", "2025__chats", None),
           ("2026 Q1", "q1__subs", "q1__chats_span", "q1__hash2span"),
           ("2026 Q2", "q2__subs", "q2__chats_fixed", None)]
MIN_MODEL_FRAC = 0.8       # frontier uses only red-teamers who attacked >=80% of the models
MIN_K = 5                  # a model needs a >=5-break frontier to be drawn

FAMILY = [("claude", "Anthropic"), ("gpt-", "OpenAI"), ("openai/", "OpenAI"),
          ("gemini", "Google"), ("gdm-eval", "Google"), ("grok", "xAI"), ("x-ai", "xAI"),
          ("deepseek", "DeepSeek"), ("mistral", "Mistral"), ("llama", "Meta")]
PAL = {"Anthropic": "#2a78d6", "Google": "#eb6834", "OpenAI": "#1baf7a", "xAI": "#eda100",
       "DeepSeek": "#e87ba4", "Mistral": "#4a3aa7", "Meta": "#e34948", "Other": "#8d8c85"}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#e4e3dd"
DOT = "#b9b7b0"            # (red-teamer, model) cloud is context, not the finding
ZERO = 0.55                # log axis cannot show y=0; park never-broke cells on a rug line


def fam(m):
    for k, v in FAMILY:
        if k in m:
            return v
    return "Other"


def rows(tag):
    with gzip.open(f"{SCAN}/{tag}.tsv.gz", "rt") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            yield dict(zip(hdr, line.rstrip("\n").split("\t")))


def ts(r):
    if r.get("created_at"):
        try:
            return datetime.datetime.fromisoformat(r["created_at"]).timestamp()
        except ValueError:
            pass
    return int(r["rec_id"][:8], 16)          # 2025 has no timestamp; ObjectId carries it


def build(stag, ctag, remap):
    rm = {}
    if remap:
        with gzip.open(f"{SCAN}/{remap}.tsv.gz", "rt") as f:
            f.readline()
            for line in f:
                h, s = line.rstrip("\n").split("\t")
                rm[h] = s

    def aid(h):
        return rm.get(h, "ah:" + h) if rm else h

    recs = [(ts(r), r["user_id"], aid(r["attack_id"]), r["model"],
             (r["break"] == "True") or (r["grade"] == "success"))
            for r in rows(stag) if r["model"]]
    join = {r["chat_id"]: aid(r["attack_id"]) for r in rows(stag) if r["chat_id"]}
    for r in rows(ctag):                     # chats: effort only, they carry no grade
        a = r.get("attack_id") or join.get(r["rec_id"], "")
        if a:
            recs.append((ts(r), r["user_id"], a, "", False))
    recs.sort()

    models = sorted({m for _, _, _, m, _ in recs if m})
    by_user = defaultdict(list)
    for t, u, a, m, b in recs:
        by_user[u].append((a, m, b))

    cells, kth = [], defaultdict(list)
    for u, seq in by_user.items():
        umodels = {m for _, m, _ in seq if m}
        if not umodels:
            continue                         # chat-only user: no graded row, nothing to plot
        eligible = len(umodels) >= MIN_MODEL_FRAC * len(models)
        seen, elapsed, nbrk = set(), 0, defaultdict(int)
        for a, m, b in seq:
            if a not in seen:
                seen.add(a)
                elapsed += 1                 # effort clock = unique strings tried SO FAR
            if not (m and b):
                continue
            nbrk[m] += 1
            if eligible:
                kth[m].append((nbrk[m], elapsed))
        for m in umodels:
            cells.append((u, m, elapsed, nbrk[m]))

    frontier = {}
    for m, pts in kth.items():
        best = {}
        for k, e in pts:
            if k not in best or e < best[k]:
                best[k] = e
        run, seq = None, []
        for k in sorted(best):
            run = best[k] if run is None else max(run, best[k])   # k-th cannot precede (k-1)-th
            seq.append((run, k))
        frontier[m] = seq
    return cells, frontier, models


def short(m):
    base, _, suf = m.split("/")[-1].partition(":")
    base = re.sub(r"-20\d{6}$", "", base)          # drop the release-date suffix
    return base[:26] + (":" + suf if suf else "")


def label_ends(ax, ends, gap_px=28):
    """Label each highlighted line in one right-hand column, staggered in display space,
    so crowded end points (2025: five Anthropic lines finish together) never overprint."""
    col_x = max(x for x, *_ in ends) * 1.45
    items = sorted(ends, key=lambda e: ax.transData.transform((e[0], e[1]))[1])
    placed = []
    for x, y, name, col in items:
        y_px = ax.transData.transform((x, y))[1]
        if placed and y_px - placed[-1] < gap_px:
            y_px = placed[-1] + gap_px
        placed.append(y_px)
        y_lab = ax.transData.inverted().transform((0, y_px))[1]
        ax.plot([x, col_x * 0.97], [y, y_lab], color=col, lw=0.8, alpha=0.55, zorder=6)
        ax.text(col_x, y_lab, name, fontsize=9.2, color=col, va="center",
                weight="medium", zorder=7)


def main():
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(19.4, 7.0), facecolor=SURFACE)
    # settle the axes boxes first: label_ends staggers in display space
    fig.subplots_adjust(left=0.048, right=0.988, top=0.80, bottom=0.10, wspace=0.17)

    for (label, stag, ctag, remap), ax in zip(CORPORA, axes):
        print(f"  building {label}", file=sys.stderr)
        cells, frontier, models = build(stag, ctag, remap)
        nzero = sum(1 for *_, b in cells if not b)

        ax.set_facecolor(SURFACE)
        ax.scatter([t for _, _, t, _ in cells], [b or ZERO for *_, b in cells],
                   s=12, color=DOT, alpha=0.5, linewidth=0, zorder=2)
        ax.axhline(ZERO, color=MUTED, lw=0.8, alpha=0.45, zorder=1)

        drawn = {m: s for m, s in frontier.items() if len(s) >= MIN_K}

        def effort_at(seq, k):
            for e, kk in seq:
                if kk >= k:
                    return e
            return None

        # Rank on effort to the 10th break. A model NOBODY reached 10 breaks on has no value and
        # is left out of both label sets -- the original scored it 0, which put two 2025 Anthropic
        # models (nobody got past 8 / 9 breaks) at the *easiest* end, the opposite of the truth.
        rankable = {m: e for m in drawn if (e := effort_at(drawn[m], 10)) is not None}
        ranked = sorted(rankable, key=lambda m: -rankable[m])
        hot = set(ranked[:3] + ranked[-3:])

        ends = []
        for m, seq in drawn.items():
            col = PAL[fam(m)]
            xs, ys = [e for e, _ in seq], [k for _, k in seq]
            is_hot = m in hot
            ax.step(xs, ys, where="post", color=col, lw=2.4 if is_hot else 1.0,
                    alpha=0.95 if is_hot else 0.28, zorder=5 if is_hot else 4)
            if is_hot:
                ends.append((xs[-1], ys[-1], short(m), col))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, max(t for _, _, t, _ in cells) * 11.0)   # headroom for the end labels
        ax.set_ylim(0.30, max(b for *_, b in cells) * 1.6)
        label_ends(ax, ends)

        ax.annotate(f"never broke it — {100*nzero/len(cells):.0f}% of pairs",
                    (1.35, ZERO), textcoords="offset points", xytext=(0, -6),
                    fontsize=10, color=MUTED, va="top")
        ax.set_title(f"{label}   ·   {len(models)} models", fontsize=style.LABEL_SIZE,
                     color=INK, loc="left", pad=10)
        ax.set_xlabel("unique attack strings tried (submitted + chat)", color=INK2)
        if ax is axes[0]:
            ax.set_ylabel("cumulative breaks on that model", color=INK2)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2)

    handles = [plt.Line2D([], [], color=v, lw=2.6, label=k) for k, v in PAL.items()]
    axes[0].legend(handles=handles, fontsize=11, frameon=False, loc="upper left", ncol=2,
                   labelcolor=INK, handlelength=1.6, columnspacing=1.2, borderpad=0.2)

    style.title_and_dataset(
        fig,
        "Attack effort per model",
        "step line = frontier, the fewest unique attack strings any red-teamer needed for their "
        "k-th break on that model · 3 hardest + 3 easiest labelled",
        title_y=0.965, sub_y=0.905,
    )
    out = HERE / "effort_frontier.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
