"""Paper figures for the strategy-survival analysis (planning/strategy_survival/).

Main text (goes right after the classifier ROC figure):
  strategy_residual_main.png            3 panels, restricted to the SAME detector roster as
                                        the main-text ROC figure.

Appendix (full rosters + the system-defense equivalent):
  strategy_residual_rho_full.png        detector-agreement matrix, all 10 detectors
  strategy_residual_categories_full.png both category profiles, all 17 strategies
                                        (SLIDES ONLY -- cut from the paper 2026-09-07 as
                                        redundant with the heatmap, which carries the same
                                        numbers per detector instead of per family)
  strategy_residual_system.png          the same question asked of the 6 system defenses
  strategy_residual_heatmap.png         per-detector delta, every strategy x every defense

Metric throughout: the behavior-stratified Mantel-Haenszel odds ratio that an attack carrying a
strategy evades the detector, strata = behavior_id (~90 behaviors). Positive log2 OR = the
strategy gets past; negative = it is caught. Behavior stratification matters because behavior is
the dominant covariate -- crude prevalence shifts collapse under it (see the report).

Source: planning/strategy_survival/{strategy_survival,system_profiles}.json (analyze.py,
system_profiles.py). Titles come from style.title_and_dataset so the paper build strips them.

Grouping note: the "LLM-based detectors" panel is the policy-aware family -- the LLM judges plus
Cygnal and Granite, which are themselves 8B LLM classifiers with a second policy head. The rho
matrix puts them in the same block for the same reason.
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

import style

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "planning" / "strategy_survival"
R = json.loads((SRC / "strategy_survival.json").read_text())
SYSP = json.loads((SRC / "system_profiles.json").read_text())

APPENDIX_ONLY = ("llmjudge-gemini-3-flash", "llmjudge-qwen3.5-9b",
                 "granite-guardian-4.1-8b-think")

LABEL = {
    "protectai-v2": "ProtectAI-v2", "promptguard2-86m": "PromptGuard2-86M",
    "datasentinel-mistral7b": "DataSentinel", "cygnal-code-8b": "Cygnal-8b",
    "granite-guardian-4.1-8b": "Granite-4.1-8b", "stackone-defender-tier2": "StackOne-T2",
    "llmjudge-gpt-5.6-luna": "GPT-5.6-Luna judge",
    "llmjudge-gemini-3-flash": "Gemini-3-Flash judge",
    "llmjudge-qwen3.5-2b": "Qwen3.5-2B judge", "llmjudge-qwen3.5-9b": "Qwen3.5-9B judge",
}
SYS_LABEL = {"causalarmor": "CausalArmor", "melon": "MELON", "firewalls": "Firewalls",
             "camel": "CaMeL", "fides": "FIDES", "ipiguard": "IPIGuard"}
SYS_ORDER = ("causalarmor", "firewalls", "melon", "camel", "fides", "ipiguard")

JUDGE = ["llmjudge-gemini-3-flash", "llmjudge-gpt-5.6-luna",
         "llmjudge-qwen3.5-9b", "llmjudge-qwen3.5-2b"]
ENCODER = ["stackone-defender-tier2", "protectai-v2", "promptguard2-86m"]
LLMCLF = ["cygnal-code-8b", "granite-guardian-4.1-8b"]      # 8B LLM classifiers
ODD = "datasentinel-mistral7b"
JUDGE_MAIN = [d for d in JUDGE if d not in APPENDIX_ONLY]
# the policy-aware family: LLM judges + the two LLM classifiers with a policy head
POLICY_MAIN = JUDGE_MAIN + LLMCLF
POLICY_FULL = JUDGE + LLMCLF

UP, DOWN = "#b2182b", "#2166ac"

# fig_main layout knobs. The width ratio is a gridspec AXES ratio, not the ratio you see
# in the saved PNG: the left panel also carries long y tick labels and a colorbar, and
# savefig crops bbox-tight, so the on-image share differs from the axes share. 0.80 is
# what puts the left block at half the cropped image width (fig_main prints the measured
# share on every build) -- re-measure after any font or label change.
MAIN_WIDTH_RATIOS = [0.90, 1.0]
MAIN_WSPACE = 0.26
MAIN_HSPACE = 0.60
MAIN_WRAP = 22

# The left panel taking half the width leaves the two stacked bar panels narrow, so the
# figure is taller: at this height the rho matrix comes out square rather than stretched,
# and each bar panel gets ~28% more room per row. Margins below are held in INCHES so the
# extra height goes to the panels and not to the gaps around the title and legend.
MAIN_FIGSIZE = (14.0, 6.6)
MAIN_MARGIN_IN = dict(left=1.61, right=0.35, top=1.06, bottom=1.00)

# Type sizes shared by the three panel headings and the legend. They are set together
# because the figure is scaled hard to 	extwidth (14in canvas -> 5.5in page, ~0.39x), so
# what looks generous in the PNG is small on the page. Raised 2026-09-07 (user).
TITLE_PT = 15.0
LEGEND_PT = 13.0


def cs(d):
    return R["classifier"][d]["correct_site"]


BASE = cs("cygnal-code-8b")["strategies"]
TAGS = [t for t, r in BASE.items() if r["pct_all"] >= 1.0]      # 17 tags, >=1% of the set


def profile(G):
    """per strategy: mean log2 MH OR over members with a finite OR, + consistency counts."""
    out = {}
    for t in TAGS:
        ors = [cs(d)["strategies"][t]["mh_or"] for d in G]
        qs = [cs(d)["strategies"][t]["mh_q"] for d in G]
        fin = [o for o in ors if o]
        zero = sum(1 for o in ors if not o)          # OR == 0 => caught every tagged attack
        pts = [np.log2(o) for o in fin]
        m = float(np.mean(pts)) if pts else -6.0
        out[t] = dict(
            mean=m, pts=pts, n=len(G), zero=zero,
            # zero-OR members are an extreme "caught", so they agree with a negative mean
            agree=max(sum(1 for o in fin if o > 1), sum(1 for o in fin if o < 1) + zero),
            sig=min(sum(1 for o, q in zip(ors, qs) if q is not None and q < 0.05), len(G)),
        )
    return out


def wrap(t, n=30):
    out, line = [], ""
    for w in t.split():
        if len(line) + len(w) + 1 > n:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    out.append(line)
    return "\n".join(out)


def rho_matrix(ax, dets, fig, fs=9, labels=None):
    """Spearman agreement between detectors' behavior-adjusted strategy profiles."""
    labels = labels or LABEL
    fin = [t for t in TAGS if all(cs(d)["strategies"][t]["mh_or"] for d in dets)]
    M = np.array([[np.log2(cs(d)["strategies"][t]["mh_or"]) for t in fin] for d in dets])
    rho = np.array([[spearmanr(M[i], M[j]).correlation for j in range(len(dets))]
                    for i in range(len(dets))])
    im = ax.imshow(rho, cmap="RdYlBu_r", vmin=-1, vmax=1)
    x = np.arange(len(dets))
    ax.set_xticks(x)
    ax.set_yticks(x)
    ax.set_xticklabels([labels[d] for d in dets], rotation=45, ha="right", fontsize=fs)
    ax.set_yticklabels([labels[d] for d in dets], fontsize=fs)
    for i in range(len(dets)):
        for j in range(len(dets)):
            ax.text(j, i, f"{rho[i, j]:.2f}".lstrip("0").replace("-0.", "-."),
                    ha="center", va="center", fontsize=fs - 1.6,
                    color="white" if abs(rho[i, j]) > 0.62 else "black")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(r"Spearman $\rho$", fontsize=fs)
    cb.ax.tick_params(labelsize=fs)
    return rho, fin


def or_ticks(ax, xlim, fs=10):
    """Label a log2-OR axis in ODDS-RATIO units, so a tick reads directly.

    The bars stay on the log2 scale (that is what makes 'twice as likely' and
    'half as likely' the same distance from the null), but a reader should not
    have to exponentiate in their head: 1 is the null, 2 means the strategy has
    twice the odds of getting past, 1/2 means half.
    """
    lo, hi = math.ceil(xlim[0]), math.floor(xlim[1])
    for step in (1, 2, 3, 4, 5, 6, 8):          # ticks grow outward from the null
        ticks = ([k for k in range(0, lo - 1, -step) if k >= lo][::-1]
                 + [k for k in range(step, hi + 1, step)])
        if len(ticks) <= 6:
            break
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1" if k == 0 else
                        (str(2 ** k) if k > 0 else f"1/{2 ** -k}") for k in ticks],
                       fontsize=fs)


def category_panel(ax, G, rows, fs=10, xlim=(-4.2, 2.9), annot=True):
    P = profile(G)
    y = np.arange(len(rows))
    for i, t in enumerate(rows):
        d = P[t]
        solid = d["agree"] == d["n"] and d["sig"] >= d["n"] / 2
        col = UP if d["mean"] > 0 else DOWN
        ax.barh(i, d["mean"], height=0.66, color=col, alpha=1.0 if solid else 0.28,
                hatch=None if solid else "///", edgecolor=col, linewidth=0.9, zorder=2)
        ax.scatter(d["pts"], [i] * len(d["pts"]), s=20, color="white",
                   edgecolor="#333", linewidth=0.9, zorder=4)
        if annot:
            ax.text(xlim[0] + 0.08, i, f"{d['agree']}/{d['n']}", va="center", ha="left",
                    fontsize=fs - 2.2, color="#333" if solid else "#aaa")
    ax.axvline(0, color="black", lw=1.3)
    ax.set_yticks(y)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xlim(*xlim)
    or_ticks(ax, xlim, fs=fs - 1.5)
    ax.grid(axis="x", color="#e8e8e8", zorder=0)
    ax.set_axisbelow(True)
    return P


def inline_labels(ax, P, rows, fs, wrap_n=22, pad=0.18):
    """Put each strategy name INSIDE the panel, on the side of the null its bar does not use.

    A bar that points right leaves its whole left half empty, and vice versa, so the name
    goes there. That frees the outer gutter entirely, which is what lets the bars fill the
    panel.

    The bar is not the only thing on the row, though: the individual-detector dots can sit
    on the other side of the null from the mean, and then the name lands on top of them. So
    the name starts past whatever the row actually reaches on its side, not at a fixed pad.
    Returns the Text objects so the caller can widen the axis until they all fit.
    """
    out = []
    for i, t in enumerate(rows):
        d = P[t]
        right_pointing = d["mean"] > 0
        solid = d["agree"] == d["n"] and d["sig"] >= d["n"] / 2
        reach = [0.0] + [q for q in d["pts"] if (q < 0) == right_pointing]
        x = min(reach) - pad if right_pointing else max(reach) + pad
        out.append(ax.text(x, i, wrap(t, wrap_n),
                           ha="right" if right_pointing else "left", va="center",
                           multialignment="right" if right_pointing else "left",
                           fontsize=fs, color="#222" if solid else "#888", zorder=5))
    return out


LABEL_PT = []       # sizes place_labels settled on, for the layout printout
FRAC_ROOM = 0.55        # data units reserved past the last dot for the agree-fraction


def bar_fractions(ax, P, rows, fs):
    """The agree-fraction, printed past the OUTERMOST dot of its row.

    It used to hang off the end of the strategy name, which made the longest line ~30% wider
    and forced the whole label down to 7.75pt once names were capped at two lines. Inside the
    bar it collides with the null rule on short bars, and just past the bar tip it lands on
    the individual-detector dots -- past the last dot is the only spot that is free on every
    row. FRAC_ROOM reserves the space in advance, so the axis never has to grow to fit it.
    """
    for i, t in enumerate(rows):
        d = P[t]
        sign = 1.0 if d["mean"] > 0 else -1.0
        end = max([d["mean"]] + d["pts"]) if sign > 0 else min([d["mean"]] + d["pts"])
        ax.text(end + sign * 0.16, i, f"{d['agree']}/{d['n']}",
                ha="left" if sign > 0 else "right", va="center",
                fontsize=fs, color="#555", zorder=6)


def fit_xlim(ax, texts, data_lo, data_hi, pad=0.10, slack=0.30):
    """Widen the axis until every inline label fits beside the null.

    Work in FRACTIONS of the panel, not in data units. A label's width is fixed in points,
    so widening the axis makes it cover MORE data units, not fewer -- iterating on data
    units diverges. As a fraction of the panel the width is constant, so each label needs
    lo <= x_i - f_i*(hi-lo) (or hi >= x_i + f_i*(hi-lo)), where x_i is where that label
    starts -- which is NOT the same on every row, since inline_labels pushes a name past
    the dots on its side. The fixed point converges whenever the widest label on the left
    and the widest on the right together need less than the whole panel.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w_axes = ax.get_window_extent(r).width
    left, right = [], []                          # (start x in data units, width fraction)
    f_left = f_right = 0.0
    for tx in texts:
        frac = tx.get_window_extent(r).width / w_axes
        x = tx.get_position()[0]
        if tx.get_horizontalalignment() == "right":
            left.append((x, frac))
            f_left = max(f_left, frac)
        else:
            right.append((x, frac))
            f_right = max(f_right, frac)
    # How much of the panel the two label columns may claim between them. Every point of
    # this is paid for by the bars: the axis widens to make room, so the bars cover less of
    # it. 0.90 is where the names read at a useful size without the bars looking stubby --
    # it is the knob to turn if the strategy names need to grow again, and it trades directly
    # against MAIN_WIDTH_RATIOS, since both decide how wide the bar panels end up.
    if f_left + f_right >= 0.90:                 # no room; caller must shrink the font
        raise ValueError(f"inline labels need {f_left + f_right:.2f} of the panel width")
    lo, hi = data_lo - slack, data_hi + slack
    for _ in range(40):
        d = hi - lo
        new_lo = min([data_lo - slack] + [x - pad - f * d for x, f in left])
        new_hi = max([data_hi + slack] + [x + pad + f * d for x, f in right])
        if abs(new_lo - lo) < 1e-3 and abs(new_hi - hi) < 1e-3:
            break
        lo, hi = new_lo, new_hi
    ax.set_xlim(lo, hi)
    return lo, hi


def wrap_for_lines(rows, max_lines=2, lo=12, hi=40):
    """Narrowest wrap width that still keeps every name within max_lines lines.

    Wrapping harder saves horizontal space but costs a line, and a label deeper than two
    lines crowds its row. So the line count is the hard constraint and the type size is the
    variable that gives (see place_labels).
    """
    for n in range(lo, hi + 1):
        if all(wrap(t, n).count("\n") + 1 <= max_lines for t in rows):
            return n
    return hi


def place_labels(ax, P, rows, data_lo, data_hi, fs=10.5, max_lines=2):
    """Draw the inline labels and fit the axis to them, shrinking the text if it will not fit.

    The bars must fill the panel, so the labels have to live inside it. Wrap width is pinned
    by max_lines; when the panel is narrow the type size is the only thing left to give.
    """
    n = wrap_for_lines(rows, max_lines)
    for size in [fs - 0.25 * k for k in range(15)]:
        texts = inline_labels(ax, P, rows, fs=size, wrap_n=n)
        try:
            lim = fit_xlim(ax, texts, data_lo, data_hi)
            LABEL_PT.append(size)      # what the size search actually settled on
            return lim
        except ValueError:
            for t in texts:
                t.remove()
    raise ValueError(f"inline labels need >{max_lines} lines to fit; widen the bar panels")


def data_span(P, rows):
    """Extent the panel must show: every bar tip and every individual-detector dot."""
    vals = []
    for t in rows:
        vals.append(P[t]["mean"])
        vals.extend(P[t]["pts"])
    return min(vals + [0.0]), max(vals + [0.0])


def _left_share(fig, ax_left, ax_cbar):
    """Rendered width share of the left block, measured on what savefig actually crops.

    MAIN_WIDTH_RATIOS is an AXES ratio; the left block also carries long y tick labels and
    a colorbar, and savefig crops bbox-tight, so the two numbers are far apart. This is the
    one that matters, and it is what the "left panel = half the figure" request is about.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox
    whole = fig.get_tightbbox(r).transformed(fig.dpi_scale_trans)   # inches -> pixels
    left = Bbox.union([ax_left.get_tightbbox(r), ax_cbar.get_tightbbox(r)])
    return (left.x1 - whole.x0) / whole.width


def _legend_handles(hatch=True):
    """hatch=False drops the split/weak-evidence swatch (main figure: only one row is
    hatched there and the caption already states the solid-bar rule)."""
    return [
        Patch(facecolor=UP, label="evades this category"),
        Patch(facecolor=DOWN, label="caught by this category"),
    ] + ([Patch(facecolor="white", edgecolor="#888", hatch="///",
                label="category members split, or weak evidence")] if hatch else []) + [
        Line2D([], [], marker="o", ls="", mfc="white", mec="#333",
               label="individual detector"),
    ]


# ------------------------------------------------------------------ main text
def fig_main(out):
    style.apply()
    Pp, Pe = profile(POLICY_MAIN), profile(ENCODER)

    # Each panel picks its OWN three most- and three least-evading strategies and
    # carries its own y labels, so the two orderings can be read against each other.
    cand = [t for t in TAGS if BASE[t]["pct_all"] >= 3.0]
    rows_p = sorted(sorted(cand, key=lambda t: Pp[t]["mean"])[:3]
                    + sorted(cand, key=lambda t: Pp[t]["mean"])[-3:],
                    key=lambda t: Pp[t]["mean"])
    rows_e = sorted(sorted(cand, key=lambda t: Pe[t]["mean"])[:3]
                    + sorted(cand, key=lambda t: Pe[t]["mean"])[-3:],
                    key=lambda t: Pe[t]["mean"])

    # Two columns: the rho matrix owns the left half, the two bar panels STACK on the right.
    # Side by side each bar panel got only a quarter of the width, which drove the inline
    # labels down to ~9pt on a figure already scaled hard to \textwidth. Stacked they get
    # half the width each, so the type can be bigger.
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    w, h = MAIN_FIGSIZE
    mg = MAIN_MARGIN_IN
    fig.subplots_adjust(left=mg["left"] / w, right=1 - mg["right"] / w,
                        top=1 - mg["top"] / h, bottom=mg["bottom"] / h)
    gs = fig.add_gridspec(2, 2, width_ratios=MAIN_WIDTH_RATIOS,
                          wspace=MAIN_WSPACE, hspace=MAIN_HSPACE)
    a0 = fig.add_subplot(gs[:, 0])
    a1, a2 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    m8 = list(POLICY_MAIN) + ENCODER + [ODD]
    # the caption says which of these are judges, so the word only costs axis width here
    before = set(fig.axes)
    rho_matrix(a0, m8, fig, fs=10.5,
               labels={d: LABEL[d].replace(" judge", "") for d in LABEL})
    cbax = [a for a in fig.axes if a not in before][0]
    for b in (3.5, 6.5):
        a0.axhline(b, color="black", lw=1.8)
        a0.axvline(b, color="black", lw=1.8)
    a0.set_aspect("auto")     # fill the half it was given; square cells would letterbox it
    a0.set_title("Rank correlation across strategies", fontsize=TITLE_PT, pad=10)

    for ax, G, rows, name in ((a1, POLICY_MAIN, rows_p, "LLM-based detectors"),
                              (a2, ENCODER, rows_e, "Encoder classifiers")):
        # each panel gets its OWN range, fitted to its own bars -- no shared padding
        lo, hi = data_span(profile(G), rows)
        P = category_panel(ax, G, rows, fs=12, xlim=(lo - 0.4, hi + 0.4), annot=False)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        lim = place_labels(ax, P, rows, lo, hi, fs=13.0, max_lines=1)
        # put the null in the middle: symmetric limits make "twice the odds" and "half the
        # odds" the same distance from centre, which is the comparison the panel is for
        m = max(abs(lim[0]), abs(lim[1]))
        ax.set_xlim(-m, m)
        or_ticks(ax, (-m, m), fs=11)
        ax.set_title(f"{name}  ({len(G)})", fontsize=TITLE_PT, pad=8)
    # the two stacked panels carry different ranges, so each keeps its own ticks; only the
    # bottom one is labelled, because the quantity is the same
    a2.set_xlabel("odds ratio of evading", fontsize=12)

    # the legend describes the bar panels only, so centre it on those two, not on the
    # whole figure -- taken from the live axes positions so it tracks the width ratios
    bars_mid = (a1.get_position().x0 + a1.get_position().x1) / 2
    fig.legend(handles=_legend_handles(hatch=False), fontsize=LEGEND_PT, loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(bars_mid, 0.03 / h))
    style.title_and_dataset(
        fig, "Detectors do not share one blind spot",
        "26,466 arena IPI trajectories (14,021 with consensus strategy labels); "
        "same detector roster as the ROC figure",
        title_y=1 - 0.08 / h, sub_y=1 - 0.37 / h)
    share = _left_share(fig, a0, cbax)
    fig.savefig(HERE / out)
    print(f"wrote {HERE / out}   rows={len(rows_p)}/{len(rows_e)}   "
          f"left panel = {share:.1%} of the cropped image width   "
          f"strategy labels {LABEL_PT[-2:]}pt")


# ------------------------------------------------------------------- appendix
def fig_rho_full(out):
    style.apply()
    fig = plt.figure(figsize=(9.2, 7.6))
    ax = fig.add_subplot(111)
    dets = list(POLICY_FULL) + ENCODER + [ODD]
    _, fin = rho_matrix(ax, dets, fig, fs=10)
    for b in (5.5, 8.5):
        ax.axhline(b, color="black", lw=1.8)
        ax.axvline(b, color="black", lw=1.8)
    ax.set_title(f"agreement of the behavior-adjusted profile over {len(fin)} strategies",
                 fontsize=12, pad=10)
    style.title_and_dataset(fig, "Two detector families, not one", "all 10 detectors")
    fig.subplots_adjust(left=0.26, right=0.99, top=0.85, bottom=0.24)
    fig.savefig(HERE / out)
    print(f"wrote {HERE / out}")


def fig_categories_full(out):
    style.apply()
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 9.0), gridspec_kw={"wspace": 0.42})
    for k, (G, name) in enumerate(((POLICY_FULL, "LLM-based detectors"),
                                    (ENCODER, "Encoder classifiers"))):
        P = profile(G)
        rows = sorted(TAGS, key=lambda t: P[t]["mean"])
        category_panel(ax[k], G, rows, fs=11, xlim=(-5.6, 3.0))
        ax[k].set_yticklabels([f"{wrap(t, 34)}  ({BASE[t]['pct_all']:.0f}%)" for t in rows],
                              fontsize=9)
        ax[k].set_title(f"{name}  ({len(G)})", fontsize=13, pad=10)
        ax[k].set_xlabel("behavior-adjusted odds ratio of evading  (log scale)",
                         fontsize=12)
    fig.legend(handles=_legend_handles(), fontsize=11, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.008))
    style.title_and_dataset(
        fig, "Which strategies survive which kind of detector",
        "all 17 strategies at >=1% prevalence; every member of each category")
    fig.subplots_adjust(left=0.215, right=0.995, top=0.87, bottom=0.13)
    fig.savefig(HERE / out)
    print(f"wrote {HERE / out}")


def fig_system(out):
    """Figure 19's question, asked of the system defenses: per-defense residual, every strategy.

    Same estimator and same colour scale as the detector heatmap, so the two can be read
    against each other. It is a much emptier grid, and that is the finding: the evader sets
    run 56 down to 3 attacks, so many cells have no estimate at all and NOT ONE cell survives
    BH correction. Read it as "there is no profile here", not as a set of effects.
    """
    style.apply()
    sy = lambda d: R["system"][d]["reliable"]["strategies"]          # noqa: E731
    nev = {d: R["system"][d]["reliable"]["n_got_past"] for d in SYS_ORDER}
    base = sy("causalarmor")
    order = [t for t, r in base.items() if r["pct_all"] >= 3.0]
    order.sort(key=lambda t: -float(np.mean(
        [np.log2(sy(d)[t]["mh_or"]) for d in SYS_ORDER if sy(d)[t]["mh_or"]] or [0.0])))

    M = np.array([[np.log2(sy(d)[t]["mh_or"]) if sy(d)[t]["mh_or"] else np.nan
                   for d in SYS_ORDER] for t in order])
    RAW = [[sy(d)[t]["mh_or"] for d in SYS_ORDER] for t in order]
    Q = [[sy(d)[t].get("mh_q") for d in SYS_ORDER] for t in order]

    fig = plt.figure(figsize=(10.5, 6.4))
    ax = fig.add_subplot(111)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(len(SYS_ORDER)))
    ax.set_xticklabels([SYS_LABEL[d] + "\n(n=" + str(nev[d]) + ")" for d in SYS_ORDER],
                       fontsize=10)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{wrap(t, 34)}  ({base[t]['pct_all']:.0f}%)" for t in order],
                       fontsize=9.5)
    for i in range(len(order)):
        for j in range(len(SYS_ORDER)):
            v, q = RAW[i][j], Q[i][j]
            if v is None:                       # too few evaders in this stratum to estimate
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#999")
                continue
            if v == 0:
                ax.text(j, i, "0", ha="center", va="center", fontsize=8, color="#555")
                continue
            star = "*" if q is not None and q < 0.05 else ""
            ax.text(j, i, f"{v:.1f}{star}" if v >= 1 else f"{v:.2f}{star}",
                    ha="center", va="center", fontsize=8.4,
                    color="white" if abs(M[i, j]) > 1.9 else "black")
    ax.grid(False)
    ax.axvline(2.5, color="black", lw=1.6)      # content detection | plan-then-execute
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=range(-3, 4))
    cb.ax.set_yticklabels(["1" if k == 0 else
                           (str(2 ** k) if k > 0 else f"1/{2 ** -k}") for k in range(-3, 4)])
    cb.set_label("behavior-adjusted odds ratio of evading")
    style.title_and_dataset(
        fig, "Per-defense residual, every strategy",
        "381-union x 4 passes, 340 attacks with consensus labels; "
        "an em dash = too few evaders to estimate; no cell reaches BH q < 0.05")
    fig.subplots_adjust(left=0.38, right=0.99, top=0.86, bottom=0.14)
    fig.savefig(HERE / out)
    print(f"wrote {HERE / out}")


def fig_heatmap(out):
    """Per-detector behavior-adjusted OR, every strategy x every detector."""
    style.apply()
    dets = list(POLICY_FULL) + ENCODER + [ODD]
    order = sorted(TAGS, key=lambda t: -float(np.mean(
        [np.log2(cs(d)["strategies"][t]["mh_or"]) for d in dets
         if cs(d)["strategies"][t]["mh_or"]] or [0.0])))
    M = np.array([[np.log2(cs(d)["strategies"][t]["mh_or"])
                   if cs(d)["strategies"][t]["mh_or"] else np.nan for d in dets]
                  for t in order])
    Q = [[cs(d)["strategies"][t]["mh_q"] for d in dets] for t in order]

    fig = plt.figure(figsize=(11.5, 8.6))
    ax = fig.add_subplot(111)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(len(dets)))
    ax.set_xticklabels([LABEL[d] for d in dets], rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{t}  ({BASE[t]['pct_all']:.0f}%)" for t in order], fontsize=9.5)
    for i in range(len(order)):
        for j in range(len(dets)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "0", ha="center", va="center", fontsize=8, color="#555")
                continue
            q = Q[i][j]
            star = "*" if q is not None and q < 0.05 else ""
            # colour still maps log2 (so 2x and 1/2x are equally far from the null),
            # but the printed number is the odds ratio itself
            odds = 2.0 ** M[i, j]
            ax.text(j, i, f"{odds:.1f}{star}" if odds >= 1 else f"{odds:.2f}{star}",
                    ha="center", va="center", fontsize=7.6,
                    color="white" if abs(M[i, j]) > 1.9 else "black")
    ax.grid(False)
    for b in (3.5, 5.5, 8.5):
        ax.axvline(b, color="black", lw=1.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=range(-3, 4))
    cb.ax.set_yticklabels(["1" if k == 0 else
                           (str(2 ** k) if k > 0 else f"1/{2 ** -k}") for k in range(-3, 4)])
    cb.set_label("behavior-adjusted odds ratio of evading")
    style.title_and_dataset(
        fig, "Per-detector residual, every strategy",
        "* = BH q < 0.05; '0' = the detector caught every attack carrying that strategy")
    fig.subplots_adjust(left=0.42, right=0.99, top=0.86, bottom=0.17)
    fig.savefig(HERE / out)
    print(f"wrote {HERE / out}")


def main():
    fig_main("strategy_residual_main.png")
    fig_rho_full("strategy_residual_rho_full.png")
    fig_categories_full("strategy_residual_categories_full.png")
    fig_system("strategy_residual_system.png")
    fig_heatmap("strategy_residual_heatmap.png")


if __name__ == "__main__":
    main()
