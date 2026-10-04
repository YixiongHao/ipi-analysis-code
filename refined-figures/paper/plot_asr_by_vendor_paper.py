"""Paper (2x2) variant of strategy_frontier_panel/asr_by_vendor.png.

Reads the committed asr_by_vendor.json side-car (numbers computed by
strategy_frontier_panel/src/plot_by_vendor.py — run that first if stores
changed).

Produces two figures:
- asr_by_vendor_2x2_norm.png (MAIN TEXT): dual-axis. Colored strategy lines are
  each arm's ASR divided by the generic-control ASR at the same quarter
  (left axis, "relative to control"); the gray control line stays raw on the
  right axis. This factors out the overall model-hardening trend so the strategy
  lines show relative durability, while the raw base rate is still visible.
- asr_by_vendor_2x2.png (APPENDIX): the original raw-ASR single-axis version.

Run: python plot_asr_by_vendor_paper.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
SFP = BASE.parent.parent / "strategy_frontier_panel"

# Strategy colors are semantic: the two over-effective picks are warm shades
# (dark red / burnt orange), the two under-effective picks cold shades
# (dark / light blue); the control stays neutral gray.
ARMS = [
    ("baseline_control", "Generic control (no strategy selection)", "#444444", "o"),
    ("fake_user_assistant", "Fake User/Assistant Msgs", "#a81527", "s"),
    ("fake_cot", "Fake Chain-of-Thought", "#d4691f", "D"),
    ("fake_social_proof", "Fake Social Proof", "#1f5fa8", "^"),
    ("claim_urgency", "Claim Urgency / Emergency", "#3d95cc", "v"),
]
VENDOR_ORDER = ["Claude", "GPT", "Qwen", "Gemini"]
CTRL = "baseline_control"
CTRL_COLOR = "#444444"


def _load():
    cfg = json.loads((SFP / "data" / "models_by_vendor.json").read_text())
    pts = json.loads((SFP / "asr_by_vendor.json").read_text())["points"]
    by_key = {(p["vendor"], p["quarter"], p["arm"]): p for p in pts}
    return cfg["quarters"], pts, by_key


# ---------------------------------------------------------------- raw (appendix)
def fig_raw(quarters, pts, by_key):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharex=True, sharey=True)
    x = list(range(len(quarters)))

    for ax, vendor in zip(axes.flat, VENDOR_ORDER):
        for arm, label, colour, marker in ARMS:
            ys = [float("nan")] * len(quarters)
            lo = [0.0] * len(quarters)
            hi = [0.0] * len(quarters)
            for i, q in enumerate(quarters):
                p = by_key.get((vendor, q, arm))
                if not p:
                    continue
                ys[i] = p["asr"] * 100
                lo[i] = max(0.0, (p["asr"] - p["ci_lo"]) * 100)
                hi[i] = max(0.0, (p["ci_hi"] - p["asr"]) * 100)
            if any(v == v for v in ys):
                ax.errorbar(x, ys, yerr=[lo, hi], label=label, color=colour,
                            marker=marker, capsize=3, lw=1.8, ms=5.5,
                            elinewidth=1.1, alpha=.9)
        ax.set_title(vendor, fontsize=15, fontweight="bold", loc="left")
        ax.grid(alpha=.25, ls=":")

    top = max(p["ci_hi"] for p in pts) * 100 + 2
    axes.flat[0].set_ylim(0, top)
    for ax in axes[:, 0]:
        ax.set_ylabel("ASR (%)")
    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(quarters, rotation=45, ha="right", fontsize=10)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    # bottom legend: warm (over-effective) pair on the first row, cold
    # (under-effective) pair + control on the second
    fig.legend(handles[1:3], labels[1:3], ncol=2, fontsize=14, frameon=False,
               mode="expand", bbox_to_anchor=(0.14, 0.055, 0.72, 0.04), borderaxespad=0)
    fig.legend(handles[3:] + handles[:1], labels[3:] + labels[:1], ncol=3,
               fontsize=14, frameon=False, mode="expand",
               bbox_to_anchor=(0.05, 0.005, 0.9, 0.04), borderaxespad=0)
    fig.tight_layout(rect=[0, 0.105, 1, 1])
    out = BASE / "asr_by_vendor_2x2.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)


# ---------------------------------------------------------- normalized (main)
def fig_norm(quarters, pts, by_key):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharex=True, sharey=True)
    x = list(range(len(quarters)))
    n = len(quarters)
    LEFT_MAX, RIGHT_MAX = 3.0, 60.0
    strat = ARMS[1:]  # 4 strategy arms

    for idx, (ax, vendor) in enumerate(zip(axes.flat, VENDOR_ORDER)):
        axr = ax.twinx()
        # keep the colored (left-axis) strategy lines on top of the gray control
        ax.set_zorder(axr.get_zorder() + 1)
        ax.patch.set_visible(False)

        # RIGHT axis: raw control ASR (gray), with its CI + deviation flags
        cys = [float("nan")] * n
        clo = [0.0] * n
        chi = [0.0] * n
        for i, q in enumerate(quarters):
            p = by_key.get((vendor, q, CTRL))
            if not p:
                continue
            cys[i] = p["asr"] * 100
            clo[i] = max(0.0, (p["asr"] - p["ci_lo"]) * 100)
            chi[i] = max(0.0, (p["ci_hi"] - p["asr"]) * 100)
        axr.errorbar(x, cys, yerr=[clo, chi], color=CTRL_COLOR, marker="o",
                     ls="--", lw=2.2, ms=5, capsize=3, elinewidth=1.0, alpha=.85, zorder=2)
        axr.set_ylim(0, RIGHT_MAX)

        # LEFT axis: strategy ASR relative to control (ratio)
        for arm, label, colour, marker in strat:
            nys = [float("nan")] * n
            for i, q in enumerate(quarters):
                p = by_key.get((vendor, q, arm))
                c = by_key.get((vendor, q, CTRL))
                if p and c and c["asr"] > 0:
                    nys[i] = p["asr"] / c["asr"]
            if any(v == v for v in nys):
                ax.plot(x, nys, color=colour, marker=marker, lw=1.9, ms=5.5,
                        alpha=.92, zorder=3, label=label)
        ax.axhline(1.0, color="#9a9a9a", ls=":", lw=1.2, zorder=1)  # strategy == control
        ax.set_ylim(0, LEFT_MAX)
        ax.set_title(vendor, fontsize=15, fontweight="bold", loc="left")
        ax.grid(alpha=.22, ls=":")

        # tick-label hygiene: left ratio ticks on col 0 only (sharey handles it),
        # right raw-% ticks on col 1 only; color the right axis to match control.
        right_col = idx % 2 == 1
        axr.tick_params(axis="y", labelright=right_col, colors=CTRL_COLOR)
        if right_col:
            axr.set_ylabel("Control ASR (%, raw)", color=CTRL_COLOR)
        axr.spines["right"].set_color(CTRL_COLOR)

    for ax in axes[:, 0]:
        ax.set_ylabel("Strategy ASR relative to control ($\\times$)")
    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(quarters, rotation=45, ha="right", fontsize=10)

    # figure legend: 4 strategies + the raw-control line (right axis)
    handles = [Line2D([0], [0], color=c, marker=m, lw=1.9, ms=6, label=lab)
               for _, lab, c, m in strat]
    handles.append(Line2D([0], [0], color=CTRL_COLOR, marker="o", ls="--", lw=2.2,
                          ms=5, label="Generic control ASR (raw, right axis)"))
    labels = [h.get_label() for h in handles]
    # bottom legend: warm (over-effective) pair on the first row, cold
    # (under-effective) pair + control on the second
    fig.legend(handles[:2], labels[:2], ncol=2, fontsize=13, frameon=False,
               mode="expand", bbox_to_anchor=(0.14, 0.055, 0.72, 0.04), borderaxespad=0)
    fig.legend(handles[2:], labels[2:], ncol=3, fontsize=13, frameon=False,
               mode="expand", bbox_to_anchor=(0.05, 0.005, 0.9, 0.04), borderaxespad=0)
    fig.tight_layout(rect=[0, 0.105, 1, 1])
    out = BASE / "asr_by_vendor_2x2_norm.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)


def main():
    quarters, pts, by_key = _load()
    fig_norm(quarters, pts, by_key)
    fig_raw(quarters, pts, by_key)


if __name__ == "__main__":
    main()
