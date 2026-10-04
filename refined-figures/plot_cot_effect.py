"""Refined: CoT-exposure effect on classifier monitors.

Source data: classifier-defenses/llm_judge/cot_effect_summary.json
(produced from the paired with/without-CoT
sidecars over the 9,271 CoT-bearing store records).

Run: python plot_cot_effect.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "classifier-defenses" / "llm_judge" / "cot_effect_summary.json"

MONITORS = [
    ("judge Qwen3.5-9B (temp 0)", "LLM judge\n(Qwen3.5-9B)"),
    ("Cygnal cygnal-code-8b", "Cygnal\n(code-8b)"),
    ("Granite-Guardian-4.1-8b", "Granite Guardian\n(4.1-8b)"),
]


def main():
    style.apply()
    data = json.loads(SRC.read_text())

    suppress, raise_, nets = [], [], []
    for key, _ in MONITORS:
        d = data[key]
        n, basepos = d["n"], d["basepos"]
        p2n, n2p = d["pos->neg"], d["neg->pos"]
        s = p2n / basepos
        r = n2p / (n - basepos)
        suppress.append(s)
        raise_.append(r)
        raw = n2p - p2n
        at_judge = n * (0.062 * r - 0.938 * s)
        at_cygnal = n * (0.351 * r - 0.649 * s)
        nets.append((raw, at_judge, at_cygnal))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.8))
    x = np.arange(len(MONITORS))
    labels = [lab for _, lab in MONITORS]

    # Panel A — flip rates by verdict class
    w = 0.34
    b1 = ax1.bar(x - w / 2, suppress, w, color="#C44E52",
                 label="Suppressed an existing detection")
    b2 = ax1.bar(x + w / 2, raise_, w, color="#4C72B0",
                 label="Raised a new detection")
    for bars in (b1, b2):
        for rect in bars:
            ax1.annotate(f"{rect.get_height():.1%}",
                         (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", fontsize=style.TICK_SIZE)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Share of verdicts flipped by adding CoT")
    ax1.set_ylim(0, 0.50)
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax1.set_title("Flip rate within each verdict class", pad=12)
    ax1.legend(loc="upper left", framealpha=0.9)

    # Panel B — net change in flagged records, raw and at matched base rates
    w = 0.26
    scen = ["Raw (own base rate)", "At judge's base rate (93.8%)",
            "At Cygnal's base rate (64.9%)"]
    colors = ["#555555", "#937860", "#8172B3"]
    for j, (name, c) in enumerate(zip(scen, colors)):
        vals = [nets[i][j] for i in range(len(MONITORS))]
        bars = ax2.bar(x + (j - 1) * w, vals, w, color=c, label=name)
        for rect, v in zip(bars, vals):
            va = "bottom" if v >= 0 else "top"
            off = 4 if v >= 0 else -4
            ax2.annotate(f"{v:+,.0f}",
                         (rect.get_x() + rect.get_width() / 2, v),
                         xytext=(0, off), textcoords="offset points",
                         ha="center", va=va, fontsize=style.TICK_SIZE - 1)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("Net change in records flagged")
    ax2.set_ylim(-3950, 1400)
    ax2.set_title("Net effect on total detections", pad=12)
    ax2.legend(loc="lower left", framealpha=0.9)

    fig.subplots_adjust(top=0.80, bottom=0.12, wspace=0.24)
    style.title_and_dataset(
        fig,
        "Effect of Showing the Agent's CoT to IPI Monitors",
        "9,271 CoT-bearing attack records (Gray Swan IPI store) — "
        "each monitor scored with vs. without the agent's chain of thought",
    )
    out = HERE / "cot_effect_monitors.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
