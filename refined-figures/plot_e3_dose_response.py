"""Refined: E3 dose-response — role-confusion probe score vs transfer attack success.

Source data: role-confusion/outputs/e3_dose_response.csv (isolated tool-wrap rendering)
             role-confusion/outputs/e3_dose_response_incontext.csv (real agent conversation)
Both written by role-confusion/analysis/dose_response.py from cached probe parquets
(gpt-oss-120b role probe, layer 20). No GPU needed.

Outputs:
  e3_dose_response.png          — headline, behavior-adjusted deciles, both renderings
  e3_dose_response_raw.png      — same but raw (global) deciles

Run: python plot_e3_dose_response.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "role-confusion" / "outputs"

METRICS = [
    ("p_user", "Userness"),
    ("toolness_suppression", "Tool-ness suppression"),
    ("confusion_margin", "Confusion margin"),
    ("p_cot", "CoT-ness"),
]
ARMS = [
    ("e3_dose_response.csv", "Attack text alone", "#C44E52"),
    ("e3_dose_response_incontext.csv", "Attack in its real agent conversation", "#4C72B0"),
]

DATASET_LINE = (
    "Role probe on gpt-oss-120b (layer 20) vs pooled arena transfer ASR  ·  "
    "n = 3,638 attacks alone / 3,977 in context  ·  deciles with 95% bootstrap CI"
)


def make(adjusted, out_name, title):
    fig, axes = plt.subplots(1, 4, figsize=(19, 6.2), sharey=True)
    for csv, label, color in ARMS:
        df = pd.read_csv(OUT / csv)
        df = df[df.adjusted == adjusted]
        for ax, (metric, pretty) in zip(axes, METRICS):
            g = df[df.metric == metric].sort_values("bin")
            # `x_mean` is always the raw metric mean in the CSV; the behavior-adjusted
            # curves are indexed by within-behavior decile, so use the decile midpoint.
            x = (g.bin + 0.5) / 10 if adjusted else g.x_mean
            ax.plot(x, g.asr, marker="o", markersize=6, lw=2.4,
                    color=color, label=label, zorder=3)
            ax.fill_between(x, g.lo, g.hi, color=color, alpha=0.16, zorder=2)

    for ax, (metric, pretty) in zip(axes, METRICS):
        ax.set_title(pretty, pad=10)
        ax.set_xlabel("Within-behavior decile" if adjusted else "Probe score")
        if adjusted:
            ax.set_xticks([0.05, 0.25, 0.5, 0.75, 0.95],
                          ["low", "", "mid", "", "high"])
    axes[0].set_ylabel("Attack success rate")
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[0].legend(loc="upper left", framealpha=0.95, fontsize=style.LEGEND_SIZE)

    fig.subplots_adjust(top=0.76, bottom=0.13, left=0.055, right=0.985, wspace=0.12)
    style.title_and_dataset(fig, title, DATASET_LINE, sub_y=0.885)
    path = HERE / out_name
    fig.savefig(path)
    print(f"wrote {path}")


def main():
    style.apply()
    make(True, "e3_dose_response.png",
         "Role-Confusion Probes vs. Attack Success (behavior-adjusted)")
    make(False, "e3_dose_response_raw.png",
         "Role-Confusion Probes vs. Attack Success (raw probe scale)")


if __name__ == "__main__":
    main()
