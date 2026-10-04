"""Refined: system-level defense figures (3 outputs).

1. sysdef_asr.png          — ASR per defense on the 381-attack IPI union set
2. sysdef_utility.png      — benign task success per defense (AgentDojo, no attacks)
3. sysdef_class_heatmap.png — attack success by attack class x defense

Source data:
- system-defenses/results/evasion_4pass.json (4-pass eval, deepseek-v4-pro)
- system-defenses/fp_agentdojo/results_fp_n4.json (n=4 benign utility)
Class map copied from system-defenses/hard_sample/eval_union381/plot_evasion.py.

Run: python plot_system_defenses.py
"""

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

import style

HERE = Path(__file__).resolve().parent
SD = HERE.parent / "system-defenses"

NAME = {"baseline": "No defense", "causalarmor": "CausalArmor", "melon": "MELON",
        "firewalls": "Firewalls", "camel": "CaMeL", "fides": "FIDES", "ipiguard": "IPIGuard"}
FORKS = {"camel", "fides", "ipiguard"}
C_BASE, C_ADAPT, C_FORK = "#8C8C8C", "#DD8452", "#4C72B0"


def color(arm):
    if arm == "baseline":
        return C_BASE
    return C_FORK if arm in FORKS else C_ADAPT


def kind_legend(ax, loc="upper right"):
    handles = [Patch(color=C_BASE, label="No defense (baseline)"),
               Patch(color=C_ADAPT, label="Other patterns"),
               Patch(color=C_FORK, label="Plan/Code-Then-Execute, Dual-LLM")]
    ax.legend(handles=handles, loc=loc, framealpha=0.9)


def fig_asr():
    pa = json.loads((SD / "results" / "evasion_4pass.json").read_text())["per_attack"]
    arms = ["baseline", "causalarmor", "melon", "firewalls", "camel", "fides", "ipiguard"]

    # ASR over ALL rollouts (not delivery-gated): denominator is n (every attempt), not
    # n_deliv. Delivery rates are 95.7-99.1% (see fig_delivery / appendix), so the two
    # estimators agree to <=0.7pp; the un-gated rate treats a non-delivered attack as a
    # non-break, consistent with the project convention that a refusal-to-deliver = attack failed.
    asr = {}
    for a in arms:
        brk = sum(pa[h][a]["break_count"] for h in pa)
        tot = sum(pa[h][a]["n"] for h in pa)
        asr[a] = brk / tot * 100

    # attack-clustered bootstrap CI, identical to eval_union381/plot_asr_dist.py
    def cluster_ci(a, B=3000):
        hs = list(pa); n = len(hs); est = []
        rng = random.Random(0)
        for _ in range(B):
            samp = [hs[rng.randrange(n)] for _ in range(n)]
            brk = sum(pa[h][a]["break_count"] for h in samp)
            tot = sum(pa[h][a]["n"] for h in samp)
            est.append(brk / tot * 100 if tot else 0.0)
        est.sort()
        return est[int(0.025 * B)], est[int(0.975 * B)]

    ci = {a: cluster_ci(a) for a in arms}
    order = ["baseline"] + sorted([a for a in arms if a != "baseline"], key=lambda d: -asr[d])

    fig, ax = plt.subplots(figsize=(12, 6.8))
    vals = [asr[a] for a in order]
    yerr = np.array([[vals[i] - ci[a][0], ci[a][1] - vals[i]] for i, a in enumerate(order)]).T
    ax.bar(range(len(order)), vals, color=[color(a) for a in order],
           yerr=yerr, capsize=5, error_kw={"ecolor": "#222", "elinewidth": 1.3})
    for i, a in enumerate(order):
        ax.text(i, ci[a][1] + 1.2, f"{vals[i]:.1f}%", ha="center", va="bottom",
                fontsize=style.LABEL_SIZE, fontweight="bold")
    ax.set_xticks(range(len(order)), [NAME[a] for a in order])
    ax.set_ylabel("Attack success rate (% of all rollouts)")
    ax.set_ylim(0, 74)
    kind_legend(ax)
    fig.subplots_adjust(top=0.90, bottom=0.09)
    style.title_and_dataset(fig, "System Defenses — Attack Success Rate", None, title_y=0.96)
    out = HERE / "sysdef_asr.png"
    fig.savefig(out)
    print(f"wrote {out}")


def fig_utility():
    d = json.loads((SD / "fp_agentdojo" / "results_fp_n4.json").read_text())
    by = d["bootstrap_ci95"]["by_arm"]
    order = ["baseline"] + sorted([a for a in by if a != "baseline"],
                                  key=lambda a: -by[a]["utility_pct"])

    fig, ax = plt.subplots(figsize=(12, 6.8))
    base = by["baseline"]["utility_pct"]
    ax.axhline(base, color="#666666", ls=(0, (5, 4)), lw=1.4, zorder=1)
    ax.text(len(order) - 0.4, base + 1.2, f"no-defense {base:.1f}%", ha="right",
            fontsize=style.ANNOT_SIZE, color="#555555")
    for i, a in enumerate(order):
        v = by[a]["utility_pct"]
        lo, hi = by[a]["ci95_bootstrap"]
        ax.bar(i, v, color=color(a),
               yerr=[[v - lo], [hi - v]], capsize=5,
               error_kw={"ecolor": "#222", "elinewidth": 1.3})
        ax.text(i, hi + 1.5, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=style.LABEL_SIZE, fontweight="bold")
        ax.scatter([i + 0.28] * 4, by[a]["per_pass_pct"], s=28, color="#C8A227",
                   zorder=5, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(order)), [NAME[a] for a in order])
    ax.set_ylabel("Benign task success (%)")
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    handles = [Patch(color=C_BASE, label="No defense (baseline)"),
               Patch(color=C_ADAPT, label="Other patterns"),
               Patch(color=C_FORK, label="Plan/Code-Then-Execute, Dual-LLM"),
               plt.Line2D([0], [0], lw=0, marker="o", color="#C8A227",
                          markeredgecolor="white", label="Per-pass rate (4 passes)")]
    ax.legend(handles=handles, loc="lower left", framealpha=1.0)
    fig.subplots_adjust(top=0.90, bottom=0.09)
    style.title_and_dataset(fig, "System Defenses — Benign Task Success", None, title_y=0.96)
    out = HERE / "sysdef_utility.png"
    fig.savefig(out)
    print(f"wrote {out}")


# Attack-class assignment, public behaviors only. The paper's figure also assigned behaviors that
# are not in the public ipi_arena_os data; those are not listed here.
CLASS = {
    "Verdict manipulation": ["fake-profile-approval", "fake-fedex-approval", "parser-backdoor"],
    "Permitted-sink exfil": ["slack-phishing"],
}


def fig_class_heatmap():
    D = json.loads((SD / "results" / "evasion_4pass.json").read_text())
    pa, pb = D["per_attack"], D["per_behavior"]
    cols = ["baseline", "causalarmor", "melon", "firewalls", "camel", "fides", "ipiguard"]

    classes = dict(CLASS)
    assigned = {b for v in classes.values() for b in v}
    classes["Tool execution (other)"] = [b for b in pb if b not in assigned]

    M, labels = [], []
    for cls, behs in classes.items():
        rates = {a: [] for a in cols}
        for h in pa:
            if pa[h]["behavior"] in behs:
                for a in cols:
                    rates[a].append(pa[h][a]["rate"])
        M.append([np.mean(rates[a]) for a in cols])
        labels.append(f"{cls}\n(n={len(rates['baseline'])} attacks)")
    M = np.array(M)

    fig, ax = plt.subplots(figsize=(14.5, 6.2))
    cmap = LinearSegmentedColormap.from_list("ev", ["#f7fbff", "#fdd", "#e34a33", "#7f0000"])
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), [NAME[a] for a in cols], fontsize=style.LABEL_SIZE)
    ax.set_yticks(range(len(labels)), labels, fontsize=style.LABEL_SIZE)
    for i in range(len(labels)):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                    fontsize=style.LABEL_SIZE,
                    color="white" if v > 0.5 else "#333333")
    ax.axvline(0.5, color="black", lw=1.4)
    ax.axvline(3.5, color="grey", lw=1.0, ls="--")
    ax.grid(False)

    # single group label spanning the CaMeL/FIDES/IPIGuard columns (Code-Then-Execute, Dual-LLM, Plan-Then-Execute)
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    fi = min(cols.index(a) for a in FORKS); la = max(cols.index(a) for a in FORKS)
    ax.plot([fi - 0.4, la + 0.4], [-0.14, -0.14], transform=tr,
            color="#4C72B0", lw=1.6, clip_on=False)
    ax.text((fi + la) / 2, -0.185, "Plan/Code-Then-Execute, Dual-LLM", transform=tr,
            ha="center", va="top", fontsize=style.LABEL_SIZE, color="#4C72B0")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Attack success rate", fontsize=style.LABEL_SIZE)
    cbar.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.subplots_adjust(top=0.88, bottom=0.18, left=0.17)
    style.title_and_dataset(fig, "Attack Success by Attack Class × Defense", None, title_y=0.95)
    out = HERE / "sysdef_class_heatmap.png"
    fig.savefig(out)
    print(f"wrote {out}")


def fig_delivery():
    """Attack delivery rate per defense: fraction of rollouts where the carrier tool actually
    executed so the injection reached the agent (the denominator that would gate a delivered-ASR).
    Uniformly high (95.7-99.1%), which is why the un-gated and delivery-gated ASR agree to <=0.7pp."""
    pa = json.loads((SD / "results" / "evasion_4pass.json").read_text())["per_attack"]
    arms = ["baseline", "causalarmor", "melon", "firewalls", "camel", "fides", "ipiguard"]
    rate = {a: sum(pa[h][a]["n_deliv"] for h in pa) / sum(pa[h][a]["n"] for h in pa) * 100
            for a in arms}

    fig, ax = plt.subplots(figsize=(12, 6.0))
    vals = [rate[a] for a in arms]
    ax.bar(range(len(arms)), vals, color=[color(a) for a in arms])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.12, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=style.LABEL_SIZE, fontweight="bold")
    ax.set_xticks(range(len(arms)), [NAME[a] for a in arms])
    ax.set_ylabel("Attack delivery rate (%)")  # y-axis truncated at 90: noted in the paper caption
    ax.set_ylim(90, 101)
    ax.axhline(100, color="grey", lw=0.6, ls=":")
    kind_legend(ax, loc="lower right")
    fig.subplots_adjust(top=0.90, bottom=0.09)
    style.title_and_dataset(fig, "System Defenses — Attack Delivery Rate", None, title_y=0.96)
    out = HERE / "sysdef_delivery.png"
    fig.savefig(out)
    print(f"wrote {out}")


def main():
    style.apply()
    fig_asr()
    fig_utility()
    fig_class_heatmap()
    fig_delivery()


if __name__ == "__main__":
    main()
