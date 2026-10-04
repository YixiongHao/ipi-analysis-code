#!/usr/bin/env python
"""Radar: Cygnal (new model) vs StackOne Defender Tier 2 across detection, FP, latency.

All five axes are computed fresh from the canonical artifacts (no hardcoded numbers):
  - Detection Tool Use / Coding : record-level flaggedCorrectMessage from
    results_ipi/{cygnal-code-8b,stackone-defender-tier2}.jsonl, split by Arena area
    (same method as t1 in plot_experiments.py; store has no Computer Use).
  - Clean pass SWE-rebench / Toolathlon : 100 - per-event ipi-head FP% from
    fp_rebench/results + fp_toolathlon/results sidecars (cygnal_error events excluded).
  - Speed : 100 * (fastest p50 / p50), serial per-event latency p50 pooled over the
    102 matched serial events in scratchpad/cygnal_latency_rows.json (2026-07-27 probe).
Defender decision = `blocked` (single-head @0.64), consistent with plot_roc/plot_gmeans.
-> radar_cygnal_vs_defender.png
"""
import collections
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CD = "classifier-defenses"
ARENA = "ipi_arena/transfer_handover"
SCRATCH = "scratchpad"
HERE = "plots"

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
C_CYGNAL, C_DEFENDER = "#2a78d6", "#eb6834"
DEFS = {"cygnal": "cygnal-code-8b", "defender": "stackone-defender-tier2"}


def _aid2area():
    Q = {"ipi_2025": "q4_25", "ipi_2026_q1": "q1_26", "ipi_2026_q2": "q2_26"}
    area = {}
    for q in ("q4_25", "q1_26", "q2_26"):
        for bid, rec in json.load(open(f"{ARENA}/behaviors_{q}.json")).items():
            a = rec.get("area_name")
            area[(q, bid)] = "Coding" if a in ("Coding", "Coding Agent") else a
    out = {}
    for l in open(f"{CD}/store/attacks.jsonl"):
        r = json.loads(l)
        out[r["attack_id"]] = area.get((Q[r["corpus"]], r["behavior_id"]))
    return out


def detection_by_area(a2a):
    det = {}
    for k, d in DEFS.items():
        by = collections.defaultdict(lambda: [0, 0])
        for l in open(f"{CD}/results_ipi/{d}.jsonl"):
            r = json.loads(l)
            a = a2a.get(r["attack_id"])
            fc = r.get("flaggedCorrectMessage")
            if a is None or fc is None:
                continue
            by[a][1] += 1
            by[a][0] += 1 if fc else 0
        det[k] = {a: 100 * v[0] / v[1] for a, v in by.items()}
    return det


def _ev_flag(e):
    if "blocked" in e:            # defender: single-head @0.64 (plots convention)
        return bool(e["blocked"])
    return bool(e.get("ipi"))     # cygnal ipi head


def fp_rate(fp_dir, defense):
    flagged = total = 0
    for l in open(f"{fp_dir}/{defense}.jsonl"):
        for e in json.loads(l).get("events", []):
            if e.get("stage") != "after_tool_response" or e.get("cygnal_error"):
                continue
            total += 1
            flagged += _ev_flag(e)
    return 100 * flagged / total


def latency_p50():
    rows = [r for r in json.load(open(f"{SCRATCH}/cygnal_latency_rows.json"))
            if r["mode"] == "serial"]
    p50 = lambda xs: sorted(xs)[len(xs) // 2]
    return {"cygnal": p50([r["ms"] for r in rows]),
            "defender": p50([r["defender_ms"] for r in rows])}


def main():
    det = detection_by_area(_aid2area())
    fp = {k: {"rebench": fp_rate(f"{CD}/fp_rebench/results", d),
              "toolathlon": fp_rate(f"{CD}/fp_toolathlon/results", d)}
          for k, d in DEFS.items()}
    lat = latency_p50()

    # per-axis scale: (label, lo, hi, invert). invert=True -> outward = LOWER raw value.
    AXES = [("Detection\nTool Use", 50, 100, False),
            ("Detection\nCoding", 50, 100, False),
            ("False positives\nSWE-rebench", 0, 10, True),
            ("False positives\nToolathlon", 0, 10, True),
            ("Latency\n(serial p50, ms)", 0, 1000, True)]
    axes = [a[0] for a in AXES]
    series = {}
    raw = {}
    for k in DEFS:
        series[k] = [det[k]["Tool Use"], det[k]["Coding"],
                     fp[k]["rebench"], fp[k]["toolathlon"],
                     lat[k]]
        raw[k] = [f"{det[k]['Tool Use']:.1f}%", f"{det[k]['Coding']:.1f}%",
                  f"FP {fp[k]['rebench']:.1f}%", f"FP {fp[k]['toolathlon']:.1f}%",
                  f"{lat[k]:.0f} ms"]
    for k in DEFS:
        print(k, dict(zip([a.replace(chr(10), ' ') for a in axes], np.round(series[k], 1))),
              "| raw:", raw[k])

    def to_r(i, v):
        _, lo, hi, inv = AXES[i]
        frac = (hi - v) / (hi - lo) if inv else (v - lo) / (hi - lo)
        return float(np.clip(frac, 0, 1))

    n = len(axes)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])

    fig, ax = plt.subplots(figsize=(8.6, 8.6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for name, color, lbl in [("cygnal", C_CYGNAL, "Cygnal (cygnal-code-8b, new model)"),
                             ("defender", C_DEFENDER, "StackOne Defender T2 (single-head @0.64)")]:
        r = [to_r(i, v) for i, v in enumerate(series[name])]
        r_c = np.concatenate([r, r[:1]])
        ax.plot(ang_c, r_c, color=color, lw=2, zorder=3, label=lbl)
        ax.plot(ang_c, r_c, "o", color=color, ms=5.5, zorder=4)
        ax.fill(ang_c, r_c, color=color, alpha=0.10, zorder=2)

    # per-vertex raw-value labels, offset per series to avoid collisions
    off = {"cygnal": 9, "defender": -12}
    for name, color in [("cygnal", C_CYGNAL), ("defender", C_DEFENDER)]:
        for i, (a, v, txt) in enumerate(zip(ang, series[name], raw[name])):
            ax.annotate(txt, (a, to_r(i, v)), textcoords="offset points",
                        xytext=(0, off[name]), ha="center", fontsize=8.6,
                        color=INK, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec=color, lw=0.8, alpha=0.85))

    ax.set_xticks(ang)
    ax.set_xticklabels(axes, fontsize=10.5, color=INK)
    ax.tick_params(axis="x", pad=26)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])
    ax.grid(True, color=GRID, lw=0.6)
    ax.spines["polar"].set_color(GRID)

    # per-spoke numeric tick labels (each axis has its own scale)
    for i, (a, (_, lo, hi, inv)) in enumerate(zip(ang, AXES)):
        for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
            val = hi - frac * (hi - lo) if inv else lo + frac * (hi - lo)
            # outermost label tucks inside the ring so it doesn't hit the axis titles
            xy, va = ((5, -11), "top") if frac == 1.0 else ((5, 3), "bottom")
            ax.annotate(f"{val:g}", (a, frac), textcoords="offset points",
                        xytext=xy, ha="left", va=va,
                        fontsize=7, color=MUTED, zorder=1.5)

    ax.set_title("Cygnal vs StackOne Defender Tier 2 — detection, over-defense, latency\n"
                 "axes independently scaled (ticks on each spoke); outward = better",
                 color=INK, fontsize=12.5, pad=26)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=9.5,
              frameon=True, facecolor="white", edgecolor=GRID)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(f"{HERE}/radar_cygnal_vs_defender.png", dpi=150, facecolor=SURFACE)
    print("wrote radar_cygnal_vs_defender.png")


if __name__ == "__main__":
    main()
