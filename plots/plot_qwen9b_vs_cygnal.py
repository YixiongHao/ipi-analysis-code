#!/usr/bin/env python
"""Qwen3.5-9B judge vs Cygnal — operating-point dots on TPR/FPR axes.

Data (08-01): qwen = n=2 complete full-set temp-1 sampled passes (store 56,085
events + SWE-rebench-50 5,672 + Toolathlon-108 6,747; sidecars .s1/.s2, dot =
mean across passes); Cygnal = full-set deterministic sidecars. NO error bars
(user call 08-01: cross-pass and binomial spread are both sub-point at these n
— cross-pass <=0.5pt everywhere despite 33% per-event flip at temp 1). If only
one pass exists and Toolathlon has no full pass, the script falls back to the
seed-0 1/5 subset sidecar (.fifth.jsonl) and flags it in footer+legend.
Judge no-verdict/error events score as NOT flagged and are counted in the footer.

Conventions match plot_roc.py: per-event rates; positives = true injection-site
events (ipi head) / harmful-action events (policy head); negatives = every event
of that head in the benign set. Marker = FP set (D SWE-rebench, ^ Toolathlon).
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from extract_scores import CD, decision, event_is, load_sites

HERE = os.path.dirname(os.path.abspath(__file__))

CYGNAL, QWEN = "cygnal-code-8b", "llmjudge-qwen3.5-9b"
COLOR = {CYGNAL: "#008300", QWEN: "#00a0a8"}          # plot_roc.py DEF_COLOR
LABEL = {CYGNAL: "Cygnal-8b (deterministic)", QWEN: "Qwen3.5-9B judge (temp 1)"}
DS_MARKER = {"swe-rebench": "D", "toolathlon": "^"}
DS_LABEL = {"swe-rebench": "SWE-rebench (coding)", "toolathlon": "Toolathlon (admin)"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"


def rates(path, head, defense, site_of=None):
    """(n_flagged, n_events, n_error) over one sidecar; positives filter via site_of."""
    flagged = total = errors = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            site = site_of.get(r["attack_id"]) if site_of is not None else None
            for ev in r.get("events", []):
                if site_of is not None:
                    if not event_is(head, ev, site):
                        continue
                else:
                    want = "after_tool_response" if head == "ipi" else "after_assistant_tool_call"
                    if ev.get("stage") != want:
                        continue
                d = decision(defense, ev, head)
                if d is not None:
                    flagged += d
                    total += 1
                    errors += 1 if ev.get("error") else 0
    return flagged, total, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-store-dir", default=f"{CD}/results_ipi/qwen9b_samples")
    ap.add_argument("--qwen-rebench-dir", default=f"{CD}/fp_rebench/results_qwen9b_samples")
    ap.add_argument("--qwen-toolathlon-dir", default=f"{CD}/fp_toolathlon/results_qwen9b_samples")
    ap.add_argument("--out", default=f"{HERE}/qwen9b_vs_cygnal.png")
    args = ap.parse_args()

    cyg_store = f"{CD}/results_ipi/{CYGNAL}.jsonl"
    cyg_fp = {"swe-rebench": f"{CD}/fp_rebench/results/{CYGNAL}.jsonl",
              "toolathlon": f"{CD}/fp_toolathlon/results/{CYGNAL}.jsonl"}

    def sample_files(d):
        return sorted(glob.glob(os.path.join(d, f"{QWEN}.s*.jsonl")))

    store_files = sample_files(args.qwen_store_dir)
    if not store_files:
        raise SystemExit(f"no qwen sample sidecars in {args.qwen_store_dir}")
    qwen_fp_files = {}
    subset_flag = {}
    for ds, d in (("swe-rebench", args.qwen_rebench_dir), ("toolathlon", args.qwen_toolathlon_dir)):
        fs = sample_files(d)
        if fs:
            qwen_fp_files[ds], subset_flag[ds] = fs, False
        else:  # no full pass ran -> the flagged 1/5-subset fallback
            fifth = os.path.join(d, f"{QWEN}.fifth.jsonl")
            if not os.path.exists(fifth):
                raise SystemExit(f"{ds}: no s*.jsonl and no fifth fallback in {d}")
            qwen_fp_files[ds], subset_flag[ds] = [fifth], True

    single_pass = len(store_files) < 2
    sites = load_sites()

    cyg, qw = {}, {}
    for head in ("ipi", "policy"):
        f, t, _ = rates(cyg_store, head, CYGNAL, site_of=sites)
        cyg[head] = {"tpr": f / t, "n_pos": t, "fpr": {}}
        for ds, path in cyg_fp.items():
            f, t, _ = rates(path, head, CYGNAL)
            cyg[head]["fpr"][ds] = {"rate": f / t, "n": t}

        tprs, n_pos, err_pos = [], 0, 0
        for p in store_files:
            f, t, e = rates(p, head, QWEN, site_of=sites)
            tprs.append((f, t))
            n_pos = t
            err_pos += e   # summed across passes (footer reports the total)
        qw[head] = {"tpr": tprs, "n_pos": n_pos, "err_pos": err_pos, "fpr": {}}
        for ds, fs in qwen_fp_files.items():
            frs, n, err = [], 0, 0
            for p in fs:
                f, t, e = rates(p, head, QWEN)
                frs.append((f, t))
                n = t
                err += e
            qw[head]["fpr"][ds] = {"counts": frs, "n": n, "err": err, "subset": subset_flag[ds]}

    with open(args.out.replace(".png", ".json"), "w") as f:
        json.dump({"single_pass": single_pass, "store_files": store_files,
                   "fp_files": qwen_fp_files, "cygnal": cyg, "qwen": qw}, f, indent=2, default=str)

    # ---- figure ----
    def qpoint(counts):
        """counts = [(k, n), ...] per pass -> mean rate (no bars: cross-pass and
        binomial spread are both sub-point at these n, user call 08-01)."""
        rs = [k / n for k, n in counts]
        return sum(rs) / len(rs)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.78, top=0.85, bottom=0.19, wspace=0.10)
    fig.patch.set_facecolor(SURFACE)
    for ax, head in zip(axes, ("ipi", "policy")):
        ax.set_facecolor(SURFACE)
        ax.set_title("injection (ipi) head" if head == "ipi" else "policy head",
                     fontsize=12, color=INK, pad=10)
        ax.set_xlabel("per-event FP rate on benign set", fontsize=10, color=MUTED)
        ax.grid(True, color=GRID, lw=0.7, alpha=0.7)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1.0)

        ty = qpoint(qw[head]["tpr"])
        for ds in DS_MARKER:
            # cygnal: filled dot, no bars (deterministic hard labels)
            ax.scatter([cyg[head]["fpr"][ds]["rate"]], [cyg[head]["tpr"]],
                       color=COLOR[CYGNAL], marker=DS_MARKER[ds], s=150,
                       edgecolor="white", linewidth=1.2, zorder=6)
            fx = qpoint(qw[head]["fpr"][ds]["counts"])
            ax.scatter([fx], [ty], facecolors="none", edgecolors=COLOR[QWEN],
                       marker=DS_MARKER[ds], s=150, linewidth=2.2, zorder=6)
    axes[0].set_ylabel("per-event TPR on attack store", fontsize=10, color=MUTED)

    dleg = [Line2D([0], [0], marker="o", lw=0, markersize=10, markerfacecolor=COLOR[CYGNAL],
                   markeredgecolor="white", label=LABEL[CYGNAL]),
            Line2D([0], [0], marker="o", lw=0, markersize=10, markerfacecolor="none",
                   markeredgecolor=COLOR[QWEN], markeredgewidth=2.2, label=LABEL[QWEN])]
    leg1 = axes[1].legend(handles=dleg, title="defense (color)", loc="upper left",
                          bbox_to_anchor=(1.03, 1.0), fontsize=9, title_fontsize=9,
                          frameon=True, facecolor="white", edgecolor=GRID)
    axes[1].add_artist(leg1)
    mleg = [Line2D([0], [0], marker=DS_MARKER[ds], lw=0, markersize=10, markerfacecolor="none",
                   markeredgecolor=MUTED, markeredgewidth=2.0,
                   label=DS_LABEL[ds] + (" *1/5" if subset_flag[ds] else ""))
            for ds in DS_MARKER]
    axes[1].legend(handles=mleg, title="FP dataset (marker)", loc="upper left",
                   bbox_to_anchor=(1.03, 0.55), fontsize=9, title_fontsize=9,
                   frameon=True, facecolor="white", edgecolor=GRID)

    fig.suptitle("Qwen3.5-9B judge vs Cygnal — operating points",
                 fontsize=13, color=INK)
    e = qw["ipi"]; ep = qw["policy"]
    ce, cp = cyg["ipi"]["fpr"], cyg["policy"]["fpr"]
    if subset_flag["toolathlon"]:
        tool_desc = (f"▲ Toolathlon: Cygnal full 108 ({ce['toolathlon']['n']:,}/{cp['toolathlon']['n']:,}), "
                     f"*qwen 1/5 subset ({e['fpr']['toolathlon']['n']:,}/{ep['fpr']['toolathlon']['n']:,})")
    else:
        tool_desc = f"▲ Toolathlon full 108 ({e['fpr']['toolathlon']['n']:,}/{ep['fpr']['toolathlon']['n']:,})"
    npass = len(store_files)
    footer = [
        f"TPR: full store, {e['n_pos']:,} ipi / {ep['n_pos']:,} policy events.  "
        f"FP — ◆ SWE-rebench full 50 ({e['fpr']['swe-rebench']['n']:,}/{ep['fpr']['swe-rebench']['n']:,});  "
        f"{tool_desc}.",
        f"Qwen = {npass} full temp-1 sampled pass{'es' if npass > 1 else ''}; "
        f"no-verdict events{' (all passes,' if npass > 1 else ' ('} scored not-flagged): "
        f"store {e['err_pos'] + ep['err_pos']}, SWE-rebench {e['fpr']['swe-rebench']['err'] + ep['fpr']['swe-rebench']['err']}, "
        f"Toolathlon {e['fpr']['toolathlon']['err'] + ep['fpr']['toolathlon']['err']}.",
    ]
    fig.text(0.07, 0.02, "\n".join(footer), fontsize=8.4, color=MUTED, va="bottom")
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print("wrote", args.out)
    for head in ("ipi", "policy"):
        print(f"{head}: qwen TPR {qpoint(qw[head]['tpr']):.3f} vs cygnal {cyg[head]['tpr']:.3f} | " +
              " | ".join(f"{ds} FP qwen {qpoint(qw[head]['fpr'][ds]['counts']):.4f} vs cyg {cyg[head]['fpr'][ds]['rate']:.4f}"
                         for ds in DS_MARKER))


if __name__ == "__main__":
    main()
