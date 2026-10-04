#!/usr/bin/env python
"""Research-update plots for classifier defenses (P1, P2, P5).

Numbers are copied from the canonical result artifacts (cited per block); no new runs.
  P1 detection landscape (ipi head)     -> p1_detection_landscape.png
  P2 detection landscape (policy head)  -> p2_policy_landscape.png
  P5 whose reasoning helps (histogram)  -> p5_reasoning.png
  P7 CoT-surfacing tradeoff plane       -> p7_cot_tradeoff.png
  P8 Cygnal model-swap tradeoff plane   -> p8_cygnal_model_swap.png
"""
import json
import os
import collections
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

CD = "classifier-defenses"
ARENA = "ipi_arena/transfer_handover"
DETS = ["protectai-v2", "promptguard2-86m", "datasentinel-mistral7b",
        "cygnal-code-8b", "granite-guardian-4.1-8b", "granite-guardian-4.1-8b-think"]
LBL = {"protectai-v2": "ProtectAI-v2", "promptguard2-86m": "PromptGuard2-86M",
       "datasentinel-mistral7b": "DataSentinel", "cygnal-code-8b": "Cygnal-8b",
       "granite-guardian-4.1-8b": "Granite-4.1-8b", "granite-guardian-4.1-8b-think": "Granite-think",
       "llmjudge-gpt-5.6-luna": "Luna judge*", "llmjudge-gemini-3-flash": "Gemini judge*",
       "llmjudge-qwen3.5-2b": "Qwen3.5-2B*", "llmjudge-qwen3.5-9b": "Qwen3.5-9B*"}

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
VIOLET, ORANGE, GREEN = "#4a3aa7", "#eb6834", "#008300"
GOOD, BAD, OFFGREY = "#0ca30c", "#d03b3b", "#c9c8c3"
HERE = "plots"
CMAP = LinearSegmentedColormap.from_list(
    "brandblue", ["#eef4fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
COLS = ["2025", "Q1", "Q2", "overall"]


def dataset_footer(fig, text, width, y=0.015, fontsize=8.4):
    """Centered gray dataset-description footer, wrapped to `width` chars so it never clips the
    figure edge (each explicit '\\n' segment is wrapped independently)."""
    wrapped = "\n".join(textwrap.fill(seg, width) for seg in text.split("\n"))
    fig.text(0.5, y, wrapped, ha="center", va="bottom", color=MUTED,
             fontsize=fontsize, linespacing=1.5)


def heatmap(dets, M, title, subtitle, datanote, out, figsize, rectbottom=0.17, footwidth=112):
    """detector x corpus detection heatmap. Shared color scale (10–90%) so P1 and P2 compare.
    datanote = dataset-explaining footer (makes the figure self-contained); subtitle = reading hint."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    im = ax.imshow(M, cmap=CMAP, vmin=10, vmax=90, aspect="auto")
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels(COLS, fontsize=11)
    ax.set_yticks(range(len(dets))); ax.set_yticklabels(dets, fontsize=11)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                    color="white" if M[i, j] >= 50 else INK,
                    fontsize=11, fontweight="bold" if j == 3 else "normal")
    ax.axvline(2.5, color=SURFACE, lw=4)  # separate the aggregate column
    ax.set_title(title, color=INK, fontsize=13, pad=12)
    dataset_footer(fig, datanote + "\n" + subtitle, footwidth)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("detection %", color=MUTED, fontsize=9); cb.ax.tick_params(colors=MUTED)
    cb.outline.set_edgecolor(GRID)
    fig.tight_layout(rect=[0, rectbottom, 1, 1])
    fig.savefig(out, dpi=150, facecolor=SURFACE); print("wrote", out); plt.close(fig)


# P1 — ipi head, correct-site %
# Cygnal = post-2026-07-13 endpoint model (re-run 2026-07-24; old model was [78.1, 67.4, 50.8, 67.4]).
# StackOne-T2 = shipped single-head default (main >= 0.64), matching the op/roc plots' `blocked`.
def p1():
    heatmap(
        ["GPT-5.6-Luna judge*", "Gemini-3-Flash judge*", "Cygnal-8b",
         "Qwen3.5-9B judge*", "StackOne-T2", "ProtectAI-v2", "Qwen3.5-2B judge*",
         "Granite-4.1-8b", "PromptGuard2-86M", "DataSentinel"],
        np.array([[93.6, 95.7, 87.1, 94.4], [94.6, 94.0, 92.6, 94.0],
                  [88.8, 81.8, 54.7, 80.0], [85.9, 75.9, 57.1, 75.5],
                  [66.9, 74.6, 59.7, 71.7], [69.0, 55.2, 55.6, 57.5],
                  [71.1, 39.1, 30.8, 43.5], [64.5, 32.5, 28.7, 37.5], [39.7, 26.2, 21.5, 27.9], [26.9, 23.4, 20.5, 23.6]]),
        "Injection detection at the true site  (correct-site %, ipi head)",
        "rows sorted by overall · only the FRONTIER judges hold flat across corpora — "
        "trained detectors AND small open judges (Qwen) all decay 2025 → Q1 → Q2",
        "Data: attack store — 26,466 recorded IPI attacks.  Columns = Arena corpora  "
        "2025 = ipi_2025 (33 beh) · Q1 = Q1'26 (54) · Q2 = Q2'26 (3);  overall = pooled.\n"
        "A hit = the detector flags the true injection turn (correct-site).\n"
        "*LLM judges scored on the seed-0 stratified 1/5 subset (5,292 records); on that same "
        "subset Cygnal reads 88.4/81.5/50.4/79.3 vs 88.8/81.8/54.7/80.0 full — subset is representative.",
        f"{HERE}/p1_detection_landscape.png", (8.4, 6.6))


# P2 — policy head, violation@harmful-action %.
# Only the two LLM monitors have a policy head.
# Cygnal = post-2026-07-13 endpoint model (re-run 2026-07-24; old model was [90.1, 64.2, 44.4, 66.1]).
def p2():
    heatmap(
        ["Gemini-3-Flash judge*", "GPT-5.6-Luna judge*", "Qwen3.5-9B judge*", "Cygnal-8b",
         "Qwen3.5-2B judge*", "Granite-4.1-8b"],
        np.array([[87.4, 91.4, 90.8, 90.7], [85.7, 88.0, 87.3, 87.6],
                  [91.6, 81.7, 86.6, 83.8], [83.7, 70.8, 40.3, 69.5],
                  [70.5, 59.1, 13.5, 55.9], [50.5, 21.0, 12.8, 24.8]]),
        "Policy-violation detection at the harmful action  (%, policy head)",
        "frontier judges hold flat; Cygnal and the small open judge (Qwen3.5-2B) decay 2025 → Q1 → Q2",
        "Data: attack store — 26,466 recorded IPI attacks (corpora ipi_2025 · Q1'26 · Q2'26);  score = "
        "policy violation flagged at the decisive harmful action.\n"
        "Only LLM monitors expose an action-monitoring 'policy' head.\n"
        "*LLM judges scored on the seed-0 stratified 1/5 subset; Cygnal on that subset reads "
        "85.0/70.6/37.8/69.3 vs 83.7/70.8/40.3/69.5 full — subset is representative.",
        f"{HERE}/p2_policy_landscape.png", (8.4, 5.9), rectbottom=0.22)


# P5 — surfacing the agent's CoT to the monitor: the two policy-head monitors SPLIT.
# New-model Cygnal (post-07-13 endpoint swap, re-run 2026-07-24) GAINS +8.6; Granite (local,
# unaffected by the swap) still drops −3.5. Old-model Cygnal dropped −5.1 (sign reversal).
# The guardian-think-mode arm was dropped: it rested on Granite alone (one defense).
# source: CoT-surfacing (agent CoT) results.
def p5():
    conds = [  # (label, off, on)
        ("Cygnal (new model)", 64.5, 73.1), ("Granite", 19.2, 15.7),
    ]
    xs = [0, 1]; w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for x, (lab, off, on) in zip(xs, conds):
        d = on - off; dc = GOOD if d > 0 else BAD
        ax.bar(x - w / 2, off, w, color=OFFGREY, zorder=3)
        ax.bar(x + w / 2, on, w, color=dc, zorder=3)
        ax.text(x - w / 2, off + 0.8, f"{off:.1f}", ha="center", va="bottom", fontsize=9, color=MUTED)
        ax.text(x + w / 2, on + 0.8, f"{on:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.text(x, max(off, on) + 3.5, f"Δ{d:+.1f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=dc)
    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in conds], fontsize=11)
    ax.set_ylabel("policy-head detection rate (%)", color=INK, fontsize=11)
    ax.set_ylim(0, 84); ax.set_xlim(-0.7, 1.7)
    ax.set_title("Surfacing the agent's chain-of-thought to the monitor: the monitors split",
                 color=INK, fontsize=12.5, pad=10)
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(handles=[Patch(color=OFFGREY, label="off — actions only"),
                       Patch(color=GOOD, label="on — CoT surfaced (rises)"),
                       Patch(color=BAD, label="on — CoT surfaced (drops)")],
              fontsize=8.5, frameon=True, facecolor="white", edgecolor=GRID,
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    dataset_footer(
        fig,
        "Policy-head detection at the harmful action, over the attack store (26,466 recorded IPI attacks; "
        "corpora ipi_2025 · Q1'26 · Q2'26), on the subset carrying real agent CoT.\n"
        "'off' = monitor sees the agent's actions only;  'on' = the agent's chain-of-thought is also surfaced.  "
        "Cygnal (post-2026-07-13 endpoint model) treats the CoT as evidence and gains; Granite is placated and drops.\n"
        "The pre-07-13 Cygnal model also dropped (59.9→54.8, −5.1) — the model swap reversed the sign.",
        width=104)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.savefig(f"{HERE}/p5_reasoning.png", dpi=150, facecolor=SURFACE)
    print("wrote p5_reasoning.png"); plt.close(fig)


# ================= task-type breakdowns (computed from artifacts) =================
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


def _ev_flag(e):
    if "blocked" in e:
        return bool(e["blocked"])
    if "flagged" in e:
        return bool(e["flagged"])
    if e.get("stage") == "after_tool_response":
        return bool(e.get("ipi"))
    return False


# T1 — injection detection by task type (Tool Use vs Coding), ipi correct-site
def t1():
    a2a = _aid2area()
    order = ["llmjudge-gpt-5.6-luna", "llmjudge-gemini-3-flash", "cygnal-code-8b",
             "llmjudge-qwen3.5-9b", "protectai-v2", "llmjudge-qwen3.5-2b", "granite-guardian-4.1-8b",
             "promptguard2-86m", "datasentinel-mistral7b"]  # match P1 row order
    det = {}
    for d in order:
        by = collections.defaultdict(lambda: [0, 0])
        for l in open(f"{CD}/results_ipi/{d}.jsonl"):
            r = json.loads(l); a = a2a.get(r["attack_id"]); fc = r.get("flaggedCorrectMessage")
            if a is None or fc is None:
                continue
            by[a][1] += 1; by[a][0] += 1 if fc else 0
        det[d] = {a: 100 * v[0] / v[1] for a, v in by.items()}
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(10.0, 6.1))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    tu = [det[d]["Tool Use"] for d in order]; co = [det[d]["Coding"] for d in order]
    b1 = ax.bar(x - w / 2, tu, w, color="#2a78d6", label="Tool Use", zorder=3)
    b2 = ax.bar(x + w / 2, co, w, color="#eb6834", label="Coding", zorder=3)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.1f", fontsize=8.5, color=INK, padding=2)
    # bracket the LEADING contiguous run that actually satisfies Coding >= Tool Use
    llm = []
    for i, d in enumerate(order):
        if det[d]["Coding"] >= det[d]["Tool Use"]:
            llm.append(i)
        else:
            break
    if llm:
        ax.annotate("LLM monitors — Coding > Tool Use", xy=((min(llm) + max(llm)) / 2, 107),
                    ha="center", va="bottom", fontsize=9.5, color=MUTED)
        ax.plot([min(llm) - 0.45, max(llm) + 0.45], [105, 105], color=MUTED, lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([LBL[d] for d in order], fontsize=9, rotation=18, ha="right")
    ax.set_ylabel("correct-site detection (%)", color=INK, fontsize=11); ax.set_ylim(0, 120)
    ax.set_title("Injection detection by task type  (ipi correct-site)\n"
                 "the CAPABLE LLM monitors do better on Coding; trained encoders and the small\n"
                 "2B judge do the reverse (DataSentinel excepted — it sits near its noise floor)",
                 color=INK, fontsize=12.5, pad=10)
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(title="task type", fontsize=9.5, title_fontsize=9.5, frameon=True,
              facecolor="white", edgecolor=GRID, loc="upper right")
    dataset_footer(
        fig,
        "ipi correct-site detection over the attack store (26,466 recorded IPI attacks), split by Arena task area.  "
        "Store covers Tool Use (49 behaviors) and Coding (41); no Computer Use.\n"
        "Same attack pool as the P1 heatmap — here disaggregated by task type rather than corpus.  "
        "*LLM judges on the seed-0 1/5 subset (3,506 Tool Use / 1,786 Coding records).",
        width=130)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.savefig(f"{HERE}/t1_detection_by_tasktype.png", dpi=150, facecolor=SURFACE)
    print("wrote t1_detection_by_tasktype.png"); plt.close(fig)


# T2 — over-defense by benign task type, per-event injection-head FP (heatmap)
def t2():
    suites = ["banking", "workspace", "travel", "slack", "coding"]
    M = np.full((len(DETS), len(suites)), np.nan)
    for di, d in enumerate(DETS):
        cnt = {s: [0, 0] for s in suites}
        for l in open(f"{CD}/fp_benign_agentdojo/detector_fp/results/{d}.jsonl"):
            r = json.loads(l); s = r["attack_id"].split("/")[0]
            for e in r.get("events", []):
                if e.get("stage") == "after_tool_response" and s in cnt:
                    cnt[s][1] += 1; cnt[s][0] += 1 if _ev_flag(e) else 0
        p = f"{CD}/fp_rebench/results/{d}.jsonl"
        if os.path.exists(p):
            for l in open(p):
                for e in json.loads(l).get("events", []):
                    if e.get("stage") == "after_tool_response":
                        cnt["coding"][1] += 1; cnt["coding"][0] += 1 if _ev_flag(e) else 0
        for si, s in enumerate(suites):
            if cnt[s][1]:
                M[di, si] = 100 * cnt[s][0] / cnt[s][1]
    order = np.argsort(-np.nanmean(M, axis=1))  # detectors by mean FP desc
    M = M[order]; rows = [LBL[DETS[i]] for i in order]
    reds = LinearSegmentedColormap.from_list(
        "fp", ["#fbeee7", "#f8c9ad", "#ef9666", "#e0562f", "#b5341a", "#7a1f0f"])
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    im = ax.imshow(M, cmap=reds, vmin=0, vmax=65, aspect="auto")
    ax.set_xticks(range(len(suites))); ax.set_xticklabels(
        ["banking", "workspace", "travel", "slack", "coding\n(SWE-rebench)"], fontsize=10)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=10.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                    color="white" if M[i, j] >= 35 else INK, fontsize=10.5)
    ax.axvline(3.5, color=SURFACE, lw=4)  # tool-use suites | coding
    ax.set_title("Over-defense by benign task type  (per-event injection-head FP %)\n"
                 "false positives concentrate in banking / workspace; coding is relatively safe",
                 color=INK, fontsize=12.5, pad=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("FP %", color=MUTED, fontsize=9); cb.ax.tick_params(colors=MUTED)
    cb.outline.set_edgecolor(GRID)
    dataset_footer(
        fig,
        "Per-event injection-head false positives on injection-free BENIGN data (any flag = FP):  "
        "banking / workspace / travel / slack from AgentDojo benign (97 rollouts, 349 tool outputs);  "
        "coding from SWE-rebench (50 clean OpenHands trajectories, 2,811 tool outputs).\n"
        "rows sorted by mean FP · left of the gap = AgentDojo tool-use suites, right = coding.",
        width=120)
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    fig.savefig(f"{HERE}/t2_overdefense_by_tasktype.png", dpi=150, facecolor=SURFACE)
    print("wrote t2_overdefense_by_tasktype.png"); plt.close(fig)


# P7 — CoT-surfacing tradeoff plane: benign FP (x) vs attack detection (y), off→on arrows.
# New-model Cygnal moves UP-LEFT (better on both axes: CoT is evidence); Granite moves DOWN-LEFT
# (loses detection and benign firing together: generic desensitization).
# Sources: compare_cot.py 2026-07-24 (detection off→on on the 9,271-record CoT-bearing store
# subset: Cygnal 64.5→73.1; Granite 19.2→15.7 — Granite is local, unaffected by the 07-13 swap)
# + the benign CoT-exposure experiment (benign per-traj FP, 108 GLM-5.2 Toolathlon tasks,
# both arms scored 07-17 = post-swap model: Cygnal 12.0→6.5 [13→7/108, exact McNemar p=0.070,
# n.s.]; Granite 80.6→68.5 [p=0.0072]).
def p7():
    ents = [  # (label, color, (fp_off, det_off), (fp_on, det_on), delta_note)
        ("Cygnal (new model)", GREEN, (12.0, 64.5), (6.5, 73.1),
         "detection +8.6 · FP −5.5pt (n.s.)"),
        ("Granite", VIOLET, (80.6, 19.2), (68.5, 15.7),
         "detection −3.5 · FP −12.1pt"),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for lab, c, off, on, note in ents:
        ax.annotate("", xy=on, xytext=off, zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2, mutation_scale=18,
                                    shrinkA=7, shrinkB=1))
        ax.scatter(*off, s=80, facecolor=SURFACE, edgecolor=c, lw=2, zorder=5)
        ax.scatter(*on, s=80, facecolor=c, edgecolor=SURFACE, lw=1.5, zorder=5)
        ax.text(off[0] + 2.5, off[1] - 0.4, "off", fontsize=8.5, color=MUTED, va="top")
        ax.text(on[0], on[1] - 2.2, "on", fontsize=8.5, color=MUTED, va="top", ha="center")
    ax.text(16, 72.0, "Cygnal (new model)", fontsize=11, fontweight="bold", color=INK)
    ax.text(16, 68.6, ents[0][4], fontsize=9, color=MUTED)
    ax.text(58, 24.5, "Granite", fontsize=11, fontweight="bold", color=INK)
    ax.text(58, 21.1, ents[1][4], fontsize=9, color=MUTED)
    ax.text(3, 95, "⬉ better", fontsize=11, color=MUTED, style="italic")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("benign false-positive rate (%, per Toolathlon trajectory)", color=INK, fontsize=11)
    ax.set_ylabel("attack detection rate (%, policy head)", color=INK, fontsize=11)
    ax.set_title("Surfacing agent CoT: the new Cygnal improves on both axes",
                 color=INK, fontsize=12.5, pad=10)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(handles=[Patch(color=GREEN, label="Cygnal (post-07-13 endpoint model)"),
                       Patch(color=VIOLET, label="Granite-Guardian-4.1-8b (local)")],
              fontsize=9, frameon=True, facecolor="white", edgecolor=GRID,
              loc="upper right")
    dataset_footer(
        fig,
        "Detection axis: policy head fired at the harmful action — attack store, the 9,271 records carrying real "
        "agent CoT (Cygnal re-run 2026-07-24).\n"
        "Benign-FP axis: any-head flag per trajectory — 108 injection-free GLM-5.2 Toolathlon admin tasks "
        "(fp_cot_exposure, scored 2026-07-17).\n"
        "FP deltas: exact McNemar p = 0.070 Cygnal (not significant, near-zero floor) · p = 0.0072 Granite.  "
        "Cygnal detection delta = net +798 flips on 9,271 pairs.",
        width=110)
    fig.tight_layout(rect=[0, 0.145, 1, 1])
    fig.savefig(f"{HERE}/p7_cot_tradeoff.png", dpi=150, facecolor=SURFACE)
    print("wrote p7_cot_tradeoff.png"); plt.close(fig)


# P8 — the 2026-07-13 Cygnal endpoint model swap, in the same FP x detection plane as P7.
# Arrow = old model -> new model (open marker = pre-07-13, filled = post). One panel per head, so
# each panel's x and y come from the SAME head (stricter than P7, which paired any-head FP with the
# policy head). Detection y is store-wide, so it is identical across the arrows within a panel —
# only x varies by benign dataset.
# ALL METRICS ARE PER EVENT, using the canonical event sets from extract_scores.py (the same
# definitions behind scores.json / the op+ROC figures), NOT per record or per trajectory:
#   ipi    positives = after_tool_response at a containsIPI turn;  negatives = every benign
#                      after_tool_response.  decision = cygnal `ipi` bool.
#   policy positives = after_assistant_tool_call with is_harmful_action;  negatives = every benign
#                      after_assistant_tool_call.  decision = violated_rules | observe_violation>0
#                      | violation>0  (detectors.py:267).
# Recomputed 2026-07-25 from both eras' sidecars; reproduces the canonical numbers exactly
# (ipi per-event TPR 59.6->70.7, policy 66.1->69.5, Toolathlon ipi FP 0.23%).
# AgentDojo is EXCLUDED here, as everywhere else in plots/ (its benign env fills carry phishy
# fixtures => not clean negatives). For the record, it is where the swap's big FP drop lives:
# per-event ipi 12.61->1.18, policy 11.79->1.90 — see the footer.
# Sources: results_ipi/{archive_oldmodel_pre20260713/,}cygnal-code-8b.jsonl (26,466 recs each era;
# 29,939 ipi-site events / 26,146 harmful-action events) + fp_rebench/ (matched first-25 trajs:
# the old-era file is MIXED-ERA, trajs 1-25 old / 26-50 new; 1,479 ipi / 1,504 action events)
# + fp_toolathlon/ (108 recs, 4,286 ipi / 2,461 action events, NEW model only — that testbed was
# first scored 07-16, after the swap, so it has no old-era point).
def p8():
    panels = [
        ("ipi head", "injection-site detection (%, per event)",
         "benign false positives (%, per event — ipi head)",
         (59.6, 70.7), 0.34, [  # (label, color, marker, x_old, x_new)
             ("SWE-rebench (coding)", VIOLET, "D", 0.00, 0.00),
             ("Toolathlon (admin)", ORANGE, "^", None, 0.23)]),
        ("policy head", "violation@harmful-action (%, per event)",
         "benign false positives (%, per event — policy head)",
         (66.1, 69.5), 1.75, [
             ("SWE-rebench (coding)", VIOLET, "D", 0.73, 0.53),
             ("Toolathlon (admin)", ORANGE, "^", None, 1.26)]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.6))
    fig.patch.set_facecolor(SURFACE)
    for ax, (head, ylab, xlab, (dy_old, dy_new), xmax, ents) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for lab, c, mk, xo, xn in ents:
            if xo is None:  # new-model-only dataset: lone dot, called out so it cannot read
                ax.scatter(xn, dy_new, s=95, marker=mk, facecolor=c,  # as another arrowhead
                           edgecolor=SURFACE, lw=1.5, zorder=6)
                ax.annotate("Toolathlon\nnew model only\n(no old-era run)",
                            xy=(xn, dy_new), xytext=(xn - 0.02 * xmax, dy_new - 3.2),
                            fontsize=8.2, color=c, ha="center", va="top", zorder=6,
                            arrowprops=dict(arrowstyle="-", color=c, lw=0.9, shrinkA=2, shrinkB=5))
                continue
            ax.annotate("", xy=(xn, dy_new), xytext=(xo, dy_old), zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=2, mutation_scale=17,
                                        shrinkA=7, shrinkB=3))
            ax.scatter(xo, dy_old, s=85, marker=mk, facecolor=SURFACE, edgecolor=c, lw=2, zorder=5)
            ax.scatter(xn, dy_new, s=85, marker=mk, facecolor=c, edgecolor=SURFACE, lw=1.5, zorder=5)
            if xo == xn:  # zero-length arrow: label to the right, else it clips the y-axis
                ax.text(xn + 0.035 * xmax, (dy_old + dy_new) / 2,
                        "SWE-rebench\nFP unchanged\n(0 of 1,479 events,\nboth eras)",
                        fontsize=8.2, color=c, ha="left", va="center", zorder=6)
            else:
                ax.text((xo + xn) / 2, dy_new + 0.7,
                        f"SWE-rebench\nFP {xo:.2f} → {xn:.2f}", fontsize=8.2, color=c,
                        ha="center", va="bottom", zorder=6)
        ax.text(0.03 * xmax, 78.4, "⬉ better", fontsize=10.5, color=MUTED, style="italic")
        ax.set_xlim(-0.045 * xmax, xmax); ax.set_ylim(55, 80)
        ax.set_xlabel(xlab, color=INK, fontsize=10)
        ax.set_ylabel(ylab, color=INK, fontsize=10)
        ax.set_title(f"{head}   ·   detection {dy_old:.1f} → {dy_new:.1f}  "
                     f"({dy_new - dy_old:+.1f} pt)", color=INK, fontsize=11.5, pad=9)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED)
    # figure-level legend: color/marker = benign dataset, fill = model era
    handles = [plt.Line2D([], [], color=VIOLET, marker="D", lw=2, label="SWE-rebench (coding)"),
               plt.Line2D([], [], color=ORANGE, marker="^", lw=0, label="Toolathlon (admin)"),
               plt.Line2D([], [], color=INK, marker="o", lw=0, mfc=SURFACE, mec=INK, mew=2,
                          label="open = old model (pre 07-13)"),
               plt.Line2D([], [], color=INK, marker="o", lw=0, mfc=INK, label="filled = new model")]
    fig.legend(handles=handles, fontsize=9, frameon=True, facecolor="white", edgecolor=GRID,
               loc="lower center", bbox_to_anchor=(0.5, 0.283), ncol=4, columnspacing=1.6)
    fig.suptitle("The 2026-07-13 Cygnal endpoint model swap: on clean negatives it is a "
                 "near-free detection gain", color=INK, fontsize=13, y=0.975)
    dataset_footer(
        fig,
        "ALL METRICS PER EVENT, on the canonical event sets behind scores.json (extract_scores.py) — "
        "not per record or per trajectory.  Detection (y): attack store, 26,466 attacks = 29,939 "
        "injection-site / 26,146 harmful-action events, both eras scored on the identical event set "
        "(old = archive_oldmodel_pre20260713 sidecar, new = 2026-07-24 re-run).\n"
        "False positives (x): every benign event of that head is a negative.  SWE-rebench = first 25 "
        "trajs (1,479 ipi / 1,504 action events), the only ones with an old-era score (its archive "
        "file is mixed-era).  Toolathlon (4,286 / 2,461 events) has NO old-model point — first scored "
        "2026-07-16, after the swap.\n"
        "AgentDojo is excluded as in every plots/ figure (phishy benign fixtures ⇒ not clean "
        "negatives), but is where the swap's large FP drop lives: per-event ipi 12.61 → 1.18, policy "
        "11.79 → 1.90.  Read together: the new model mostly stopped flagging sus-but-injection-free "
        "text; on clean traffic it was already near zero, so the swap bought detection almost free.",
        width=150)
    fig.tight_layout(rect=[0, 0.335, 1, 0.95])
    fig.savefig(f"{HERE}/p8_cygnal_model_swap.png", dpi=150, facecolor=SURFACE)
    print("wrote p8_cygnal_model_swap.png"); plt.close(fig)


p1(); p2(); p5(); t1(); t2(); p7(); p8()
