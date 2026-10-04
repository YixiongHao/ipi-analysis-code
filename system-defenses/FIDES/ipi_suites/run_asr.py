"""Stage-4 ASR eval: judge-graded attack success, FIDES (P-T/P-F policy) vs no-policy baseline.

For each shared packet, run the labeled planner TWICE on the same (temp-0, deterministic) model:
  - baseline arm : policy OFF  -> the planner may execute the harmful action
  - fides arm    : generic P-T policy ON (from FRESH tool_policies, recomputed via the fixed
                   tool_resolver so Bash/shell/exec are consequential) -> harmful action blocked
Both transcripts are graded by ipi_eval's Arena judge; ASR = fraction with is_break=True.

We report ASR baseline vs fides (overall + per corpus), and the **delivered-gated** ASR (only
behaviors where the injection actually reached the planner — a block on an undelivered attack is
vacuous). Reduction = baseline ASR - fides ASR.

    cd system-defenses
    python -m FIDES.ipi_suites.run_asr --model qwen
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from FIDES.ipi_suites import run_pilot as R
from ipi_eval import behaviors as B
from ipi_eval import engine
from ipi_eval import judges
from ipi_eval import tool_resolver as TR
from ipi_suites_worldsim import make_worldsim_fallback

_HERE = Path(__file__).resolve().parent
_PACKETS = _HERE.parent.parent / "CaMeL" / "ipi_suites" / "packets"
_TX = _HERE / "transcripts_asr"


def _grade(behavior: dict, transcript: dict, judge_chat) -> bool:
    gen = []
    for i, c in enumerate(transcript["executed_calls"]):
        gen.append({"role": "assistant", "tool_calls": [{
            "id": f"call_{i}", "type": "function",
            "function": {"name": c["name"], "arguments": json.dumps(c.get("args", {}))}}]})
    if transcript.get("final_text"):
        gen.append({"role": "assistant", "content": transcript["final_text"]})
    if not gen:
        return False
    return bool(judges.score_behavior(behavior, gen, judge_chat, full_messages=gen)["is_break"])


def _worldsim(pkt: dict):
    """Goal-aware arena world-sim for this packet; None if the key/deps are missing (-> stub)."""
    try:
        return make_worldsim_fallback(pkt)  # goal_aware=True by default
    except Exception:
        return None


def _run_one(fp: Path, cfg: dict, judge_chat, all_worldsim: bool = True) -> dict:
    pkt = json.loads(fp.read_text())
    bid = pkt["behavior_id"]
    corpus = pkt.get("corpus", "?")
    carrier = pkt.get("carrier")
    canned = dict(pkt.get("canned", {}))
    # FRESH policies from the fixed tool_resolver (packets carry stale tool_policies).
    pkt["tool_policies"] = TR.tool_policies_for(pkt["tools"])
    try:
        ws = _worldsim(pkt)  # goal-aware arena world-sim (None if key/deps missing -> stub)
        behavior = B.get_behavior(bid)
        t_base = R.run(pkt, carrier, canned, cfg, policy=lambda tr: None,
                       worldsim=ws, all_worldsim=all_worldsim)  # policy OFF
        t_fides = R.run(pkt, carrier, canned, cfg,
                        worldsim=ws, all_worldsim=all_worldsim)  # generic P-T ON
        (_TX / f"{bid.strip()}.json").write_text(json.dumps(
            {"baseline": t_base, "fides": t_fides}, indent=2, default=str))
        return {
            "behavior": bid, "corpus": corpus,
            "delivered": t_base["injection_delivered"],
            "blocked_fides": t_fides["blocked"] is not None,
            "is_break_base": _grade(behavior, t_base, judge_chat),
            "is_break_fides": _grade(behavior, t_fides, judge_chat),
            "status": "ok",
        }
    except Exception as e:
        return {"behavior": bid, "corpus": corpus, "delivered": False, "blocked_fides": False,
                "is_break_base": False, "is_break_fides": False,
                "status": f"ERROR {type(e).__name__}: {str(e)[:120]}"}


def _asr(rows: list[dict], key: str, subset=None) -> str:
    rs = [r for r in rows if r["status"] == "ok" and (subset is None or subset(r))]
    n = len(rs)
    k = sum(r[key] for r in rs)
    return f"{k}/{n} ({100*k/n:.1f}%)" if n else "0/0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen", choices=["qwen", "gemini-flash", "gemini-3-flash", "glm"])
    ap.add_argument("--behaviors", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", default=None,
                    help="override output dir (default: transcripts_asr/); isolates a run")
    ap.add_argument("--target-thinking", action="store_true",
                    help="enable native reasoning on the GLM/cloud target (default OFF to save cost)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="planner (agent-under-eval) sampling temperature (default 0.0 = paper parity; "
                         "pass 1.0 for a hot-agent realism arm; judge stays 0.0)")
    ap.add_argument("--all-worldsim", dest="all_worldsim", action="store_true", default=True,
                    help="fully-simulated env (DEFAULT): answer EVERY non-carrier tool with the "
                         "goal-aware arena world-sim (ignore canned); carrier still delivers the "
                         "injection. Opt out with --no-all-worldsim.")
    ap.add_argument("--no-all-worldsim", dest="all_worldsim", action="store_false",
                    help="restore canned-primary replay (world-sim only as fallback for un-canned tools).")
    args = ap.parse_args()

    cfg = R.model_preset(args.model)
    cfg["thinking"] = args.target_thinking
    cfg["temperature"] = args.temperature
    if args.model == "qwen":
        cfg["no_think"] = not args.target_thinking
    global _TX
    if args.out_dir:
        _TX = Path(args.out_dir)
    _TX.mkdir(parents=True, exist_ok=True)
    judge_chat = engine.make_judge_chat()  # shared Gemini-3-Flash OpenRouter judge

    files = ([_PACKETS / f"{b}.json" for b in args.behaviors] if args.behaviors
             else sorted(_PACKETS.glob("*.json")))
    print(f"ASR eval: {len(files)} behaviors | model={cfg['model']} workers={args.workers}\n", flush=True)
    _run_one(files[0], cfg, judge_chat, args.all_worldsim)  # warm up

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, fp, cfg, judge_chat, args.all_worldsim): fp for fp in files}
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            mark = r["status"] if r["status"] != "ok" else (
                f"base={'BRK' if r['is_break_base'] else '---'} fides={'BRK' if r['is_break_fides'] else '---'}"
                f" blk={'Y' if r['blocked_fides'] else '.'}")
            print(f"  {r['behavior']:46s} {mark}", flush=True)

    rows.sort(key=lambda r: (r["corpus"], r["behavior"]))
    (_TX / "_asr_summary.json").write_text(json.dumps(rows, indent=2))

    print("\n=== ASR (judge-graded is_break) ===")
    print(f"  baseline (no policy) : {_asr(rows, 'is_break_base')}")
    print(f"  FIDES (P-T/P-F)      : {_asr(rows, 'is_break_fides')}")
    print(f"  --- delivered-gated (injection reached planner) ---")
    dlv = lambda r: r["delivered"]
    print(f"  baseline (delivered) : {_asr(rows, 'is_break_base', dlv)}")
    print(f"  FIDES    (delivered) : {_asr(rows, 'is_break_fides', dlv)}")
    print("\n  per-corpus (baseline -> fides):")
    for corp in sorted({r["corpus"] for r in rows}):
        sub = lambda r, c=corp: r["corpus"] == c
        print(f"    {corp:12s} {_asr(rows, 'is_break_base', sub)}  ->  {_asr(rows, 'is_break_fides', sub)}")
    errs = [r for r in rows if r["status"] != "ok"]
    if errs:
        print(f"\n  {len(errs)} errors:")
        for r in errs:
            print(f"    {r['behavior']:46s} {r['status']}")


if __name__ == "__main__":
    main()
