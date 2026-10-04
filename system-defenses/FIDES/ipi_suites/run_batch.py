"""Stage-3 generic harness: run the real FIDES labeled planner over all shared packets.

Reuses the CaMeL/FIDES shared packet set (`CaMeL/ipi_suites/packets/<id>.json`, each carrying the
deterministically-derived `carrier` + recorded `canned` outputs + `tool_policies`) and drives each
through `run_pilot.run` (bridge -> LabeledPlanningLoop -> generic P-T policy) on local Qwen3-32B.

Primary metric: **attack delivery coverage** — for how many behaviors does the injection actually
reach the planner (the planner calls the carrier, tainting the context)?  This is the vacuous-block
gate denominator: a "block" only counts where the injection was delivered. Also records whether the
P-T policy fired (FIDES engaged) and the executed tool calls. Grading/ASR is Stage 4.

    cd system-defenses
    python -m FIDES.ipi_suites.run_batch --model qwen
    # subset:  --behaviors cancel-appointment modify-cicd-workflow
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from FIDES.ipi_suites import run_pilot as R
from ipi_suites_worldsim import make_worldsim_fallback

_HERE = Path(__file__).resolve().parent
_PACKETS = _HERE.parent.parent / "CaMeL" / "ipi_suites" / "packets"  # shared packet set
_TX = _HERE / "transcripts"


def _worldsim(pkt: dict):
    """Per-packet world-sim fallback (None if no key / deps unavailable -> keep '{"status":"ok"}')."""
    try:
        return make_worldsim_fallback(pkt)
    except Exception:
        return None


def _run_one(fp: Path, cfg: dict, no_worldsim: bool = False, all_worldsim: bool = False,
             forced_delivery: bool = False, attack: dict | None = None) -> dict:
    pkt = json.loads(fp.read_text())
    if forced_delivery:
        pkt["forced_delivery"] = True  # run_pilot.framed_user emits the carrier-args hint
    bid = pkt["behavior_id"]
    corpus = pkt.get("corpus", "?")
    carrier = pkt.get("carrier")
    has_policy_tool = bool(pkt.get("tool_policies"))
    # Multi-string mode: swap the explicit injection into the packet (the plan/policy is injection-
    # blind); write a per-string transcript so N strings/behavior -> N records. attack_hash = md5 is
    # the cross-arm join key (matches build_transfer_store + the adapter valset).
    if attack is not None:
        pkt["injection"] = attack["attack_text"]
        out_name = f"{bid.strip()}#{attack['idx']:03d}.json"
    else:
        out_name = f"{bid.strip()}.json"
    # all_worldsim implies world-sim ON (overrides --no-worldsim); build the fallback then route
    # EVERY non-carrier tool through it.
    ws = None if (no_worldsim and not all_worldsim) else _worldsim(pkt)
    try:
        t = R.run(pkt, carrier, dict(pkt.get("canned", {})), cfg, worldsim=ws, all_worldsim=all_worldsim)
        t.setdefault("behavior_id", bid)
        t.setdefault("corpus", corpus)
        if attack is not None:
            t["attack_id"] = attack["attack_id"]
            t["attack_hash"] = attack["attack_hash"]
        (_TX / out_name).write_text(json.dumps(t, indent=2, default=str))
        return {
            "behavior": bid, "corpus": corpus, "carrier": carrier,
            "attack_id": attack["attack_id"] if attack else None,
            "has_policy_tool": has_policy_tool,
            "delivered": t["injection_delivered"],
            "blocked": t["blocked"] is not None,
            "blocked_call": (t.get("blocked_call") or {}).get("name"),
            "n_calls": len(t["called_names"]), "calls": t["called_names"], "status": "ok",
        }
    except Exception as e:
        return {"behavior": bid, "corpus": corpus, "carrier": carrier,
                "attack_id": attack["attack_id"] if attack else None,
                "has_policy_tool": has_policy_tool, "delivered": False, "blocked": False,
                "blocked_call": None, "n_calls": 0, "calls": [],
                "status": f"ERROR {type(e).__name__}: {str(e)[:140]}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen",
                    choices=["qwen", "gemini-flash", "gemini-3-flash", "glm", "deepseek"])
    ap.add_argument("--behaviors", nargs="*", default=None)
    ap.add_argument("--strings-json", default=None,
                    help="explicit {behavior_id:[injection,...]} source (e.g. hard_sample/eval_strings_381union.json); "
                         "runs every string per behavior as its own record (attack_hash=md5), swapped into "
                         "the behavior's base packet (planner/policy are injection-blind).")
    ap.add_argument("--out-dir", default=None,
                    help="override transcript output dir (isolate a run; default: transcripts/)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--thinking", action="store_true",
                    help="enable reasoning on cloud targets (mirrors run_asr --target-thinking). "
                         "GLM delivery parity = thinking-ON (79/88); the preset defaults OFF.")
    ap.add_argument("--no-worldsim", action="store_true",
                    help="disable the arena world-sim fallback for un-canned non-carrier tools "
                         "(keep the '{\"status\":\"ok\"}' stub); world-sim is ON by default.")
    ap.add_argument("--all-worldsim", dest="all_worldsim", action="store_true", default=True,
                    help="fully-simulated env (DEFAULT): answer EVERY non-carrier tool with the "
                         "goal-aware arena world-sim (IGNORE the recorded canned outputs); the "
                         "carrier still returns the injection. Implies world-sim ON. Opt out with "
                         "--no-all-worldsim.")
    ap.add_argument("--no-all-worldsim", dest="all_worldsim", action="store_false",
                    help="restore the recorded-canned replay: canned outputs primary, world-sim only "
                         "as a fallback for un-canned non-carrier tools.")
    ap.add_argument("--forced-delivery", action="store_true",
                    help="per-packet forced-delivery arm: hand the agent the recorded carrier call "
                         "+ args so the injection is delivered organically-but-guaranteed "
                         "(run_pilot.framed_user emits the carrier-args hint).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="planner (agent-under-eval) sampling temperature (default 0.0 = paper "
                         "parity; pass 1.0 for a hot-agent realism arm; judge stays 0.0)")
    args = ap.parse_args()

    if args.all_worldsim and args.no_worldsim:
        print("NB: --all-worldsim overrides --no-worldsim (world-sim ON, fully simulated)", flush=True)
    cfg = R.model_preset(args.model)
    if args.thinking:
        cfg["thinking"] = True
    cfg["temperature"] = args.temperature
    global _TX
    if args.out_dir:
        _TX = Path(args.out_dir)
    _TX.mkdir(parents=True, exist_ok=True)
    # Build the job list: (packet_file, attack_or_None). Multi-string mode expands each behavior's
    # base packet across all its injection strings; else one job per behavior (attack=None).
    if args.strings_json:
        import hashlib
        raw = json.loads(Path(args.strings_json).read_text())
        for old in _TX.glob("*.json"):  # fresh run: clear stale per-string transcripts
            if not old.name.startswith("_"):
                old.unlink()
        jobs = []
        for bid, strings in raw.items():
            fp = _PACKETS / f"{bid}.json"
            if not fp.exists():
                continue
            for i, s in enumerate(strings):
                jobs.append((fp, {"idx": i, "attack_id": f"{bid}#u{i:03d}", "attack_text": s,
                                  "attack_hash": hashlib.md5(s.encode()).hexdigest()}))
    else:
        files = ([_PACKETS / f"{b}.json" for b in args.behaviors] if args.behaviors
                 else sorted(_PACKETS.glob("*.json")))
        jobs = [(fp, None) for fp in files]
    ws_mode = "all" if args.all_worldsim else ("off" if args.no_worldsim else "on")
    print(f"running {len(jobs)} jobs | model={cfg['model']} workers={args.workers} "
          f"worldsim={ws_mode}\n", flush=True)
    _run_one(jobs[0][0], cfg, args.no_worldsim, args.all_worldsim, args.forced_delivery,
             jobs[0][1])  # warm up imports before threading

    summary = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, fp, cfg, args.no_worldsim, args.all_worldsim,
                          args.forced_delivery, attack): fp for fp, attack in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            summary.append(r)
            tag = r["status"] if r["status"] != "ok" else (
                f"BLOCKED({r['blocked_call']})" if r["blocked"] else " ".join(r["calls"][:6]))
            print(f"  deliv={r['delivered']!s:5s} blk={r['blocked']!s:5s} | {r['behavior']:44s} {tag}",
                  flush=True)

    summary.sort(key=lambda r: (r["corpus"], r["behavior"]))
    (_TX / "_coverage_summary.json").write_text(json.dumps(summary, indent=2))

    n = len(summary)
    deliv = sum(r["delivered"] for r in summary)
    blk = sum(r["blocked"] for r in summary)
    blk_deliv = sum(r["blocked"] and r["delivered"] for r in summary)
    errs = sum(r["status"] != "ok" for r in summary)
    print("\n=== ATTACK DELIVERY COVERAGE ===")
    print(f"  delivered (injection reached planner) : {deliv}/{n}  ({100*deliv/n:.1f}%)")
    print(f"  policy fired (FIDES blocked an action): {blk}/{n}   (delivered+blocked: {blk_deliv})")
    print(f"  errors                                : {errs}")
    for corp in sorted({r["corpus"] for r in summary}):
        rows = [r for r in summary if r["corpus"] == corp]
        d = sum(r["delivered"] for r in rows)
        print(f"    {corp:12s} delivered {d}/{len(rows)}")
    nd = [r for r in summary if not r["delivered"] and r["status"] == "ok"]
    print(f"\n  {len(nd)} delivered=False (any block here would be vacuous):")
    for r in nd:
        print(f"    {r['corpus']:12s} {r['behavior']:46s} carrier={r['carrier']} calls={r['calls'][:5]}")
    if errs:
        print("\n  errors:")
        for r in summary:
            if r["status"] != "ok":
                print(f"    {r['behavior']:46s} {r['status']}")


if __name__ == "__main__":
    main()
