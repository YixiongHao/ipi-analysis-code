"""Static-plan builder: plan ONCE per behavior, SAVE the program, verify it by replay.

The planner's program depends ONLY on the trusted request + tools (CaMeL isolation never sees
the injection), so one delivering program per behavior is valid for ALL its attack strings. We
plan once (expensive P-LLM codegen), then REPLAY the saved program to confirm it actually reaches
the carrier (plan-time "delivered" is unreliable — delivery can hinge on a flaky Q-LLM extraction
that doesn't reproduce). Saved plans feed run_plan_eval.py (cheap per-attack replay, no codegen).

Runs in the reference repo's uv env:
    cd system-defenses/CaMeL/ipi_suites
    uv run --project ../camel-prompt-injection python build_plans.py [--model gemini-3-flash] \
        [--workers 6] [--temps 0.0,0.5,0.8] [--force] [--only bid1,bid2]
Writes plans/<bid>.json. Resumable: skips behaviors with an existing plan unless --force.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import harness

_HERE = Path(__file__).resolve().parent
_PACKETS = _HERE / "packets"
_PLANS = _HERE / "plans"


def _plan_one(pkt: dict, model_cfg: dict, temps: tuple, verify_trials: int) -> dict:
    """Plan (P-LLM codegen, isolation) then verify by replaying the saved program list."""
    bid = pkt["behavior_id"]
    plan = harness.plan_behavior(pkt, model_cfg, arm="isolation", temps=temps)
    programs = plan.get("programs") or []
    replay_delivered = False
    replay_calls: list[str] = []
    for _ in range(verify_trials if programs else 0):
        rep = harness.replay_programs(programs, pkt, model_cfg, arm="isolation")
        replay_calls = [c["name"] for c in rep["executed_calls"]]
        if rep.get("injection_delivered"):
            replay_delivered = True
            break
    return {
        "behavior_id": bid, "corpus": pkt.get("corpus"), "carrier": plan["carrier"],
        "model": model_cfg["model"], "temp": plan["temp"], "n_frags": len(programs),
        "programs": programs, "program": plan.get("program"),
        "plan_delivered": plan["delivered"],
        "plan_executed_calls": [c["name"] for c in plan["executed_calls"]],
        "replay_delivered": replay_delivered, "replay_executed_calls": replay_calls,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3-flash")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--temps", default="0.0,0.5,0.8")
    ap.add_argument("--verify-trials", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    cfg = harness.model_preset(args.model)
    harness.configure_model(cfg["base_url"], cfg["api_key"])
    temps = tuple(float(x) for x in args.temps.split(","))
    _PLANS.mkdir(exist_ok=True)

    only = {b for b in args.only.split(",") if b}
    pkts = []
    for fp in sorted(_PACKETS.glob("*.json")):
        if fp.name.startswith("_"):
            continue
        bid = fp.stem
        if only and bid not in only:
            continue
        if (_PLANS / f"{bid}.json").exists() and not args.force:
            continue
        pkts.append(json.loads(fp.read_text()))
    print(f"planning {len(pkts)} behaviors | model={cfg['model']} temps={temps} workers={args.workers}\n",
          flush=True)

    done = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_plan_one, p, cfg, temps, args.verify_trials): p["behavior_id"] for p in pkts}
        for fut in as_completed(futs):
            bid = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                print(f"  ERROR {bid}: {e}"[:160], flush=True)
                continue
            (_PLANS / f"{bid}.json").write_text(json.dumps(rec, indent=2))
            done.append(rec)
            print(f"  {bid:50s} plan_deliv={rec['plan_delivered']!s:5s} "
                  f"replay_deliv={rec['replay_delivered']!s:5s} nfrags={rec['n_frags']} "
                  f"temp={rec['temp']}", flush=True)

    pd = sum(r["plan_delivered"] for r in done)
    rd = sum(r["replay_delivered"] for r in done)
    print(f"\n=== built {len(done)} plans ===")
    print(f"  plan-time delivered : {pd}/{len(done)}")
    print(f"  replay  delivered   : {rd}/{len(done)}  (the static-plan delivery rate)")


if __name__ == "__main__":
    main()
