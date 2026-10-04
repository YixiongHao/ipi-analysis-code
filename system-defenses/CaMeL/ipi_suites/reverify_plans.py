"""Re-verify replay-delivery of all saved plans with the current harness (camel uv env).

Cheap: replays each saved program list (no re-planning). Updates replay_delivered/replay_executed_calls
in plans/<bid>.json. Use after a replay_programs fix to see the new delivery rate.

    cd system-defenses/CaMeL/ipi_suites
    uv run --project ../camel-prompt-injection python reverify_plans.py [--workers 6] [--trials 3] [--only ...]
"""
from __future__ import annotations
import argparse, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import harness

_HERE = Path(__file__).resolve().parent
_PLANS = _HERE / "plans"
_PACKETS = _HERE / "packets"


def _verify(plan: dict, cfg: dict, trials: int) -> dict:
    bid = plan["behavior_id"]
    pkt = json.loads((_PACKETS / f"{bid}.json").read_text())
    programs = plan.get("programs") or []
    n = 0
    last = []
    for _ in range(trials if programs else 0):
        rep = harness.replay_programs(programs, pkt, cfg, arm="isolation")
        last = [c["name"] for c in rep["executed_calls"]]
        n += int(rep.get("injection_delivered"))
    delivered = n >= max(1, (trials + 1) // 2)  # majority of trials
    plan["replay_delivered"] = delivered
    plan["replay_executed_calls"] = last
    plan["replay_deliver_count"] = f"{n}/{trials}"
    return plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3-flash")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    cfg = harness.model_preset(args.model)
    harness.configure_model(cfg["base_url"], cfg["api_key"])
    only = {b for b in args.only.split(",") if b}
    plans = [json.loads(fp.read_text()) for fp in sorted(_PLANS.glob("*.json"))
             if not only or json.loads(fp.read_text())["behavior_id"] in only]
    print(f"re-verifying {len(plans)} plans | trials={args.trials} workers={args.workers}\n", flush=True)
    done = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_verify, p, cfg, args.trials): p["behavior_id"] for p in plans}
        for fut in as_completed(futs):
            try:
                p = fut.result()
            except Exception as e:
                print(f"  ERR {futs[fut]}: {e}"[:140], flush=True); continue
            (_PLANS / f"{p['behavior_id']}.json").write_text(json.dumps(p, indent=2))
            done.append(p)
            mark = "OK " if p["replay_delivered"] else "no "
            print(f"  {mark}{p['behavior_id']:48s} {p.get('replay_deliver_count')} calls={p['replay_executed_calls'][:6]}", flush=True)
    rd = sum(p["replay_delivered"] for p in done)
    print(f"\n=== replay-delivered: {rd}/{len(done)} ===")
    nd = sorted(p["behavior_id"] for p in done if not p["replay_delivered"])
    print(f"still NOT delivering ({len(nd)}): {nd}")


if __name__ == "__main__":
    main()
