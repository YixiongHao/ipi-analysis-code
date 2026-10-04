"""Capture (and optionally save) a reusable delivering DAG for ONE behavior.

The single unit both the main agent and the goal-loop subagents call. Construct the DAG
k times on the chosen model; "delivers" ⟺ carrier ∈ DAG (3/3). On all-deliver, run one
end-to-end traversal (confirm) and, with --save, write plans/<bid>.json. On failure, print
the captured DAG + planned tools so a subagent can escalate --hint-level.

    cd system-defenses/IPIGuard/ipi_suites
    PY=python
    $PY plan_one.py --behavior cancel-appointment --model gemini-3-flash
    $PY plan_one.py --behavior book-hotel --model gemini-3-flash --hint-level 2 --save
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

_PLANS = Path(__file__).resolve().parent / "plans"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", required=True)
    ap.add_argument("--model", default="gemini-3-flash")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--hint-level", type=int, default=0)
    ap.add_argument("--no-confirm", action="store_true", help="skip end-to-end traversal confirm")
    ap.add_argument("--save", action="store_true", help="write plans/<bid>.json if it delivers")
    args = ap.parse_args()

    _PLANS.mkdir(exist_ok=True)
    cfg = harness.model_preset(args.model)
    packet = harness.load_packet(args.behavior)
    res = harness.plan_behavior(packet, cfg, k=args.k, hint_level=args.hint_level)
    print(f"{args.behavior}: k_delivered={res['k_delivered']} carrier={res['carrier']} hint={args.hint_level}")
    print(f"  planned_tools={res['planned_tools']}")
    if res["errors"]:
        print(f"  errors={res['errors']}")

    delivered = None
    if res["all_delivered"] and not args.no_confirm:
        t = harness.confirm_traversal(packet, res["dag"], cfg)
        delivered = t["injection_delivered"]
        print(f"  confirm: status={t['status']} delivered={delivered} "
              f"exec={[c['name'] for c in t['executed_calls']]} faked={[c['name'] for c in t['faked_calls']]}")

    ok_to_save = res["all_delivered"] and (delivered is None or delivered)
    if args.save and ok_to_save:
        out = {"behavior_id": args.behavior, "corpus": res["corpus"], "carrier": res["carrier"],
               "model": cfg["model"], "k_delivered": res["k_delivered"], "hint_level": args.hint_level,
               "delivered": (bool(delivered) if delivered is not None else None),
               "planned_tools": res["planned_tools"], "dag": res["dag"]}
        (_PLANS / f"{args.behavior}.json").write_text(json.dumps(out, indent=2))
        print(f"  SAVED plans/{args.behavior}.json")
    elif args.save:
        print(f"  NOT SAVED (all_delivered={res['all_delivered']}, confirm_delivered={delivered})")


if __name__ == "__main__":
    main()
