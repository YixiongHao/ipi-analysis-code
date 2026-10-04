"""Stage-1 gold pilot: run the REAL fork IPIGuard over ONE behavior, both arms.

    cd system-defenses/IPIGuard/ipi_suites
    python run_pilot.py            # cancel-appointment, qwen
    python run_pilot.py --behavior <bid> --model qwen

Writes transcripts/<bid>.<arm>.json. Grade with grade_batch.py (master venv, separate
process). Verifies: TDG built, injection delivered (carrier traversed), the harmful
command call is FAKED (not executed) under ipiguard, and executed for real under baseline.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import harness` works
import harness  # noqa: E402

_TX = Path(__file__).resolve().parent / "transcripts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", default="cancel-appointment")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--arms", default="baseline,ipiguard")
    ap.add_argument("--timeout", type=int, default=240)
    args = ap.parse_args()

    _TX.mkdir(exist_ok=True)
    cfg = harness.model_preset(args.model)
    packet = harness.load_packet(args.behavior)
    print(f"behavior={args.behavior} corpus={packet.get('corpus')} model={cfg['model']}")
    print(f"carrier={packet['carrier']}  n_tools={len(packet['tools'])}  "
          f"consequential={sorted(packet.get('tool_policies', {}))}\n", flush=True)

    for arm in args.arms.split(","):
        arm = arm.strip()
        print(f"--- arm={arm} ---", flush=True)
        t = harness.run_behavior(packet, arm, cfg, behavior_timeout_s=args.timeout)
        out = _TX / f"{args.behavior}.{arm}.json"
        out.write_text(json.dumps(t, indent=2))
        print(f"  status={t['status']}  delivered={t['injection_delivered']}")
        print(f"  executed (REAL): {[c['name'] for c in t['executed_calls']]}")
        print(f"  faked (FTI):     {[c['name'] for c in t['faked_calls']]}")
        ft = (t.get('final_text') or '')[:200].replace('\n', ' ')
        print(f"  final_text: {ft}")
        print(f"  -> {out}\n", flush=True)


if __name__ == "__main__":
    main()
