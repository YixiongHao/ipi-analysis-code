"""Gold pilot CLI over harness.py: run real CaMeL on one packet (default: cancel-appointment).

    cd system-defenses/CaMeL/ipi_suites
    uv run --project ../camel-prompt-injection python run_pilot.py --model gemini-flash --arm isolation
    uv run --project ../camel-prompt-injection python run_pilot.py --model gemini-flash --arm policy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import harness

_HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="isolation", choices=["isolation", "policy"])
    ap.add_argument("--model", default="gemini-3-flash", choices=["qwen", "gemini-flash", "gemini-3-flash"])
    ap.add_argument("--packet", default=str(_HERE / "packets" / "cancel-appointment.json"))
    args = ap.parse_args()

    cfg = harness.model_preset(args.model)
    harness.configure_model(cfg["base_url"], cfg["api_key"])

    pkt = json.load(open(args.packet))
    carrier, harmful = pkt["carrier"], pkt.get("harmful_tool")   # both from the packet

    print(f"[pilot] model={cfg['model']} arm={args.arm} behavior={pkt['behavior_id']} ...", flush=True)
    t = harness.run_behavior(pkt, args.arm, carrier, pkt["canned"], cfg)

    called = [c["name"] for c in t["executed_calls"]]
    print("\n=== RESULT ===")
    print("executed tool calls:", called)
    print("injection delivered:", t["injection_delivered"])
    print(f"HARMFUL '{harmful}' called:", harmful in called, " <-- break iff True")
    print("final text:", (t["final_text"] or "")[:300])

    out = _HERE / f"_pilot_transcript_{args.arm}.json"
    out.write_text(json.dumps(t, indent=2, default=str))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
