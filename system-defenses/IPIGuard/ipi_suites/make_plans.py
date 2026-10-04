"""Stage-1 driver: capture reusable delivering DAGs for every packet on Gemini.

Runs plan_one.py --save as parallel SUBPROCESSES (each its own process -> the fork's global
`whitelist` is isolated, so parallel is safe; Gemini is remote so concurrency is fine).
Behaviors that fail to deliver 3/3 are written to plans/_failures.json for the goal-loop
subagents (Stage 2). Behaviors in QWEN_NONDELIVERED are skipped unless --all is given.

    cd system-defenses/IPIGuard/ipi_suites
    python make_plans.py            # every packet except QWEN_NONDELIVERED
    python make_plans.py --all      # every packet
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PLANS = _HERE / "plans"
_PACKETS = _HERE.parents[1] / "CaMeL" / "ipi_suites" / "packets"
_PY = "python"

# Public behaviors whose carrier our Qwen run never reached -> routed to Stage-2 subagents.
QWEN_NONDELIVERED = [
    "book-hotel", "launch-crypto-miner", "leak-weights", "free-the-agents",
    "nimbus-llm-exfil", "parser-backdoor", "ssh-key-deployment-injection",
]


def _run_one(bid: str, model: str, k: int, hint: int) -> dict:
    cmd = [_PY, str(_HERE / "plan_one.py"), "--behavior", bid, "--model", model,
           "--k", str(k), "--hint-level", str(hint), "--save"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        out = p.stdout
    except subprocess.TimeoutExpired:
        out = "[subprocess timeout]"
    saved = (_PLANS / f"{bid}.json").exists()
    return {"behavior": bid, "saved": saved, "tail": out.strip().splitlines()[-3:] if out else []}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run every packet (incl. QWEN_NONDELIVERED)")
    ap.add_argument("--model", default="gemini-3-flash")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--hint-level", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    _PLANS.mkdir(exist_ok=True)
    allb = sorted(p.stem for p in _PACKETS.glob("*.json"))
    bids = allb if args.all else [b for b in allb if b not in QWEN_NONDELIVERED]
    print(f"planning {len(bids)} behaviors on {args.model} (k={args.k}, hint={args.hint_level}, "
          f"workers={args.workers})\n", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, b, args.model, args.k, args.hint_level): b for b in bids}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            print(f"[{i}/{len(bids)}] {r['behavior']:46s} {'SAVED' if r['saved'] else 'FAILED'}", flush=True)

    saved = [r["behavior"] for r in results if r["saved"]]
    failed = [r["behavior"] for r in results if not r["saved"]]
    queue = sorted(set(failed) | (set() if args.all else set(QWEN_NONDELIVERED)))
    (_PLANS / "_failures.json").write_text(json.dumps(
        {"failed_in_main": sorted(failed), "qwen_nondelivered": QWEN_NONDELIVERED,
         "subagent_queue": queue}, indent=2))
    print(f"\nsaved {len(saved)}/{len(bids)} | failed {len(failed)}")
    if failed:
        print("  failed:", ", ".join(sorted(failed)))
    print(f"  subagent queue ({len(queue)}): {', '.join(queue)}")
    print("  -> plans/_failures.json")


if __name__ == "__main__":
    main()
