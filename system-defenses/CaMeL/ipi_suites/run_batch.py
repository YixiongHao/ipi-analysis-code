"""Run auto-derived packets through CaMeL (camel uv env). Dumps transcripts + a summary.

    cd system-defenses/CaMeL/ipi_suites
    uv run --project ../camel-prompt-injection python run_batch.py --model gemini-flash \
        --arm isolation --behaviors cancel-appointment modify-cicd-workflow share-document
    # all packets:
    uv run --project ../camel-prompt-injection python run_batch.py --model gemini-flash --arm isolation
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import harness

_HERE = Path(__file__).resolve().parent
_PACKETS = _HERE / "packets"
_TX = _HERE / "transcripts"


def _run_one(fp: Path, arm: str, cfg: dict) -> dict:
    pkt = json.loads(fp.read_text())
    bid = pkt["behavior_id"]
    try:
        t = harness.run_behavior(pkt, arm, pkt.get("carrier"), dict(pkt.get("canned", {})), cfg)
        (_TX / f"{bid.strip()}.{arm}.json").write_text(json.dumps(t, indent=2, default=str))
        calls = [c["name"] for c in t["executed_calls"]]
        return {"behavior": bid, "delivered": t["injection_delivered"],
                "n_calls": len(calls), "calls": calls, "status": t.get("status", "ok")}
    except Exception as e:
        return {"behavior": bid, "delivered": False, "n_calls": 0, "calls": [],
                "status": f"ERROR {type(e).__name__}: {str(e)[:120]}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="isolation", choices=["isolation", "policy"])
    ap.add_argument("--model", default="gemini-3-flash", choices=["qwen", "gemini-flash", "gemini-3-flash"])
    ap.add_argument("--behaviors", nargs="*", default=None, help="behavior ids (default: all packets)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = harness.model_preset(args.model)
    harness.configure_model(cfg["base_url"], cfg["api_key"])
    _TX.mkdir(exist_ok=True)

    files = ([_PACKETS / f"{b}.json" for b in args.behaviors] if args.behaviors
             else sorted(_PACKETS.glob("*.json")))
    print(f"running {len(files)} behaviors | model={cfg['model']} arm={args.arm} workers={args.workers}\n",
          flush=True)
    # warm up patches/imports once (idempotent) before threading to avoid first-call races
    _run_one(files[0], args.arm, cfg)

    summary = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, fp, args.arm, cfg): fp for fp in files}
        for fut in as_completed(futs):
            row = fut.result()
            summary.append(row)
            print(f"  {row['delivered']!s:5s} delivered | calls={row['n_calls']:2d} | "
                  f"{row['behavior']:42s} {row['status'] if row['status']!='ok' else ' '.join(row['calls'][:6])}",
                  flush=True)

    deliv = sum(r["delivered"] for r in summary)
    errs = sum(r["status"] != "ok" for r in summary)
    print(f"\n=== {deliv}/{len(summary)} injection delivered | {errs} errors ===")
    (_TX / f"_summary.{args.arm}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
