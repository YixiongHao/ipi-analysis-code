"""CausalArmor post-hoc verification on recorded IPI rollouts (Phase 5).

Loads a stratified slice of originally-successful attacks via classifier-defenses/loader.py,
runs CausalArmor's dominance-shift detector at the malicious privileged decision, and reports
the **flag rate** (= fraction of real successful attacks CausalArmor would have caught /
sanitized) plus coverage.

Run (master venv; Qwen3-32B vLLM on port 8000):
    python run_ipi.py --corpus ipi_2026_q1 --target 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "classifier-defenses"))

import loader  # noqa: E402

from defense import Defense  # noqa: E402
from ipi_adapter import analyze_rollout  # noqa: E402

RESULTS = HERE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="ipi_2026_q1", choices=list(loader.CORPORA))
    ap.add_argument("--target", type=int, default=30, help="number of rollouts in the slice")
    ap.add_argument("--per-behavior-cap", type=int, default=3)
    ap.add_argument("--max-scan", type=int, default=4000)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--areas", default=None, help="comma-separated area filter (e.g. 'Tool-Use')")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    keep_areas = loader._areas(args.areas) if args.areas else None
    slice_, buckets = loader.stratified(
        loader.CORPORA[args.corpus], args.target, args.per_behavior_cap, args.max_scan, keep_areas=keep_areas
    )
    print(f"Loaded {len(slice_)} rollouts from {args.corpus} across {len(buckets)} behaviors", flush=True)

    defense = Defense(tau=args.tau)
    events, counts = [], {"flagged": 0, "not_flagged": 0, "no_decision": 0}
    for i, rec in enumerate(slice_):
        r = rec["record"]
        try:
            res = analyze_rollout(r, defense)
        except Exception as e:  # keep the slice going; record the failure
            res = {"status": "no_decision", "reason": f"error:{type(e).__name__}:{str(e)[:120]}"}
        counts[res["status"]] += 1
        events.append({"attack_id": rec["attack_id"], "behavior_id": rec["behavior_id"],
                       "model_name": rec["model_name"], **res})
        print(f"[{i+1}/{len(slice_)}] {rec['behavior_id']}: {res['status']}"
              + (f" (score={res.get('score')})" if "score" in res else f" ({res.get('reason','')})"), flush=True)

    n_decision = counts["flagged"] + counts["not_flagged"]
    summary = {
        "corpus": args.corpus, "tau": args.tau, "n_total": len(slice_),
        "n_with_decision": n_decision,
        "coverage": round(n_decision / len(slice_), 3) if slice_ else 0.0,
        "n_flagged": counts["flagged"],
        "flag_rate_of_decisions": round(counts["flagged"] / n_decision, 3) if n_decision else 0.0,
        "flag_rate_of_total": round(counts["flagged"] / len(slice_), 3) if slice_ else 0.0,
        "counts": counts,
    }
    envelope = {"defense": defense.name, "flagged_any": counts["flagged"] > 0,
                "summary": summary, "events": events}
    tau_tag = f"_tau{args.tau}" if args.tau != 0.0 else ""
    out_path = RESULTS / f"ipi_{args.corpus}{tau_tag}.json"
    out_path.write_text(json.dumps(envelope, indent=2))
    print(f"\nSummary: {json.dumps(summary, indent=2)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
