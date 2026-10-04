"""FIDES post-hoc verification on recorded IPI rollouts (Phase 5).

Loads a stratified slice of originally-successful attacks via the shared
``classifier-defenses/loader.py`` and applies the FIDES P-T rule post-hoc (see ipi_adapter.py):
of attacks that succeeded against the real model, how many would P-T have blocked (because the
attacker's action was a consequential tool call emitted after the untrusted injected message).

No model server needed — pure trajectory analysis.

Run (master venv):
    python run_ipi.py --corpus q1 --n 40
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
sys.path.insert(0, str(Path("classifier-defenses")))

import loader  # noqa: E402

from defense import Defense  # noqa: E402
from ipi_adapter import analyze_rollout  # noqa: E402

STREAMS = {"q1": loader.stream_q1, "q2": loader.stream_q2, "2025": loader.stream_2025}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=list(STREAMS), default="q1")
    ap.add_argument("--n", type=int, default=40, help="number of rollouts to score")
    ap.add_argument("--per-behavior-cap", type=int, default=3)
    ap.add_argument("--max-scan", type=int, default=6000)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    picked, stats = loader.stratified(STREAMS[args.corpus], target=args.n,
                                      per_behavior_cap=args.per_behavior_cap, max_scan=args.max_scan)
    print(f"Loaded {len(picked)} successful-attack rollouts across {len(stats)} behaviors "
          f"(corpus={args.corpus}).", flush=True)

    defense = Defense()
    out_path = RESULTS / f"ipi_{args.corpus}.jsonl"
    results = []
    with open(out_path, "w") as f:
        for i, rec in enumerate(picked):
            try:
                r = analyze_rollout(rec, defense)
            except Exception as e:
                r = {"attack_id": rec["attack_id"], "behavior_id": rec["behavior_id"],
                     "corpus": rec["corpus"], "error": repr(e)[:300], "would_block": None}
            results.append(r)
            f.write(json.dumps(r) + "\n")
            mark = "BLOCK" if r.get("would_block") else ("ERR" if r.get("would_block") is None else "pass")
            print(f"[{i+1}/{len(picked)}] {mark:5s} [{r.get('attack_type','?')}] {r['behavior_id']}", flush=True)

    scored = [r for r in results if r.get("would_block") is not None]
    n_block = sum(1 for r in scored if r["would_block"])
    n_err = sum(1 for r in results if r.get("would_block") is None)
    types = Counter(r["attack_type"] for r in scored)
    rate = round(100.0 * n_block / len(scored), 2) if scored else 0.0
    summary = {
        "corpus": args.corpus, "n_scored": len(scored), "n_would_block": n_block,
        "n_errors": n_err, "would_block_rate_pct": rate, "attack_type_breakdown": dict(types),
    }
    (RESULTS / f"ipi_{args.corpus}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFIDES P-T would-block rate ({args.corpus}): {n_block}/{len(scored)} = {rate}%  (errors: {n_err})")
    print(f"attack types: {dict(types)}\nWrote {out_path}")


if __name__ == "__main__":
    main()
