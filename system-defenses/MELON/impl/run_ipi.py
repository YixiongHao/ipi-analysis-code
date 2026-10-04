"""MELON verification on recorded IPI rollouts (Phase 5).

Loads a small stratified slice of *originally-successful* attacks via the shared
``classifier-defenses/loader.py``, runs MELON's masked re-execution + tool-call comparison
over each (see ``ipi_adapter.py``), and reports the **flag rate** — of attacks that succeeded
against the real model, how many MELON would have caught.

Run (master venv; Qwen on 8000, BGE on 8001):
    python run_ipi.py --n 20 --corpus q1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
sys.path.insert(0, str(Path("classifier-defenses")))

import loader  # noqa: E402

from agentdojo_adapter import make_bge_embed_fn  # noqa: E402
from ipi_adapter import score_rollout  # noqa: E402
import openai  # noqa: E402

STREAMS = {"q1": loader.stream_q1, "q2": loader.stream_q2, "2025": loader.stream_2025}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=list(STREAMS), default="q1")
    ap.add_argument("--n", type=int, default=20, help="number of rollouts to score")
    ap.add_argument("--per-behavior-cap", type=int, default=2)
    ap.add_argument("--max-scan", type=int, default=4000)
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    picked, stats = loader.stratified(STREAMS[args.corpus], target=args.n,
                                      per_behavior_cap=args.per_behavior_cap, max_scan=args.max_scan)
    print(f"Loaded {len(picked)} successful-attack rollouts across {len(stats)} behaviors "
          f"(corpus={args.corpus}).", flush=True)

    client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    embed_fn = make_bge_embed_fn()

    out_path = RESULTS / f"ipi_{args.corpus}.jsonl"
    results = []
    with open(out_path, "w") as f:
        for i, rec in enumerate(picked):
            try:
                r = score_rollout(rec, client, embed_fn, sim_threshold=args.threshold)
            except Exception as e:  # keep the slice going; record the failure
                r = {"attack_id": rec["attack_id"], "behavior_id": rec["behavior_id"],
                     "corpus": rec["corpus"], "error": repr(e)[:300], "flagged": None}
            results.append(r)
            f.write(json.dumps(r) + "\n")
            mark = "FLAG" if r.get("flagged") else ("ERR" if r.get("flagged") is None else "miss")
            print(f"[{i+1}/{len(picked)}] {mark:4s} sim={r.get('max_sim')} {r['behavior_id']}", flush=True)

    scored = [r for r in results if r.get("flagged") is not None]
    n_flag = sum(1 for r in scored if r["flagged"])
    n_err = sum(1 for r in results if r.get("flagged") is None)
    rate = round(100.0 * n_flag / len(scored), 2) if scored else 0.0
    summary = {"corpus": args.corpus, "threshold": args.threshold, "n_scored": len(scored),
               "n_flagged": n_flag, "n_errors": n_err, "flag_rate_pct": rate}
    (RESULTS / f"ipi_{args.corpus}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nMELON IPI flag rate ({args.corpus}): {n_flag}/{len(scored)} = {rate}%  "
          f"(errors: {n_err})\nWrote {out_path}")


if __name__ == "__main__":
    main()
