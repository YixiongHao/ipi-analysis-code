#!/usr/bin/env python3
"""Sharded/parallel driver for the stackone-defender-tier2 detector.

The defense is pure-CPU ONNX; `run_detector.py` runs it single-process (4 intra-op
threads), which is fine for the 50-record FP set but slow on the 26k-record store
(~222M chars to chunk-score). This driver processes records across a ProcessPool —
each worker loads its OWN DefenderTier2 (ONNX session is per-process) and scores a
disjoint slice, so the per-record output is byte-identical to the serial path
(scoring is per-record independent; no cross-record batching). Writes the same
`results/<defense>.jsonl` sidecar keyed by attack_id.

  python run_defender_parallel.py \
      --store store/attacks.jsonl --results results_ipi --workers 16 --threads 4
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFENSE_ID = "stackone-defender-tier2"
_bundle = None  # per-worker


def _init(params, threads):
    global _bundle
    import detectors as D
    p = dict(params)
    p["intra_op_threads"] = threads
    _bundle = D.load_defender(p)


def _score(args):
    import detectors as D
    idx, attack_id, record = args
    env = D.defender_envelope(record, _bundle)
    return idx, attack_id, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=os.path.join(HERE, "store", "attacks.jsonl"))
    ap.add_argument("--results", default=os.path.join(HERE, "results_ipi"))
    ap.add_argument("--defenses", default=os.path.join(HERE, "defenses.json"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4, help="intra-op ONNX threads per worker")
    ap.add_argument("--chunksize", type=int, default=64)
    args = ap.parse_args()

    params = {d["defense_id"]: d for d in json.load(open(args.defenses))}[DEFENSE_ID]["params"]
    rows = [json.loads(l) for l in open(args.store) if l.strip()]
    n = len(rows)
    print(f"[defender-parallel] {n} records, {args.workers} workers x {args.threads} threads "
          f"(<= {args.workers*args.threads} cores)")

    tasks = [(i, r["attack_id"], r["record"]) for i, r in enumerate(rows)]
    results = [None] * n
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(params, args.threads)) as ex:
        for idx, attack_id, env in ex.map(_score, tasks, chunksize=args.chunksize):
            results[idx] = (attack_id, env)
            done += 1
            if done % 2000 == 0 or done == n:
                el = time.time() - t0
                print(f"  ...{done}/{n}  ({done/el:.0f} rec/s, {el:.0f}s elapsed)", flush=True)

    os.makedirs(args.results, exist_ok=True)
    out = os.path.join(args.results, f"{DEFENSE_ID}.jsonl")
    flagged = 0
    with open(out, "w") as f:
        for attack_id, env in results:
            flagged += int(bool(env["flagged_any"]))
            f.write(json.dumps({"attack_id": attack_id, "defense_id": DEFENSE_ID, **env},
                               ensure_ascii=False) + "\n")
    print(f"=== {DEFENSE_ID}: flagged {flagged}/{n} trajectories -> {out} "
          f"({time.time()-t0:.0f}s total) ===")


if __name__ == "__main__":
    main()
