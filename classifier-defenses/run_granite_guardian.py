#!/usr/bin/env python3
"""Run the local Granite Guardian 4.1-8b two-head monitor over the store and write its
envelopes to results/<defense_id>.jsonl (one row per attack_id), Cygnal-shaped so validate.py
and the README tables ingest it with no special-casing.

Unlike run_detector.py's per-record loop, this collects EVERY monitored event's guardian prompt
across the whole store, then batch-generates them through one model (sorted by length, left-
padded) for GPU efficiency. Two heads, via the shared events_common._events definition:
  - ipi head    @ each containsIPI tool turn   -> injection-presence
  - policy head @ the isolated harmful action  -> policy-violation (IPI success)

  CUDA_VISIBLE_DEVICES=4 <gpu_venv>/bin/python run_granite_guardian.py            # full store
  CUDA_VISIBLE_DEVICES=4 <gpu_venv>/bin/python run_granite_guardian.py --limit 100  # pilot
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reduce fragmentation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detectors as D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", default="granite-guardian-4.1-8b")
    ap.add_argument("--store", default=os.path.join(HERE, "store", "attacks.jsonl"))
    ap.add_argument("--defenses", default=os.path.join(HERE, "defenses.json"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--limit", type=int, default=None, help="pilot: first N store records")
    ap.add_argument("--batch-size", type=int, default=None, help="override params batch_size")
    ap.add_argument("--nshards", type=int, default=1, help="split the store across N GPUs")
    ap.add_argument("--shard", type=int, default=0, help="this shard's index in [0, nshards)")
    args = ap.parse_args()

    cfg = {d["defense_id"]: d for d in json.load(open(args.defenses))}[args.defense]
    params = cfg["params"]
    policy_only = params.get("heads") == ["policy"]
    batch_size = args.batch_size or params.get("batch_size", 16)

    rows = [json.loads(l) for l in open(args.store) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if args.nshards > 1:
        rows = rows[args.shard::args.nshards]   # interleaved shard; merge by concatenation
    print(f"loading {args.defense} ({params['model_id']}) ...", flush=True)
    bundle = D.load_granite_guardian(params)

    # --- build every event prompt across the store, remembering (record, event) position ---
    t0 = time.time()
    per_record = []          # metas per record (aligned to rows)
    all_prompts, index = [], []  # flat prompts + (record_pos, event_pos)
    for ri, r in enumerate(rows):
        metas, prompts = D.gg_build_event_prompts(r["record"], bundle, policy_only=policy_only)
        per_record.append(metas)
        for ei, p in enumerate(prompts):
            all_prompts.append(p)
            index.append((ri, ei))
    print(f"built {len(all_prompts)} guardian prompts over {len(rows)} records "
          f"({time.time() - t0:.0f}s); scoring (batch={batch_size}) ...", flush=True)

    # --- one batched generation pass ---
    labels, probs, errs = bundle["score_batch"](all_prompts, batch_size=batch_size, progress=2000)

    # --- regroup results back to their records, assemble Cygnal-shaped envelopes ---
    results = [[None] * len(m) for m in per_record]
    for (ri, ei), lab, pr, er in zip(index, labels, probs, errs):
        results[ri][ei] = (lab, pr, er)
    envs = [D.gg_assemble_envelope(per_record[ri], results[ri], params) for ri in range(len(rows))]

    os.makedirs(args.results, exist_ok=True)
    suffix = f".shard{args.shard}of{args.nshards}" if args.nshards > 1 else ""
    out = os.path.join(args.results, f"{args.defense}{suffix}.jsonl")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        for r, env in zip(rows, envs):
            f.write(json.dumps({"attack_id": r["attack_id"], "defense_id": args.defense, **env},
                               ensure_ascii=False) + "\n")
    os.replace(tmp, out)

    flagged = sum(e["flagged_any"] for e in envs)
    ipi = sum(e["summary"]["ipi_flagged"] for e in envs)
    pol = sum(e["summary"]["harmful_action_violation"] for e in envs)
    err = sum(e["summary"]["any_error"] for e in envs)
    print(f"=== {args.defense}: flagged_any {flagged}/{len(rows)} | ipi-head {ipi} | "
          f"policy@harmful {pol} | parse-errors {err} -> {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
