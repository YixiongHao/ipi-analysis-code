#!/usr/bin/env python3
"""Granite-Guardian-with-CoT variant: run GG's policy-violation head on the HARMFUL ACTION, but
with each assistant turn's externally-keyed (`reasoning`-field) chain-of-thought surfaced INLINE
in `content` as a <think>..</think> block. Direct analog of run_cygnal_cot.py (same population,
same inject_cot), so the two CoT experiments are comparable.

Why: GG (like Cygnal) reads standard chat messages and ignores the non-standard `reasoning` key,
so it normally judges the action blind to the model's stated intent. This moves that CoT into
`content` to test whether the policy head fires more once it can see the reasoning. Paired 1:1
against the baseline results/granite-guardian-4.1-8b.jsonl (same records, same harmful turn,
no CoT).

Population: the records whose isHarmfulAction turn carries a non-summarized reasoning-field CoT
(reuses run_cygnal_cot.in_population). Policy head only. Output a new store
results/granite-guardian-4.1-8b-cot.jsonl, same envelope shape + summary.cot_injection.

  CUDA_VISIBLE_DEVICES=5 <gpu_venv>/bin/python run_granite_guardian_cot.py
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reduce fragmentation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detectors as D  # noqa: E402
from run_cygnal_cot import in_population, inject_cot  # noqa: E402  (shared CoT population + injector)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFENSE_ID = "granite-guardian-4.1-8b-cot"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", default=DEFENSE_ID)
    ap.add_argument("--store", default=os.path.join(HERE, "store", "attacks.jsonl"))
    ap.add_argument("--defenses", default=os.path.join(HERE, "defenses.json"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    params = {d["defense_id"]: d for d in json.load(open(args.defenses))}[args.defense]["params"]
    max_chars = params["cot_injection"]["max_chars"]
    batch_size = args.batch_size or params.get("batch_size", 16)

    rows = [r for r in (json.loads(l) for l in open(args.store) if l.strip()) if in_population(r)]
    if args.limit:
        rows = rows[: args.limit]
    if args.nshards > 1:
        rows = rows[args.shard::args.nshards]   # interleaved shard; merge by concatenation
    print(f"population (externally-keyed CoT @ harmful turn): {len(rows)}", flush=True)

    print(f"loading {args.defense} ({params['model_id']}) ...", flush=True)
    bundle = D.load_granite_guardian(params)

    t0 = time.time()
    per_record, infos = [], []          # metas + cot_info per record
    all_prompts, index = [], []
    for ri, r in enumerate(rows):
        rc, info = inject_cot(r["record"], max_chars)   # deep-copies; CoT -> <think> in content
        metas, prompts = D.gg_build_event_prompts(rc, bundle, policy_only=True)
        per_record.append(metas)
        infos.append(info)
        for ei, p in enumerate(prompts):
            all_prompts.append(p)
            index.append((ri, ei))
    print(f"built {len(all_prompts)} policy prompts over {len(rows)} records "
          f"({time.time() - t0:.0f}s); scoring (batch={batch_size}) ...", flush=True)

    labels, probs, errs = bundle["score_batch"](all_prompts, batch_size=batch_size, progress=2000)

    results = [[None] * len(m) for m in per_record]
    for (ri, ei), lab, pr, er in zip(index, labels, probs, errs):
        results[ri][ei] = (lab, pr, er)
    envs = [D.gg_assemble_envelope(per_record[ri], results[ri], params, cot_info=infos[ri])
            for ri in range(len(rows))]

    os.makedirs(args.results, exist_ok=True)
    suffix = f".shard{args.shard}of{args.nshards}" if args.nshards > 1 else ""
    out = os.path.join(args.results, f"{args.defense}{suffix}.jsonl")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        for r, env in zip(rows, envs):
            f.write(json.dumps({"attack_id": r["attack_id"], "defense_id": args.defense, **env},
                               ensure_ascii=False) + "\n")
    os.replace(tmp, out)

    nv = sum(e["summary"]["harmful_action_violation"] for e in envs)
    tr = sum(e["summary"]["cot_injection"]["truncated"] for e in envs)
    err = sum(e["summary"]["any_error"] for e in envs)
    print(f"=== {args.defense}: policy head fired @ action (with CoT): {nv}/{len(envs)} = "
          f"{nv / max(1, len(envs)):.1%} | truncated CoT: {tr} | parse-errors: {err} -> {out} "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
