#!/usr/bin/env python3
"""Run ONE detector over the store and write its envelopes to a per-defense
sidecar `results/<defense_id>.jsonl` (one row per attack_id), keyed so it can be
joined back to the store by `attack_id` (see validate.py).

Each detector owns its own sidecar, so detectors run fully independently — no
shared-file rewrite, and two can run concurrently without clobbering each other.
The store (store/attacks.jsonl) is read-only here; the loader is its only writer.
Encoders + DataSentinel run locally; Cygnal needs CYGNAL_API_KEY (skipped if unset).

  # encoders + datasentinel (GPU venv, on a free GPU):
  CUDA_VISIBLE_DEVICES=2 <gpu_venv>/bin/python run_detector.py --defense protectai-v2
  CUDA_VISIBLE_DEVICES=2 <gpu_venv>/bin/python run_detector.py --defense datasentinel-mistral7b
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detectors as D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load_registry(path):
    return {d["defense_id"]: d for d in json.load(open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", required=True)
    ap.add_argument("--store", default=os.path.join(HERE, "store", "attacks.jsonl"))
    ap.add_argument("--defenses", default=os.path.join(HERE, "defenses.json"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    cfg = load_registry(args.defenses)[args.defense]
    dtype, params = cfg["defense_type"], cfg["params"]
    rows = [json.loads(l) for l in open(args.store) if l.strip()]

    if dtype == "encoder_classifier":
        bundle = D.load_classifier(params)
        env_fn = D.classifier_envelope
    elif dtype == "defender_tier2_onnx":
        bundle = D.load_defender(params)
        env_fn = D.defender_envelope
    elif dtype == "kad_detector_llm":
        bundle = D.load_datasentinel(params)
        env_fn = D.datasentinel_envelope
    elif dtype == "granite_guardian_local":
        print(f"SKIP {args.defense}: local two-head LLM guardrail (needs a GPU + batched "
              f"generation). Registered in defenses.json; run via run_granite_guardian.py "
              f"(base) / run_granite_guardian_cot.py (CoT variant).")
        return
    elif dtype == "llm_monitor":
        import asyncio
        api_key = os.getenv("CYGNAL_API_KEY")
        if not api_key:
            print(f"SKIP {args.defense}: CYGNAL_API_KEY unset (remote monitor). "
                  f"Registered in defenses.json; run when the key is available.")
            return
        env_fn = None  # async pass below returns envelopes directly
    else:
        raise SystemExit(f"unknown defense_type {dtype}")

    if env_fn is not None:
        envs = []
        for i, r in enumerate(rows):
            envs.append(env_fn(r["record"], bundle))
            if (i + 1) % 20 == 0:
                print(f"  ...{i + 1}/{len(rows)}")
    else:
        envs = asyncio.run(D.cygnal_run_async(rows, D.load_cygnal(params), api_key))

    os.makedirs(args.results, exist_ok=True)
    out = os.path.join(args.results, f"{args.defense}.jsonl")
    with open(out, "w") as f:
        for r, env in zip(rows, envs):
            f.write(json.dumps({"attack_id": r["attack_id"], "defense_id": args.defense, **env},
                               ensure_ascii=False) + "\n")

    flagged = sum(e["flagged_any"] for e in envs)
    print(f"=== {args.defense}: flagged {flagged}/{len(rows)} trajectories -> wrote {out} ===")


if __name__ == "__main__":
    main()
