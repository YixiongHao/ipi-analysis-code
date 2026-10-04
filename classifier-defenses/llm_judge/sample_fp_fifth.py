#!/usr/bin/env python3
"""Build 1/5 subsamples of the two FP testbeds for LLM-judge full runs.

- fp_rebench swe_rebench_clean50 -> 10 trajectories (seed-0 shuffle)
- fp_toolathlon toolathlon_verified_108 -> ~22 records (seed-0, stratified by
  source model to preserve the 5-model pool balance)

Outputs *_fifth_seed0.jsonl next to the source records + a MANIFEST line.
"""
import json
import os
import random
from collections import defaultdict

_CD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = [
    (os.path.join(_CD, "fp_rebench", "records", "swe_rebench_clean50.jsonl"), None, 10),
    (os.path.join(_CD, "fp_toolathlon", "records", "toolathlon_verified_108.jsonl"), "model", 22),
]


def main():
    for src, stratum_key, target in JOBS:
        rows = [json.loads(l) for l in open(src)]
        strata = defaultdict(list)
        for i, r in enumerate(rows):
            strata[r.get(stratum_key) if stratum_key else None].append(i)
        rng = random.Random(0)
        keep = set()
        for key in sorted(strata, key=str):
            idxs = list(strata[key])
            rng.shuffle(idxs)
            keep.update(idxs[: max(1, round(len(idxs) * target / len(rows)))])
        out = src.replace(".jsonl", "_fifth_seed0.jsonl")
        with open(out + ".tmp", "w") as f:
            for i, r in enumerate(rows):
                if i in keep:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(out + ".tmp", out)
        print(f"{os.path.basename(out)}: {len(keep)}/{len(rows)} records "
              f"(seed 0, strata={stratum_key})")


if __name__ == "__main__":
    main()
