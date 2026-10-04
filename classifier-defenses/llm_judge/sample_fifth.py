#!/usr/bin/env python3
"""Build the 1/5 store subsample for LLM-judge full runs.

Stratified by (corpus, behavior_id): within each stratum, seed-0 shuffle and
keep round(n/5) records (min 1). Output preserves store order, so sidecars
written against it stay aligned with store-order conventions.

Writes store/fifth_seed0/attacks.jsonl + fifth_seed0/MANIFEST.json.
"""
import argparse
import json
import os
import random
from collections import defaultdict

_CD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(_CD, "store", "attacks.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=int, default=5, help="keep 1/FRACTION per stratum")
    ap.add_argument("--target-n", type=int, default=None,
                    help="instead of --fraction: total record target, allocated ∝ stratum size")
    ap.add_argument("--out-name", default="fifth_seed0", help="subdir under store/")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = os.path.join(_CD, "store", args.out_name)

    strata = defaultdict(list)  # (corpus, behavior_id) -> [line_no]
    ids = []
    with open(SRC) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            strata[(row.get("corpus"), row.get("behavior_id"))].append(i)
            ids.append(row["attack_id"])

    keep = set()
    rng = random.Random(args.seed)
    for key in sorted(strata, key=lambda k: (str(k[0]), str(k[1]))):
        lines = list(strata[key])
        rng.shuffle(lines)
        if args.target_n is not None:
            n_keep = max(1, round(len(lines) * args.target_n / len(ids)))
        else:
            n_keep = max(1, round(len(lines) / args.fraction))
        keep.update(lines[:n_keep])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "attacks.jsonl")
    tmp = out_path + ".tmp"
    n = 0
    with open(SRC) as f, open(tmp, "w") as out:
        for i, line in enumerate(f):
            if i in keep:
                out.write(line)
                n += 1
    os.replace(tmp, out_path)

    manifest = {
        "source": "store/attacks.jsonl",
        "fraction": None if args.target_n is not None else f"1/{args.fraction}",
        "target_n": args.target_n,
        "seed": args.seed,
        "strata": "(corpus, behavior_id)",
        "rule": "per-stratum seeded shuffle, proportional keep (min 1), output in store order",
        "n_source": len(ids),
        "n_kept": n,
        "n_strata": len(strata),
    }
    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
