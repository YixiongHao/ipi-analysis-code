"""Phase 2 gate: validate the ported judges against ground truth.

Every record in attacks.jsonl is a labeled SUCCESS (grade_status == "success" == is_break).
So running our ported panel judge on the ORIGINAL recorded trajectory (not a re-rollout)
should reproduce is_break == True at ~100%. Anything far below indicates a port bug.

Judges the model's recorded continuation after the injection point:
  generated_messages = agent_messages[inj+1:]   (the original model's authored turn(s))
  full_messages      = agent_messages           (for the algorithmic tool judge)

Usage:
  python -m ipi_eval.validate_judge --per-behavior 3 --workers 8
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import behaviors as B
from . import engine
from . import judges

ATTACKS = B._REPO / "classifier-defenses" / "store" / "attacks.jsonl"


def _sample(per_behavior: int, seed: int) -> list[dict]:
    by_beh: dict[str, list[dict]] = defaultdict(list)
    with open(ATTACKS) as f:
        for line in f:
            rec = json.loads(line)
            bid = (rec.get("behavior_id") or "").strip()
            beh = B.get_behavior(bid)
            if beh is None or not judges.has_judge(beh):
                continue
            am = (rec.get("record") or {}).get("agent_messages") or []
            if not any(m.get("isIndirectPromptToolMessage") for m in am):
                continue
            if len(by_beh[bid]) < per_behavior * 4:  # collect a few, sample below
                by_beh[bid].append(rec)
    rng = random.Random(seed)
    out = []
    for bid, recs in by_beh.items():
        rng.shuffle(recs)
        out.extend(recs[:per_behavior])
    return out


def _judge_recorded(rec: dict, judge_chat) -> dict:
    bid = (rec.get("behavior_id") or "").strip()
    behavior = B.get_behavior(bid)
    am = rec["record"]["agent_messages"]
    inj = next(i for i, m in enumerate(am) if m.get("isIndirectPromptToolMessage"))
    generated = am[inj + 1:]
    try:
        v = judges.score_behavior(behavior, generated, judge_chat, full_messages=am)
        return {"behavior_id": bid, "corpus": rec.get("corpus"),
                "is_break": v["is_break"], "criteria": v["criteria"], "status": v["status"]}
    except Exception as e:
        return {"behavior_id": bid, "corpus": rec.get("corpus"), "is_break": False,
                "status": "error", "error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-behavior", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-model", default="google/gemini-3-flash-preview")
    ap.add_argument("--judge-base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "judge_validation.jsonl"))
    args = ap.parse_args()

    judge_chat = engine.make_judge_chat(engine.make_judge_client(args.judge_model, args.judge_base_url))
    sample = _sample(args.per_behavior, args.seed)
    print(f"validating judge on {len(sample)} recorded successes ...")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_judge_recorded, r, judge_chat) for r in sample]
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 50 == 0:
                print(f"  {i}/{len(sample)} ...", flush=True)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n = len(results)
    rep = sum(r["is_break"] for r in results)
    print(f"\nREPRODUCED break (is_break=True) on {rep}/{n} = {rep/n:.1%}  (target ~100%)")
    # behaviors where we most often FAIL to reproduce -> likely judge-port gaps
    miss = defaultdict(lambda: [0, 0])
    for r in results:
        miss[r["behavior_id"]][0] += int(not r["is_break"])
        miss[r["behavior_id"]][1] += 1
    worst = sorted(((b, m[0], m[1]) for b, m in miss.items() if m[0] > 0),
                   key=lambda x: -x[1])[:20]
    if worst:
        print("behaviors with non-reproductions (missed/total):")
        for b, mi, tot in worst:
            print(f"  {b:40s} {mi}/{tot}")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
