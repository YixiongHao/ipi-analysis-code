"""Build a stratified ~1k validation set of single-injection IPI breaks.

Each output record is self-contained for replay: system prompt + prefill conversation up
to (and including) the injected tool message, plus a marker (prefill_end_index) telling the
engine where the defended rollout should take over. Structured tools / criteria / judges are
resolved at run time from behaviors.py by behavior_id (single source of truth).

Stratify across corpus x behavior_id (round-robin so every cell is represented evenly, all 3
corpora included), deduping by attack_hash within a cell for attack diversity.

Usage:
  python -m ipi_eval.build_valset --out valset/ipi_defense_valset.jsonl --target 1000
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from . import behaviors as B
from . import tool_resolver
from .judges import has_judge

ATTACKS = B._REPO / "classifier-defenses" / "store" / "attacks.jsonl"


def _injection_index(agent_messages: list[dict]) -> int | None:
    for i, m in enumerate(agent_messages):
        if m.get("isIndirectPromptToolMessage"):
            return i
    return None


def _candidate(rec: dict) -> dict | None:
    bid = (rec.get("behavior_id") or "").strip()
    behavior = B.get_behavior(bid)
    if behavior is None or not has_judge(behavior):
        return None
    am = (rec.get("record") or {}).get("agent_messages") or []
    inj = _injection_index(am)
    if inj is None:
        return None
    sysp = (rec.get("record") or {}).get("system_prompt") or B.system_prompt(behavior)
    if not sysp:
        return None
    prefill = [{"role": "system", "content": sysp}] + am[: inj + 1]
    # Resolve the genuine tool list from the FULL recorded trajectory (declared schema merged
    # with parameter shapes observed in the actual tool calls). Persisted so every rollout /
    # defense sees an identical, non-empty tool list (see tool_resolver.resolve_tools).
    tools, tool_policies = tool_resolver.resolve_tools(behavior, trajectory_messages=am)
    return {
        "attack_id": rec.get("attack_id"),
        "behavior_id": bid,
        "corpus": rec.get("corpus"),
        "attack_hash": rec.get("attack_hash"),
        "attack_text": rec.get("attack_text"),
        "orig_model_name": rec.get("model_name"),
        "user_request": (rec.get("record") or {}).get("user_request") or "",
        "system_prompt": sysp,
        "prefill_messages": prefill,
        # marker: index of the last prefill message (the injected tool msg) within prefill_messages
        "prefill_end_index": len(prefill) - 1,
        "rollout_starts_at": len(prefill),
        "tools": tools,
        "tool_policies": tool_policies,
    }


def build(target: int, seed: int) -> list[dict]:
    # cell key = (corpus, behavior_id); dedup by attack_hash within a cell
    cells: dict[tuple, dict[str, dict]] = defaultdict(dict)
    with open(ATTACKS) as f:
        for line in f:
            rec = json.loads(line)
            cand = _candidate(rec)
            if cand is None:
                continue
            key = (cand["corpus"], cand["behavior_id"])
            ahash = cand["attack_hash"] or cand["attack_id"]
            cells[key].setdefault(ahash, cand)

    rng = random.Random(seed)
    pools = {}
    for key, by_hash in cells.items():
        items = list(by_hash.values())
        rng.shuffle(items)
        pools[key] = items

    # round-robin across cells until target reached or pools exhausted
    keys = sorted(pools)
    rng.shuffle(keys)
    out: list[dict] = []
    progressed = True
    while len(out) < target and progressed:
        progressed = False
        for key in keys:
            if pools[key]:
                out.append(pools[key].pop())
                progressed = True
                if len(out) >= target:
                    break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "valset" / "ipi_defense_valset.jsonl"))
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = build(args.target, args.seed)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # summary
    by_corpus = defaultdict(int)
    by_behavior = defaultdict(int)
    for r in records:
        by_corpus[r["corpus"]] += 1
        by_behavior[r["behavior_id"]] += 1
    print(f"wrote {len(records)} records to {outp}")
    print("by corpus:", dict(by_corpus))
    print("distinct behaviors:", len(by_behavior))
    print("min/max per behavior:", min(by_behavior.values()), "/", max(by_behavior.values()))


if __name__ == "__main__":
    main()
