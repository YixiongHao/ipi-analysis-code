"""Grade a pilot CaMeL transcript with the ipi_eval Arena judge (MASTER venv).

Closes the second half of the spine: CaMeL run (camel env) dumps a transcript of executed
tool calls; here we convert it to ipi_eval's generated-messages format and score it with the
ported Arena criteriaScore panel. is_break=False == attack defended.

    cd system-defenses
    python -m CaMeL.ipi_suites.grade_pilot --arm isolation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ipi_eval import behaviors as B
from ipi_eval import engine
from ipi_eval import judges

_HERE = Path(__file__).resolve().parent


def to_generated(calls: list[dict], final_text: str | None) -> list[dict]:
    msgs = []
    for i, c in enumerate(calls):
        msgs.append({
            "role": "assistant",
            "tool_calls": [{
                "id": f"call_{i}", "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c.get("args", {}))},
            }],
        })
    if final_text:
        msgs.append({"role": "assistant", "content": final_text})
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="isolation")
    args = ap.parse_args()

    t = json.load(open(_HERE / f"_pilot_transcript_{args.arm}.json"))
    behavior = B.get_behavior(t["behavior_id"])
    gen = to_generated(t["executed_calls"], t.get("final_text"))
    judge_chat = engine.make_judge_chat()  # shared Gemini-3-Flash OpenRouter judge

    verdict = judges.score_behavior(behavior, gen, judge_chat, full_messages=gen)
    print(f"\n=== JUDGE ({t['behavior_id']}, arm={args.arm}) ===")
    print("is_break:", verdict["is_break"], " status:", verdict["status"])
    for c in verdict["criteria"]:
        print(f"  [{c['name']}] score={c['score']} thr={c['threshold']} pass={c['pass']} "
              f"judges={c.get('judges')}")


if __name__ == "__main__":
    main()
