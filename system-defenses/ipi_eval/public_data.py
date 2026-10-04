"""Build replay records from the public IPI Arena behavior format (`ipi_arena_os/data/`).

The harness was written against the arena's internal behavior export. This module converts a
public behavior (schema: `ipi_arena_os/data/SCHEMA.md`) into that internal shape and builds one
replay record per (behavior, attack), so `run_eval.py` runs on public data unchanged:

    python -m ipi_eval.public_data --config ../ipi_arena_os/examples/inline_attacks.yaml \\
        --out records.jsonl
    python -m ipi_eval.run_eval --defense baseline --valset records.jsonl ...

`--config` takes any public `ipi-arena-bench` config; only its `behaviors:` section is read
(inline `attack`, `attack_file`, `all_from_file`, `attack_dir`, `hf_dataset`). Its `target` /
`judge` sections are ignored, because models come from `run_eval.py` flags.

Supported categories: `tool` and `coding`. `browser` behaviors need a rendered screenshot and are
out of scope for this harness.

Mapping (public -> internal):
  system_prompt                -> system_prompt_config.content
  tools[{name,description,parameters}] -> tools_config.custom_tool_schema
  criteria[i]                  -> criteria[i] with id "c<i>"
  tool_judge + "tool_call" criteria -> one toolUsageJudge panel entry
      expected_args value      -> case-insensitive exact match
      expected_args {pattern}  -> case-insensitive regex
      check ""/missing         -> correct_parameters_once (the public judge's fallback)
  "llm" criteria               -> one configuredPromptLlmJudge panel entry
A "tool_call" criterion with no `tool_judge` gets no judge, so it cannot pass (the record is never
a break). One public behavior has this (`flashing-lights-seizure`).

The record's prefill is the WHOLE public conversation with the attack filled in. The injected
message (the one that held `{fill}`) is marked by `prefill_end_index`; it is usually the last.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from ipi_arena_bench.behavior import build_messages, load_behaviors
from ipi_arena_bench.config import _resolve_paths, iter_behavior_attacks

from . import behaviors as B
from . import tool_labels

SUPPORTED = {"tool", "coding"}
CORPUS = "ipi_arena_os"


def _expected(v) -> dict:
    if isinstance(v, dict) and "pattern" in v:
        return {"value": v["pattern"], "isRegex": True, "case_insensitive": True}
    return {"value": str(v), "isRegex": False, "case_insensitive": True}


def to_internal(pub: dict) -> dict:
    """Public behavior -> the internal behavior doc that engine.py and judges.py read."""
    criteria = [{"id": f"c{i}", "name": c["name"], "description": c.get("description", ""),
                 "threshold": c.get("threshold", 7)} for i, c in enumerate(pub["criteria"])]
    by_judge: dict[str, dict] = {"tool_call": {}, "llm": {}}
    for c, pc in zip(criteria, pub["criteria"]):
        by_judge[pc["judge"]][c["id"]] = {"enabled": True}

    panel = []
    tj = pub.get("tool_judge")
    if tj and by_judge["tool_call"]:
        expected = {k: _expected(v) for k, v in (tj.get("expected_args") or {}).items()}
        panel.append({"judge_id": "toolUsageJudge", "enabled": True,
                      "criteria_config": by_judge["tool_call"],
                      "parameters": {"target_tool": tj["tool_name"],
                                     "break_condition": tj.get("check") or "correct_parameters_once",
                                     "required_parameters": list(expected),
                                     "expected_parameter_values": expected}})
    if by_judge["llm"]:
        panel.append({"judge_id": "configuredPromptLlmJudge", "enabled": True,
                      "criteria_config": by_judge["llm"]})

    return {
        "id": pub["id"], "name": pub["name"],
        "description": pub.get("description", ""), "summary": pub.get("summary", ""),
        "system_prompt_config": {"content": pub.get("system_prompt", "")},
        "tools_config": {"custom_tool_schema": [
            {"name": t["name"], "description": t.get("description", ""),
             "input_schema": t.get("parameters") or {"type": "object", "properties": {}}}
            for t in pub.get("tools", [])]},
        "criteria": criteria,
        "judging": {"submit": {"panel": panel,
                               "criteria_config": {c["id"]: {"enabled": True} for c in criteria}}},
        "_corpus": CORPUS,
    }


def make_record(pub: dict, attack_text: str) -> dict:
    """One replay record (same fields build_valset.py writes), with the converted behavior inline."""
    if pub["category"] not in SUPPORTED:
        raise ValueError(f"{pub['id']}: category '{pub['category']}' is not supported "
                         f"(supported: {sorted(SUPPORTED)})")
    fill_idx = [i for i, m in enumerate(pub["messages"]) if "{fill}" in json.dumps(m)]
    if not fill_idx:
        raise ValueError(f"{pub['id']}: no {{fill}} placeholder in messages")
    messages = build_messages(pub, attack_text)
    offset = len(messages) - len(pub["messages"])          # 1 when a system message was prepended
    behavior = to_internal(pub)
    attack_hash = hashlib.md5(attack_text.encode("utf-8")).hexdigest()
    return {
        "attack_id": f"{pub['id']}:{attack_hash[:12]}",
        "attack_hash": attack_hash,
        "attack_text": attack_text,
        "behavior_id": pub["id"],
        "corpus": CORPUS,
        "user_request": next((m["content"] for m in pub["messages"]
                              if m["role"] == "user" and isinstance(m.get("content"), str)), ""),
        "system_prompt": pub.get("system_prompt", ""),
        "prefill_messages": messages,
        "prefill_end_index": fill_idx[-1] + offset,
        "tools": B.openai_tools(behavior),
        "tool_policies": tool_labels.tool_policies(pub["id"]),   # FIDES classes from the audit
        "behavior": behavior,
    }


def _behaviors_section(config_path: Path) -> dict:
    """The `behaviors:` section of a public config, with relative attack paths resolved."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return _resolve_paths({"behaviors": config["behaviors"]}, config_path.parent)


def build_records(config_path: Path, data_dir: Path | None = None) -> list[dict]:
    pubs = load_behaviors(data_dir)
    records = []
    for bid, attack in iter_behavior_attacks(_behaviors_section(config_path)):
        if bid not in pubs:
            raise KeyError(f"behavior '{bid}' not found in {data_dir or 'the default data dir'}")
        records.append(make_record(pubs[bid], attack))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, type=Path,
                    help="public ipi-arena-bench config (only `behaviors:` is read)")
    ap.add_argument("--data", type=Path, default=None,
                    help="behavior JSON dir (default: ipi_arena_os/data)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    records = build_records(args.config, args.data)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
