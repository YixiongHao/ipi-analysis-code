#!/usr/bin/env python3
"""Local-GPU LLM judge over the Cygnal monitor prompts.

Same event set, same prompts, and the SAME sidecar schema as the API judges
(`run_llm_judge.py`) so the results drop straight into results_ipi / fp_* and
the plot pipeline. The only difference is transport: local transformers
inference instead of an HTTP API.

Batching: events are sorted by token length and packed into batches with a
token budget, left-padded (decoder-only). This is a large win over per-event
generation because prompt lengths vary by ~100x across the store.

Usage:
  run_local_judge.py --defense llmjudge-qwen3.5-2b --store <records.jsonl> \
      --results <dir> [--gpu 6] [--batch-tokens 60000] [--limit N]

Run with a CUDA venv (the master venv's torch is CPU-only):
  claude-exps/datasentinel-smoke/.venv/bin/python
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_CD = os.path.dirname(_HERE)
sys.path.insert(0, _CD)
sys.path.insert(0, _HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--results", default=os.path.join(_CD, "results_ipi"))
    ap.add_argument("--gpu", default="6")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--heads", default="ipi,policy")
    ap.add_argument("--batch-tokens", type=int, default=60000,
                    help="max padded tokens per batch (batch_size = budget // longest)")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HOME", "~/.cache/huggingface")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import run_llm_judge as rj
    import cygnal_prompt

    entry = rj.load_defense(args.defense)
    if entry.get("defense_type") != "llm_judge_local":
        raise SystemExit(f"{args.defense} is not an llm_judge_local defense")
    params = entry["params"]
    cfg_hash = rj.config_hash(params)
    heads = set(args.heads.split(","))

    tok = AutoTokenizer.from_pretrained(params["model"])
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        params["model"], dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    rows = []
    with open(args.store) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    # flatten every event into a job, tagged with its row for reassembly
    jobs: List[Dict[str, Any]] = []
    per_row_events: Dict[int, List[Dict[str, Any]]] = {}
    for ri, row in enumerate(rows):
        evs, js = rj.build_event_prompts(row["record"], params.get("token_limit"))
        per_row_events[ri] = evs
        for j in js:
            if j["head"] not in heads:
                continue
            j["row_idx"] = ri
            if "prompt" in j:
                # keep the TAIL: the evaluation target is at the end of the prompt.
                # 8/1314 Toolathlon events exceed even a 262k window, so this is not
                # hypothetical — without it those run with out-of-range positions.
                ids = tok(j["prompt"], add_special_tokens=False)["input_ids"]
                budget = params.get("ctx", 262144) - args.max_new_tokens - 64
                if len(ids) > budget:
                    j["truncated"] = True
                    j["prompt"] = tok.decode(ids[-budget:])
                chat = tok.apply_chat_template(
                    [{"role": "user", "content": j["prompt"]}],
                    tokenize=False, add_generation_prompt=True)
                j["chat"] = chat
                j["ntok"] = len(tok(chat, add_special_tokens=False)["input_ids"])
            jobs.append(j)

    runnable = [j for j in jobs if "chat" in j]
    runnable.sort(key=lambda j: j["ntok"])  # length-sorted -> tight padding
    print(f"{len(rows)} records, {len(jobs)} events ({len(runnable)} runnable), "
          f"max prompt {max((j['ntok'] for j in runnable), default=0):,} tok", flush=True)

    verdicts: Dict[int, Dict[str, Any]] = {}
    t0 = time.time()
    done = 0
    i = 0
    while i < len(runnable):
        longest = runnable[i]["ntok"]
        bs = max(1, min(args.max_batch, args.batch_tokens // max(1, longest)))
        batch = runnable[i:i + bs]
        i += len(batch)
        enc = tok([b["chat"] for b in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for b, g in zip(batch, gen):
            text = tok.decode(g, skip_special_tokens=True)
            p = cygnal_prompt.parse_judge_response(text, b["head"])
            if not p["ok"]:  # keep the raw text so failures are auditable offline
                p["raw_head"] = text.strip()[:400]
            verdicts[id(b)] = p
        done += len(batch)
        if done % 200 < len(batch):
            el = time.time() - t0
            print(f"  {done}/{len(runnable)} events ({el:.0f}s, {done/el:.1f} ev/s, "
                  f"eta {(len(runnable)-done)/max(done/el,1e-9)/60:.0f}m)", flush=True)

    # reassemble into the standard envelope, one line per record, store order
    out_rows = []
    by_row: Dict[int, List[Dict[str, Any]]] = {}
    for j in jobs:
        by_row.setdefault(j["row_idx"], []).append(j)
    for ri, row in enumerate(rows):
        evs = per_row_events[ri]
        events_out = []
        for j in by_row.get(ri, []):
            ev = evs[j["event_idx"]]
            parsed = verdicts.get(id(j))
            flagged = parsed["flagged"] if (parsed and parsed["ok"]) else None
            e: Dict[str, Any] = {
                "stage": ev["stage"],
                "agent_message_index": ev["msg_index"],
                "head": j["head"],
                "flagged": bool(flagged) if flagged is not None else False,
                "prob_flagged": None,
            }
            if j["head"] == "ipi":
                e["ipi"] = e["flagged"]
                e["injection_known"] = ev.get("injection_known", False)
            else:
                e["is_harmful_action"] = True
                e["tool_calls"] = ev.get("tool_calls") or []
                e["injection_known"] = ev.get("injection_known", False)
            if j.get("nonstandard_target"):
                e["nonstandard_target"] = True
            if j.get("truncated"):
                e["truncated"] = True
            e["error"] = flagged is None
            if parsed and not parsed.get("ok") and parsed.get("raw_head"):
                e["raw_head"] = parsed["raw_head"]
            if "render_error" in j:
                e["render_error"] = j["render_error"]
            events_out.append(e)
        events_out.sort(key=lambda e: (e["agent_message_index"], e["stage"]))
        out_rows.append(rj.assemble_envelope(
            args.defense, cfg_hash, row, events_out,
            {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0}))

    os.makedirs(args.results, exist_ok=True)
    path = os.path.join(args.results, f"{args.defense}.jsonl")
    with open(path + ".tmp", "w") as f:
        for env in out_rows:
            f.write(json.dumps(env, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    n_ev = sum(len(e["events"]) for e in out_rows)
    n_err = sum(1 for e in out_rows for ev in e["events"] if ev["error"])
    print(f"wrote {path}: {len(out_rows)} records, {n_ev} events, {n_err} errors "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
