#!/usr/bin/env python3
"""Validate a LOCAL small model as a Cygnal-prompt judge before committing to a run.

Answers three questions, in order:
  1. Does it emit parseable JSON in the required schema at all?
  2. Does that survive on prompts that must be TRUNCATED to fit its context?
  3. Are the verdicts non-degenerate (not all-true / all-false)?

Truncation strategy (documented deviation, mirrors the repo's Granite handling):
left-truncate the rendered prompt to `ctx - reserve` tokens, keeping the TAIL —
the evaluation target lives at the end, so the tail is the part that must
survive. The head (task instructions + early context) is what gets dropped.
With a 262k-context model this should never fire; the counter proves it.

Usage:
  validate_local.py --model Qwen/Qwen3.5-2B --gpu 4 --n 40

Thinking: Qwen3.5's chat template emits a pre-closed `<think>\n\n</think>` block
unless `enable_thinking=True`, so the default path here is NON-thinking — which
matches the API judges' effort-none arms.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CD = os.path.dirname(_HERE)
sys.path.insert(0, _CD)
sys.path.insert(0, _HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--n", type=int, default=40, help="prompts per head")
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--reserve", type=int, default=96, help="tokens kept for the answer")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--store", default=os.path.join(_CD, "store", "fifth_seed0", "attacks.jsonl"))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HOME", "~/.cache/huggingface")

    import random
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import run_llm_judge as rj
    import cygnal_prompt

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda:0")
    except (ValueError, KeyError) as e:  # multimodal wrapper (Qwen3_5ForConditionalGeneration)
        print(f"[causal-lm load failed: {e}; trying image-text-to-text head]")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(args.store)]
    random.Random(0).shuffle(rows)

    jobs = {"ipi": [], "policy": []}
    for row in rows:
        if all(len(v) >= args.n for v in jobs.values()):
            break
        _, js = rj.build_event_prompts(row["record"], None)
        for j in js:
            if "prompt" in j and len(jobs[j["head"]]) < args.n:
                jobs[j["head"]].append(j)

    budget = args.ctx - args.reserve
    results = {}
    for head, js in jobs.items():
        ok = bad = 0
        truncated = 0
        verdicts = []
        samples = []
        for j in js:
            ids = tok(j["prompt"], add_special_tokens=False)["input_ids"]
            was_trunc = len(ids) > budget
            if was_trunc:
                truncated += 1
                ids = ids[-budget:]  # keep the TAIL (evaluation target)
            text = tok.decode(ids)
            chat = tok.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
            enc = tok(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            parsed = cygnal_prompt.parse_judge_response(gen, head)
            if parsed["ok"]:
                ok += 1
                verdicts.append(parsed["flagged"])
            else:
                bad += 1
                if len(samples) < 4:
                    samples.append(gen.strip()[:200])
        n = ok + bad
        results[head] = dict(n=n, parseable=ok, unparseable=bad, truncated=truncated,
                             flagged=sum(verdicts), samples=samples)
        print(f"\n{head}: {ok}/{n} parseable ({ok/n:.0%}) | truncated inputs {truncated}/{n} "
              f"| verdict true-rate {sum(verdicts)}/{len(verdicts) or 1}")
        for s in samples:
            print("   unparseable sample:", repr(s))
    print("\n" + json.dumps(results, indent=2)[:1500])


if __name__ == "__main__":
    main()
