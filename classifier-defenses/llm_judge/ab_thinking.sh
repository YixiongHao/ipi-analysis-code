#!/usr/bin/env bash
# Step-2 A/B: reasoning effort none vs minimal for the GPT-5.6-Luna judge,
# on a small TP set (store/tp100_seed0, 126 tagged store records) and a small
# FP set (fp_toolathlon smoke5, 5 benign admin records).
# Results land in llm_judge/ab_results/{tp,fp}/<defense_id>.jsonl;
# score with: ../.venv/bin/python llm_judge/score_ab.py
set -euo pipefail
cd "$(dirname "$0")/.."   # classifier-defenses/
PY=../.venv/bin/python
AB=llm_judge/ab_results
if [ $# -gt 0 ]; then ARMS=("$@"); else ARMS=(llmjudge-gpt-5.6-luna llmjudge-gpt-5.6-luna-lowthink); fi

for defense in "${ARMS[@]}"; do
  echo "=== $defense / TP (tp100_seed0) ==="
  $PY llm_judge/run_llm_judge.py --defense "$defense" \
      --store store/tp100_seed0/attacks.jsonl --results "$AB/tp" --concurrency 16
  echo "=== $defense / FP (toolathlon smoke5) ==="
  $PY llm_judge/run_llm_judge.py --defense "$defense" \
      --store fp_toolathlon/records/smoke5.jsonl --results "$AB/fp" --concurrency 16
done

$PY llm_judge/score_ab.py "${ARMS[@]}"
