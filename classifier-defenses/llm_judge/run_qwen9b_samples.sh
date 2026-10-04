#!/bin/bash
# 3 sampled passes of llmjudge-qwen3.5-9b (provider-default temp=1 sampling) over the
# FULL store + FULL FP sets, both heads. Each pass gets its own sidecar via --out-suffix;
# passes s2/s3 salt the cache key so they draw fresh instead of replaying s1's draws
# (s1 is unsalted and therefore reuses the 07-28 1/5-run cached draws where they exist).
# The pinned bf16 providers 429 intermittently ("temporarily rate-limited upstream"), so
# each run uses --retries 8 (exp backoff to ~2min) and is re-invoked until 0 errors
# (errors are never cached; a re-run only re-hits the failed events).
set -u
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
LOG=llm_judge/qwen9b_samples.log
MAX_ATTEMPTS=5

wait_for_disk() {  # the shared box hit 100% on 07-31 and killed the s1 store pass mid-run;
    # don't start (or retry) a paid pass unless there is headroom for sidecar+cache writes
    while true; do
        avail_kb=$(df --output=avail <path> | tail -1 | tr -d ' ')
        [ "${avail_kb:-0}" -ge 786432 ] && return 0   # >= 768MB
        echo "$(date '+%F %T') disk low (${avail_kb}KB avail) - waiting 10min" >>"$LOG"
        sleep 600
    done
}

run_until_clean() {  # defense store results suffix [salt]
    local store=$1 results=$2 suffix=$3 salt=${4:-}
    local salt_args=()
    [ -n "$salt" ] && salt_args=(--cache-salt "$salt")
    for attempt in $(seq 1 $MAX_ATTEMPTS); do
        wait_for_disk
        echo "=== $(date '+%F %T') $store -> $results ($suffix) attempt $attempt ===" >>"$LOG"
        out=$($PY llm_judge/run_llm_judge.py --defense llmjudge-qwen3.5-9b \
            --store "$store" --results "$results" --out-suffix "$suffix" \
            "${salt_args[@]}" --concurrency 64 --retries 8 --resume 2>&1 | tail -5)
        echo "$out" >>"$LOG"
        errs=$(echo "$out" | grep -oP '\d+(?= errors)' | tail -1)
        [ "${errs:-1}" = "0" ] && return 0
        sleep 120
    done
    echo "!!! $store ($suffix): still ${errs:-?} errors after $MAX_ATTEMPTS attempts" >>"$LOG"
}

# n=2 plan (user decision 08-01): complete s1 (toolathlon full-108 was never run;
# store/rebench legs are cache-hit re-derives) then one fresh salted pass s2.
for suffix in s1 s2; do
    salt=$suffix
    [ "$suffix" = s1 ] && salt=""   # s1 unsalted: reuses the 1/5-run cache
    run_until_clean store/attacks.jsonl                                results_ipi/qwen9b_samples      "$suffix" "$salt"
    run_until_clean fp_rebench/records/swe_rebench_clean50.jsonl       fp_rebench/results_qwen9b_samples    "$suffix" "$salt"
    run_until_clean fp_toolathlon/records/toolathlon_verified_108.jsonl fp_toolathlon/results_qwen9b_samples "$suffix" "$salt"
done
echo "=== $(date '+%F %T') ALL DONE ===" >>"$LOG"
