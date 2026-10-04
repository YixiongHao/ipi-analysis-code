#!/usr/bin/env bash
# Full banking run, all three arms, in dependency order (camel logs before secpol replay).
# Launched from impl/ so logs/ lands here. Uses the repo's uv env.
set -uo pipefail
cd "$(dirname "$0")"
REPO=../camel-prompt-injection
RUN() { echo "=== $* ==="; uv run --project "$REPO" python run_agentdojo.py "$@" 2>&1 | grep -v VIRTUAL_ENV; }

RUN --suite banking --variant undefended   --mode both
RUN --suite banking --variant camel         --mode both
RUN --suite banking --variant camel+secpol  --mode both

echo "ALL_DONE"
