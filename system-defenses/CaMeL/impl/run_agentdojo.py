# Replication harness: run CaMeL on AgentDojo with the local Qwen3-32B vLLM model.
#
# Reproduces the paper's config: suite x important_instructions attack, comparing
# undefended vs CaMeL (no policies) vs CaMeL+secpol. Metrics: utility (no injection)
# and security/ASR (with injection).
#
# Run from THIS directory so the relative `logs/` dir (used by the secpol replay path)
# lands in impl/logs:
#   cd impl
#   uv run --project ../camel-prompt-injection python run_agentdojo.py \
#       --suite banking --variant camel --mode both
#
# Order matters for camel+secpol: run --variant camel (both modes) FIRST so its code
# logs exist, then --variant camel+secpol reads them.

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Adapter sets OPENAI_BASE_URL/KEY on import and exposes the pipeline builder.
import agentdojo_adapter  # noqa: F401  (side effect: env routing)

# agentdojo top-level first (avoids camel circular import), then camel.
from agentdojo import attacks, benchmark, logging  # noqa: E402
from agentdojo.task_suite import get_suite  # noqa: E402

from defense import Defense  # noqa: E402

ATTACK = "important_instructions"
RESULTS_DIR = Path(__file__).parent / "results"


def _mean(d):
    vals = list(d.values())
    return sum(vals) / len(vals) if vals else float("nan")


def run(variant: str, suite_name: str, mode: str, user_tasks, limit, model="Qwen3-32B"):
    suite = get_suite("v1.2", suite_name)
    if user_tasks is None and limit is not None:
        user_tasks = [t.ID for t in list(suite.user_tasks.values())[:limit]]

    defense = Defense(model=model, variant=variant, attack_name=ATTACK)
    pipeline = defense.build_agentdojo_pipeline(suite_name)
    logdir = Path("./logs")

    out = {"variant": variant, "suite": suite_name, "pipeline_name": pipeline.name,
           "attack": ATTACK, "user_tasks": user_tasks}

    with logging.OutputLogger(str(logdir)):
        if mode in ("utility", "both"):
            res = benchmark.benchmark_suite_without_injections(
                pipeline, suite, logdir, force_rerun=False, user_tasks=user_tasks
            )
            ur = res["utility_results"]
            out["utility"] = {f"{k[0]}": v for k, v in ur.items()}
            out["utility_mean"] = _mean(ur)
            print(f"[{variant}/{suite_name}] utility = {out['utility_mean']:.3f} (n={len(ur)})")

        if mode in ("security", "both"):
            attack = attacks.load_attack(ATTACK, suite, pipeline)
            res = benchmark.benchmark_suite_with_injections(
                pipeline, suite, attack, logdir, force_rerun=False, user_tasks=user_tasks
            )
            ur = res["utility_results"]
            sr = res["security_results"]
            # security_results[k] == True means the *attack* succeeded.
            out["utility_under_attack_mean"] = _mean(ur)
            out["asr"] = _mean(sr)
            out["n_successful_attacks"] = sum(1 for v in sr.values() if v)
            out["n_attacks"] = len(sr)
            out["security_detail"] = {f"{k[0]}|{k[1]}": v for k, v in sr.items()}
            print(f"[{variant}/{suite_name}] ASR = {out['asr']:.3f} "
                  f"({out['n_successful_attacks']}/{out['n_attacks']} successful attacks), "
                  f"utility-under-attack = {out['utility_under_attack_mean']:.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = variant.replace("+", "_")
    fp = RESULTS_DIR / f"banking_{tag}_{mode}.json" if suite_name == "banking" else \
        RESULTS_DIR / f"{suite_name}_{tag}_{mode}.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    print(f"  -> wrote {fp}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="banking")
    ap.add_argument("--variant", default="camel",
                    choices=["undefended", "camel", "camel+secpol"])
    ap.add_argument("--mode", default="both", choices=["utility", "security", "both"])
    ap.add_argument("--user-tasks", nargs="*", default=None,
                    help="explicit user task IDs (e.g. user_task_0) for a smoke run")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N user tasks")
    ap.add_argument("--model", default="Qwen3-32B",
                    help="agent model; a provider/model id (e.g. deepseek/deepseek-v4-pro) routes to OpenRouter")
    args = ap.parse_args()
    run(args.variant, args.suite, args.mode, args.user_tasks, args.limit, model=args.model)


if __name__ == "__main__":
    main()
