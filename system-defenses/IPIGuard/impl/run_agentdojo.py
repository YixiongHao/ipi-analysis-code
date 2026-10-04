"""Phase-4 replication harness: run the fork's `ipiguard` defense (and a No-Defense
baseline) on AgentDojo with a local Qwen3-32B vLLM model.

Reuses the reference repo's own eval loop (`run/eval.py`: `eval`, `benign_eval`,
`initialize_dataset`) for fidelity, swapping in a pipeline composed by our adapter
(so the LLM points at our local endpoint). Supports slicing the dataset for cost.

Examples:
  # smoke: 1 user task, no attack, ipiguard, banking
  python run_agentdojo.py --suite banking --mode benign --defense ipiguard --max-user 1
  # attack slice: 2 users x 2 injections, important_instructions, ipiguard, banking
  python run_agentdojo.py --suite banking --mode attack --defense ipiguard --max-user 2 --max-inj 2
"""
import argparse
import os
import sys

import agentdojo_adapter as A  # sets up fork sys.path + stubs

_RUN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ipiguard", "run"
)
sys.path.insert(0, _RUN_DIR)
import eval as ipiguard_eval  # noqa: E402  (repo's run/eval.py; __main__ guarded)
from data_module import initialize_dataset  # noqa: E402

from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


class _Args:
    """Minimal stand-in for the repo's ScriptArguments (only fields eval() reads)."""

    def __init__(self, output_dir, uid=0, iid=0):
        self.output_dir = output_dir
        self.uid = uid
        self.iid = iid


def slice_attack_dataset(suite, max_user, max_inj):
    ds = initialize_dataset(suite)  # list of (user_id, injection_id)
    delta = 1 if suite == "slack" else 0
    out = [(u, i) for (u, i) in ds if u < max_user and (i - delta) < max_inj]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="banking",
                    choices=["banking", "workspace", "slack", "travel"])
    ap.add_argument("--mode", default="attack", choices=["benign", "attack"])
    ap.add_argument("--defense", default="ipiguard", choices=["ipiguard", "None"])
    ap.add_argument("--attack", default="important_instructions")
    ap.add_argument("--max-user", type=int, default=2)
    ap.add_argument("--max-inj", type=int, default=2)
    ap.add_argument("--model", default=A.DEFAULT_MODEL)
    ap.add_argument("--base-url", default=A.DEFAULT_BASE_URL)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    benchmark_version = "v1.1.2"
    suite = get_suite(benchmark_version, args.suite)
    pipeline = A.build_pipeline(args.defense, model=args.model, base_url=args.base_url)

    tag = args.tag or f"{args.suite}_{args.mode}_{args.attack}_{args.defense}_u{args.max_user}_i{args.max_inj}"
    out_dir = os.path.join(RESULTS_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    script_args = _Args(out_dir)

    attacker = load_attack(args.attack, suite, pipeline)

    if args.mode == "benign":
        dataset = initialize_dataset(args.suite, benign=True)[: args.max_user]
        security, useful, n, asr, ability = ipiguard_eval.benign_eval(
            script_args, pipeline, suite, attacker, dataset
        )
    else:
        dataset = slice_attack_dataset(args.suite, args.max_user, args.max_inj)
        security, useful, n, asr, ability = ipiguard_eval.eval(
            script_args, pipeline, suite, attacker, dataset
        )

    print(f"\n=== {tag} ===")
    print(f"n={n}  ASR={asr:.2f}%  Utility={ability:.2f}%  (out_dir={out_dir})")
    ipiguard_eval.save_result(
        {"summary": {"tag": tag, "model": args.model, "n": n, "asr": asr,
                     "utility": ability, "mode": args.mode, "defense": args.defense,
                     "attack": args.attack, "suite": args.suite}},
        out_dir,
    )


if __name__ == "__main__":
    main()
