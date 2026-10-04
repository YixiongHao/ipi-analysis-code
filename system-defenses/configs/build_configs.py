"""Generate the per-(target-model x defense) system-defense config stores.

Mirrors the classifier-defenses `defenses.json` pattern (config-driven, `config_hash`-stamped),
but adds the system-defense dimensions: a `target` model block (each model x defense pair is its
own store, per the design decision), the `runner` (fork vs ipi_eval adapter), the `worldsim`
regime, the `aux_models` (Gemini unless the component IS the attacked target), and the `judge`.

Emits  configs/<target>/<defense>.json  (2 targets x 6 defenses = 12 stores) + a machine index
`configs/index.json`. Regenerable: edit the templates below and re-run.

    python system-defenses/configs/build_configs.py
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
OPENROUTER = "https://openrouter.ai/api/v1"
GEMINI_AUX = "google/gemini-3-flash-preview"   # shared machinery model (worldsim / judge / planners)

# --- shared blocks (identical across all 12 stores) -----------------------------------------
WORLDSIM = {"mode": "all", "goal_aware": True, "model": GEMINI_AUX, "base_url": OPENROUTER}
JUDGE = {"model": GEMINI_AUX, "base_url": OPENROUTER, "thinking": True, "temperature": 0.0,
         "max_tokens": 8192, "rubric_source": "per-corpus behaviors_detailed.json (auto by behavior_id)"}

# --- the two target-model stores (orthogonal to the defense) --------------------------------
TARGETS = {
    "glm":    {"model": "z-ai/glm-5.2",               "provider_pin": "z-ai",           "fork_preset": "glm"},
    "gemini": {"model": "google/gemini-3-flash-preview", "provider_pin": "google-ai-studio", "fork_preset": "gemini-3-flash"},
    "deepseek": {"model": "deepseek/deepseek-v4-pro", "provider_pin": "deepseek",        "fork_preset": "deepseek"},
}

# --- the six defenses (model-agnostic templates; {preset}/{model}/{base_url} filled per target) ---
DEFENSES = [
    {
        "defense_id": "camel", "defense_type": "capability_interpreter",
        "runner": {"kind": "fork", "entry": "CaMeL/ipi_suites/run_plan_eval.py",
                   "venv": "uv run --project ../camel-prompt-injection python",
                   "cmd": "run_plan_eval.py --arm policy --model {preset} --use-packet-injection --thinking-off --temperature 1.0",
                   "grader": "delivered-gated policy_blocked (grade_forks.py)"},
        "delivery": "carrier-first plan (unconditional by construction)",
        "params": {"arm": "policy", "use_packet_injection": True, "thinking_off": True,
                   "temperature": 1.0, "all_worldsim": True,
                   "plan_source": "static reauthored plans/<bid>.json (injection-blind)"},
        "aux_models": {"p_llm_planner": GEMINI_AUX, "q_llm": "target"},
        "description": "Real reference CaMeL (P-LLM -> restricted-Python interpreter -> capabilities -> security policy) over the 88 IPI packets; policy arm.",
    },
    {
        "defense_id": "fides", "defense_type": "labeled_planner_ifc",
        "runner": {"kind": "fork", "entry": "FIDES/ipi_suites/run_batch.py",
                   "venv": "python -m FIDES.ipi_suites.run_batch  (cwd=system-defenses)",
                   "cmd": "run_batch.py --model {preset} --forced-delivery --temperature 1.0",
                   "grader": "grade_forks.py + Arena judge (or run_asr.py for direct baseline-vs-fides ASR)"},
        "delivery": "--forced-delivery (per-packet flag)",
        "params": {"forced_delivery": True, "thinking": False, "temperature": 1.0,
                   "all_worldsim": True, "pt_labels": "tool_labels.json (P-T / P-F)"},
        "aux_models": {"labeled_planner": "target"},
        "description": "Real labeled planner (typed tools + IFC taint + per-tool P-T + readers P-F) over the 88 IPI packets.",
    },
    {
        "defense_id": "ipiguard", "defense_type": "tdg_fake_tool",
        "runner": {"kind": "fork", "entry": "IPIGuard/ipi_suites/run_batch.py",
                   "venv": "python  (cwd=system-defenses/IPIGuard/ipi_suites)",
                   "cmd": "run_batch.py --all --model {preset} --arms baseline,ipiguard --reuse-plans --temperature 1.0",
                   "grader": "delivered-gated grade_forks.py"},
        "delivery": "delivering hardened DAG (reuse-plans)",
        "params": {"reuse_plans": True, "timeout_s": 240, "temperature": 1.0, "all_worldsim": True,
                   "plan_source": "hardened audit-stripped DAGs plans/<bid>.json"},
        "aux_models": {"construct_llm": GEMINI_AUX, "traverse_llm": "target"},
        "description": "Real TDG fork (Argument Estimation + Node Expansion + Fake Tool Invocation) over the 88 IPI packets.",
    },
    {
        "defense_id": "melon", "defense_type": "masking_detector",
        "runner": {"kind": "adapter", "entry": "ipi_eval/run_eval.py",
                   "venv": "python -m ipi_eval.run_eval  (cwd=system-defenses)",
                   "cmd": "run_eval.py --defense melon --model {model} --base-url {base_url} --temperature 1.0 --live-carrier",
                   "grader": "built-in Arena judge -> results/melon.jsonl"},
        "delivery": "--live-carrier",
        "params": {"registry": "melon", "embed_backend": "openai", "sim_threshold": 0.8,
                   "augment": False, "live_carrier": True, "temperature": 1.0},
        "aux_models": {"embedder": "text-embedding-3-large", "masking_reexec": "target"},
        "description": "Per-step masking detector (accumulating cache, tool-calls-only theta compare, abort -> strip tool_calls).",
    },
    {
        "defense_id": "causalarmor", "defense_type": "loo_attribution",
        "runner": {"kind": "adapter", "entry": "ipi_eval/run_eval.py",
                   "venv": "python -m ipi_eval.run_eval  (cwd=system-defenses)",
                   "cmd": ("run_eval.py --defense causalarmor-nocot --model {model} --base-url {base_url} "
                           "--temperature 1.0 --live-carrier "
                           "--defense-kwarg proxy_model=gemma-3-12b-it "
                           "--defense-kwarg base_url=http://localhost:8002/v1 "
                           "--defense-kwarg tokenizer_path=google/gemma-3-12b-it"),
                   "grader": "built-in Arena judge -> results/causalarmor-nocot.jsonl"},
        "delivery": "--live-carrier",
        "params": {"registry": "causalarmor-nocot", "cot_masking": False, "live_carrier": True,
                   "temperature": 1.0},
        "aux_models": {"proxy_logprob": "gemma-3-12b-it", "proxy_base_url": "http://localhost:8002/v1",
                       "sanitizer": GEMINI_AUX, "agent": "target"},
        "notes": "proxy needs a LOCAL vLLM server on :8002 exposing token logprobs (Gemini can't); nocot per validation-best.",
        "description": "Leave-one-out logprob attribution (gemma proxy) + injected-span sanitize (Gemini) + action re-generation; CoT masking OFF.",
    },
    {
        "defense_id": "firewalls", "defense_type": "io_firewall",
        "runner": {"kind": "adapter", "entry": "ipi_eval/run_eval.py",
                   "venv": "python -m ipi_eval.run_eval  (cwd=system-defenses)",
                   "cmd": ("run_eval.py --defense firewalls --model {model} --base-url {base_url} "
                           "--temperature 1.0 --live-carrier "
                           "--defense-kwarg model=google/gemini-3-flash-preview "
                           "--defense-kwarg base_url=https://openrouter.ai/api/v1"),
                   "grader": "built-in Arena judge -> results/firewalls.jsonl"},
        "delivery": "--live-carrier",
        "params": {"registry": "firewalls", "live_carrier": True, "temperature": 1.0},
        "aux_models": {"minimizer_sanitizer": GEMINI_AUX},
        "notes": "Aux (Minimizer F_I + Sanitizer F_O) = Gemini-3-Flash via the cmd's --defense-kwarg model/base_url (FirewallsDefense auto-loads the OpenRouter key). No local aux server needed.",
        "description": "Tool-I/O firewall (Minimizer F_I on tool args + Sanitizer F_O on tool outputs).",
    },
]


def _hash(cfg: dict) -> str:
    payload = {k: v for k, v in cfg.items() if k != "config_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def build() -> list[dict]:
    index = []
    for tkey, t in TARGETS.items():
        for d in DEFENSES:
            runner = dict(d["runner"])
            runner["cmd"] = runner["cmd"].format(preset=t["fork_preset"], model=t["model"], base_url=OPENROUTER)
            cfg = {
                "store_id": f"{tkey}__{d['defense_id']}",
                "defense_id": d["defense_id"],
                "defense_type": d["defense_type"],
                "target": {"model": t["model"], "base_url": OPENROUTER, "provider_pin": t["provider_pin"],
                           "thinking": False, "temperature": 1.0},
                "runner": runner,
                "delivery": d["delivery"],
                "params": d["params"],
                "worldsim": WORLDSIM,
                "aux_models": d["aux_models"],
                "judge": JUDGE,
                "description": d["description"],
            }
            if "notes" in d:
                cfg["notes"] = d["notes"]
            cfg["config_hash"] = _hash(cfg)
            out = _HERE / tkey / f"{d['defense_id']}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(cfg, indent=2) + "\n")
            index.append({"store_id": cfg["store_id"], "path": str(out.relative_to(_HERE)),
                          "target_model": t["model"], "defense_id": d["defense_id"],
                          "runner_kind": runner["kind"], "config_hash": cfg["config_hash"]})
    (_HERE / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


if __name__ == "__main__":
    idx = build()
    print(f"wrote {len(idx)} store configs + index.json under {_HERE}")
    for r in idx:
        print(f"  {r['store_id']:24s} {r['runner_kind']:8s} {r['config_hash']}  {r['path']}")
