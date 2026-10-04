# Injection and defenses

Code for our study of indirect prompt injection (IPI) defenses: re-implementations of six system
defenses, callers for classifier defenses, a harness that replays an agent conversation up to the
injection and re-runs it under a defense, and the figure scripts.

## What's here

| Path | Contents |
|---|---|
| `system-defenses/ipi_eval/` | The replay harness: build records, roll out under a defense, grade. |
| `system-defenses/<Defense>/impl/` | Our re-implementations of CaMeL, FIDES, IPIGuard, MELON, CausalArmor and Firewalls, each with an AgentDojo runner and tests. |
| `system-defenses/<Defense>/ipi_suites/` | Runners that drive the reference code of CaMeL, FIDES and IPIGuard directly. |
| `system-defenses/configs/` | The exact settings of the paper's runs. |
| `classifier-defenses/` | Callers for detector defenses (StackOne Defender, Granite Guardian, PromptGuard, DataSentinel, LLM judges) and a registry, `defenses.json`. |
| `plots/`, `refined-figures/` | Figure scripts. They read run outputs, which are not shipped. |
| `ipi_arena_compat.py` | Additions the harness needs on top of the public `ipi_arena_bench` package (see its docstring). |
| `ipi_arena_os/` (submodule) | The public IPI Arena benchmark: behavior data, LLM client and world-sim. |
| `agentdojo/` | A vendored copy of AgentDojo (MIT). |

This repository ships no attack data, transcripts or run outputs. You run it on the public
behaviors in `ipi_arena_os/data/` with your own attack strings, or on AgentDojo.

## Setup

```bash
git submodule update --init
pip install -e ipi_arena_os -e agentdojo -e ".[detectors,analysis,ipiguard,dev]"
# or with uv, which resolves both local packages itself:
uv pip install -e ".[detectors,analysis,ipiguard,dev]"
```

Leave out the extras you don't need: `detectors` (local classifier models), `analysis` (figures),
`ipiguard` (IPIGuard's AgentDojo fork) and `dev` (pytest). The harness runs on Linux and macOS.

**API keys.** Set the ones your run uses as environment variables: `OPENROUTER_API_KEY`,
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`. You can instead write lines
such as `openrouter = <key>` in a `secrets.md` at the repository root (git-ignored). Environment
variables win.

## Run a defense on public data

```bash
cd system-defenses
# 1. One record per (behavior, attack). Any ipi-arena-bench config works; only its
#    `behaviors:` section is read.
python -m ipi_eval.public_data --config ../ipi_arena_os/examples/inline_attacks.yaml --out records.jsonl
# 2. Roll out under a defense and grade.
python -m ipi_eval.run_eval --defense melon --valset records.jsonl \
    --model <target-model> --base-url <openai-compatible-url>
```

- **Defenses:** `baseline` (no defense), `camel`, `causalarmor`, `fides`, `firewalls`, `ipiguard`, `melon`.
- **Defaults:** the world-sim is `gpt-5-mini` (needs `OPENAI_API_KEY`) and the judge is
  Gemini 3 Flash via OpenRouter (needs `OPENROUTER_API_KEY`). Change them with `--worldsim-model`
  and `--judge-model`. `python -m ipi_eval.run_eval --help` lists every option.
- **Scope:** the 33 `tool` and `coding` behaviors. The `browser` behaviors need rendered
  screenshots and are not supported.
- **Grading** uses this repository's judge (`ipi_eval/judges.py`), not the `ipi-arena-bench`
  judge. Tool criteria follow the public `tool_judge` rules and count only calls made during the
  rollout. LLM criteria use the arena judge prompt. The docstring of `ipi_eval/public_data.py`
  gives the full mapping.

## Run a defense on AgentDojo

Each `system-defenses/<Defense>/impl/run_agentdojo.py` runs that defense on an AgentDojo suite and
needs no external data. The docstring at the top of each script gives its command and the model
servers it expects.

## Plan-based runs (`ipi_suites/`)

CaMeL and IPIGuard's `ipi_suites/` runners replay a plan made once per behavior. **Plans are not
shipped; generate them with the scripts below.** Planning calls a hosted model (default
`gemini-3-flash` via OpenRouter). Generated packets and plans contain your attack strings and are
git-ignored. You don't need any of this for `ipi_eval`, which runs every defense without plans.

```bash
cd system-defenses
python -m ipi_eval.public_data --config <your-config.yaml> --out records.jsonl
# One packet per behavior, shared by all three runners -> CaMeL/ipi_suites/packets/
python -m CaMeL.ipi_suites.build_packets --records records.jsonl
# CaMeL plans -> CaMeL/ipi_suites/plans/ (in CaMeL's own uv environment)
(cd CaMeL/ipi_suites && uv run --project ../camel-prompt-injection python build_plans.py)
# IPIGuard plans -> IPIGuard/ipi_suites/plans/
(cd IPIGuard/ipi_suites && python make_plans.py --all && python build_index.py)
```

Then run CaMeL with `run_plan_eval.py --use-packet-injection`, IPIGuard with `run_batch.py --all`,
and FIDES (which plans at run time) with `python -m FIDES.ipi_suites.run_batch`. Each script's
docstring lists its options.

## Tests

The tests use fixtures built from public data only (`system-defenses/testlib/build_fixtures.py`).
Run each suite in its own process, because IPIGuard loads its own AgentDojo fork:

```bash
cd system-defenses
for d in ipi_eval/tests */impl/tests; do python -m pytest -q "$d"; done
```

Tests that need a local model server (Qwen on port 8000, BGE on port 8001) or an API key skip when
it is missing.

## Submodules

`ipi_arena_os/` is required. The other four are the defense papers' own code. CaMeL and IPIGuard
run on theirs (CaMeL's interpreter, IPIGuard's AgentDojo fork). Our FIDES and MELON
re-implementations do not need theirs.
