# Run configs

The exact settings of the paper's system-defense runs: one JSON per (target model, defense) at
`configs/<target>/<defense>.json`, 3 targets x 6 defenses = 18 files, indexed in `index.json`.

They are generated. To change them, edit the templates in `build_configs.py` and run
`python system-defenses/configs/build_configs.py`; do not edit the JSON by hand.

## Fields

| Field | Meaning |
|---|---|
| `store_id` | `<target>__<defense>` |
| `defense_id`, `defense_type` | The defense and its kind |
| `target` | The attacked model: `model`, `base_url`, `provider_pin`, `thinking`, `temperature` |
| `runner` | `kind` (`fork` = an `ipi_suites/` runner, `adapter` = `ipi_eval`), `entry`, `venv`, the full `cmd`, and `grader` |
| `delivery` | How the injection is made to reach the model |
| `params` | The defense's own settings |
| `worldsim` | World-sim model and mode |
| `aux_models` | Model for each defense component; `"target"` means the run's target model |
| `judge` | Grading model and sampling settings |
| `config_hash` | First 12 hex characters of the SHA-256 of the file without this field; identifies the run |

## Shared settings

- Target: temperature 1.0, reasoning off, at most 5 steps.
- World-sim: simulates every tool and sees the attack's goal.
- Judge: Gemini 3 Flash, reasoning on, temperature 0, 8192 max tokens.
- Helper models: Gemini 3 Flash, except where a component is the model under test (CaMeL's Q-LLM,
  FIDES's planner, IPIGuard's traversal, MELON's re-run), which uses the target. MELON's embedder is
  `text-embedding-3-large`. CausalArmor's attribution proxy is `gemma-3-12b-it`, served locally on
  port 8002 because it needs log-probabilities; without that server, CausalArmor allows every call.

Targets: `glm` = `z-ai/glm-5.2`, `gemini` = `google/gemini-3-flash-preview`, `deepseek` =
`deepseek/deepseek-v4-pro`, all via OpenRouter with a provider pin.
