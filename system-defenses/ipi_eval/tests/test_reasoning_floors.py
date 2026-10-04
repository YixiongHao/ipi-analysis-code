"""Per-model reasoning floors: resolution + the payload each level actually produces.

Guards the two failure modes the 2026-08-05 audit found (planning/reasoning_floor_audit.md):
  - sending a reasoning param to a model with no reasoning knob (OpenAI 400s), and
  - sending a disable flag to a reasoning-mandatory model (400s, or silently ignored).
"""
from __future__ import annotations

import pytest

from ipi_eval import engine
from ipi_eval.reasoning_floors import FLOORS, LEVELS, describe, floor_for

OR = "https://openrouter.ai/api/v1"


def _eb(model: str, effort: str | None):
    return engine.make_client(model, OR, reasoning_effort=effort, api_key="test").extra_body


def test_table_levels_are_valid():
    for model, (level, evidence, _note) in FLOORS.items():
        assert level in LEVELS, f"{model} has bogus level {level!r}"
        assert evidence in ("MEASURED", "SOURCED"), f"{model} has bogus evidence {evidence!r}"


@pytest.mark.parametrize("model,expected", [
    ("openai/gpt-4", "none"),
    ("gpt-4o-2024-08-06", "none"),
    ("openai/gpt-5.6-sol", "off"),
    ("z-ai/glm-5.2", "off"),
    ("openai/o1", "low"),
    ("google/gemini-2.5-pro", "low"),
    ("anthropic/claude-fable-5", "low"),
    ("openai/gpt-5.4-pro", "medium"),
    ("moonshotai/kimi-k3", "max"),
    ("deepseek/deepseek-r1", "mandatory"),
    ("qwen/qwen3-235b-a22b-thinking-2507", "mandatory"),
])
def test_floor_lookup(model, expected):
    assert floor_for(model) == expected


def test_slug_normalization():
    # provider prefix and :tag suffix both stripped
    assert floor_for("z-ai/glm-5.2:batch") == floor_for("glm-5.2") == "off"
    assert floor_for("OpenAI/GPT-5.6-Sol") == "off"


def test_unknown_model_defaults_to_off_and_says_so():
    assert floor_for("some-vendor/never-seen-1") == "off"
    assert "UNVERIFIED" in describe("some-vendor/never-seen-1")


def test_no_knob_sends_no_reasoning_field():
    # gpt-4 has no reasoning knob: a reasoning param is an API error, so the field must be absent
    assert "reasoning" not in _eb("openai/gpt-4", "floor")


def test_off_sends_disable_flag():
    assert _eb("openai/gpt-5.6-sol", "floor")["reasoning"] == {"enabled": False}


def test_effort_floor_sends_effort_not_disable():
    eb = _eb("openai/gpt-5.4-pro", "floor")
    assert eb["reasoning"] == {"effort": "medium"}


def test_mandatory_never_sends_a_disable_flag():
    # the silent-failure case: a disable flag here would be ignored and the arm mislabeled
    eb = _eb("qwen/qwen3-235b-a22b-thinking-2507", "floor")
    assert eb.get("reasoning") != {"enabled": False}
    assert "effort" not in eb.get("reasoning", {})


@pytest.mark.parametrize("model,cap", [
    ("openai/gpt-4", 4096),           # none -> no reasoning -> small cap fine
    ("openai/gpt-5.6-sol", 4096),     # off  -> no reasoning -> small cap fine
    ("openai/o1", 8192),              # low  -> hidden CoT needs headroom
    ("moonshotai/kimi-k3", 8192),     # max
    ("deepseek/deepseek-r1", 8192),   # mandatory
])
def test_nonzero_floors_get_the_bigger_token_cap(model, cap):
    # 4096 under hidden CoT is the documented "CoT eats the budget -> empty content" failure
    assert engine.make_client(model, OR, reasoning_effort="floor", api_key="test").max_tokens == cap


def test_explicit_levels_still_work_unchanged():
    # backward compatibility: the union381 ablation arms must be byte-identical
    assert _eb("z-ai/glm-5.2", "off")["reasoning"] == {"enabled": False}
    assert _eb("z-ai/glm-5.2", "low")["reasoning"] == {"effort": "low"}
    assert _eb("z-ai/glm-5.2", None)["reasoning"] == {"enabled": False}
