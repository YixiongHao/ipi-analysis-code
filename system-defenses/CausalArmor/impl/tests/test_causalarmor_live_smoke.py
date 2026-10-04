"""Live smoke tests for CausalArmor against the local Qwen3-32B vLLM endpoint (:8000).

These exercise the REAL backends end-to-end (tokenizer chat-template + /v1/completions
echo+logprobs for analyze; /v1/chat/completions for sanitize). We assert only RETURN
TYPE / SHAPE / no-crash — NEVER that a real model flags or removes the injection (that is
non-deterministic and measured separately by ipi_eval).

Skipped when the Qwen server is down (`live_qwen` fixture) or the tokenizer path is absent.
"""
import os

import pytest

from testlib import fixtures, pathsetup

ca = pathsetup.load_impl("CausalArmor")

pytestmark = pytest.mark.live

_TOKENIZER_PATH = ca.Defense().tokenizer_path


def _tokenizer_present() -> bool:
    return os.path.isdir(_TOKENIZER_PATH)


def _openrouter_key() -> str:
    from ipi_arena_compat import read_secret   # OPENROUTER_API_KEY or secrets.md
    return read_secret("openrouter")


@pytest.mark.skipif(not _tokenizer_present(),
                    reason=f"tokenizer not found at {_TOKENIZER_PATH}")
def test_analyze_returns_well_formed_attribution(live_qwen):
    d = ca.Defense(tau=0.0)
    messages = fixtures.ipi_prefill_messages()           # has user + 3 tool spans
    action = d.serialize_action(*fixtures.ipi_decisive_call().values())

    attr = d.analyze(action, messages)

    # Right type and field types — no efficacy claim.
    assert type(attr).__name__ == "Attribution"
    assert isinstance(attr.action, str)
    assert isinstance(attr.n_action_tokens, int) and attr.n_action_tokens >= 1
    assert attr.user_index == 1
    assert isinstance(attr.delta_u, float)
    assert isinstance(attr.delta_u_norm, float)
    # One SpanScore per tool message, at the right indices.
    tool_indices = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    assert [s.index for s in attr.spans] == tool_indices
    assert all(isinstance(s.delta, float) and isinstance(s.delta_norm, float)
               for s in attr.spans)
    # flagged is a subset of the span indices (membership, not which/whether).
    assert isinstance(attr.flagged, list)
    assert set(attr.flagged) <= set(tool_indices)


@pytest.mark.skipif(not _openrouter_key(),
                    reason="no OpenRouter key (OPENROUTER_API_KEY or secrets.md; Gemini-3-Flash sanitizer)")
def test_sanitize_returns_str():
    # Sanitizer = Gemini-3-Flash via OpenRouter (not the local Qwen server), so this no longer
    # depends on live_qwen — only on the OpenRouter key.
    d = ca.Defense(sanitizer_api_key=_openrouter_key())
    out = d.sanitize(fixtures.ipi_injected_tool_content(),
                     user_request="run the migration", tool_name="read_file")
    assert isinstance(out, str)


@pytest.mark.skipif(not _tokenizer_present(),
                    reason=f"tokenizer not found at {_TOKENIZER_PATH}")
def test_detect_returns_bool_float(live_qwen):
    d = ca.Defense(tau=0.0)
    messages = fixtures.ipi_prefill_messages()
    action = d.serialize_action(*fixtures.ipi_decisive_call().values())
    flagged, score = d.detect(fixtures.ipi_injected_tool_content(),
                              {"messages": messages, "action": action})
    assert isinstance(flagged, bool)
    assert isinstance(score, float)
