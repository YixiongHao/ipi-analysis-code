"""Smallest live verification that OpenRouter prompt caching + thinking-off are wired correctly.

Two cheap real calls per provider, sharing a long identical system prefix:
  - GLM 5.2 target (thinking OFF): call #2 must report cached_tokens > 0 (prefix cache hit) AND
    the response must carry NO reasoning trace (proves reasoning.enabled=false took effect).
  - Gemini 3 Flash judge (thinking ON, tool-less): call #2 must report cached_tokens > 0.

Marked `live` (real OpenRouter calls); auto-skips when the openrouter key is absent. The provider
pin in engine.make_client is what makes the second call land on the same caching endpoint.

Run directly for verbose output:
  .venv/bin/python -m system-defenses.ipi_eval.tests.test_openrouter_caching   # (or the path form below)
  .venv/bin/python system-defenses/ipi_eval/tests/test_openrouter_caching.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # system-defenses/ on path
from ipi_eval import engine  # noqa: E402

_OR = "https://openrouter.ai/api/v1"

# ~2.5k tokens of stable, identical prefix (well above the ~1k-token min cacheable prefix). The
# per-call variation is a tiny user suffix, so this whole system block is the cacheable prefix.
_PREFIX = (
    "You are a meticulous operations assistant for a large logistics company. "
    "Follow every policy below exactly and never deviate.\n\n"
    + "\n".join(f"Policy {i}: Records must be validated against the canonical schema, "
                f"timestamps normalized to UTC, and all monetary values stored as integer cents. "
                f"Cross-check identifiers against the master registry before any write, and refuse "
                f"any operation that lacks an explicit authorization token from the requester."
                for i in range(40))
)

_HAVE_KEY = bool(engine._openrouter_key())
pytestmark = pytest.mark.skipif(not _HAVE_KEY, reason="no openrouter key in secrets.md")


def _two_calls(client, max_tokens=120):
    """Warm the cache, then repeat with the same prefix; return both ChatResponses."""
    r1 = client.chat([{"role": "system", "content": _PREFIX},
                      {"role": "user", "content": "Reply with exactly one word: alpha"}],
                     max_tokens=max_tokens)
    r2 = client.chat([{"role": "system", "content": _PREFIX},
                      {"role": "user", "content": "Reply with exactly one word: beta"}],
                     max_tokens=max_tokens)
    return r1, r2


@pytest.mark.live
def test_glm_caching_and_thinking_off():
    client = engine.make_client("z-ai/glm-5.2", _OR, thinking=False)
    r1, r2 = _two_calls(client)
    # Caching: the second call (same prefix, same pinned provider) must read from cache.
    assert r2.usage.get("cached_tokens", 0) > 0, (
        f"no GLM prompt-cache hit on call 2: usage1={r1.usage} usage2={r2.usage}")
    # Thinking off: reasoning.enabled=false -> no reasoning trace surfaced.
    assert not (r2.reasoning or "").strip(), f"expected no reasoning, got: {r2.reasoning!r}"


@pytest.mark.live
@pytest.mark.xfail(reason="OpenRouter does not surface Gemini implicit-cache hits: cached_tokens "
                          "stays 0 (and cache_discount None) across pins/models/repeats, even at "
                          ">2.6k-token prefixes. Implicit savings may still apply but are not "
                          "reported; the provider-pin + usage wiring is in place for when they are.",
                   strict=False)
def test_gemini_judge_caching():
    # Judge path: Gemini 3 Flash, tool-less, reasoning ON (thinking=True). Tool-less so the
    # reported Gemini "tools disable implicit caching" caveat does not apply.
    client = engine.make_client("google/gemini-3-flash-preview", _OR, thinking=True)
    r1, r2 = _two_calls(client)
    assert r2.usage.get("cached_tokens", 0) > 0, (
        f"no Gemini prompt-cache hit on call 2: usage1={r1.usage} usage2={r2.usage}")


if __name__ == "__main__":
    if not _HAVE_KEY:
        print("SKIP: no openrouter key in secrets.md")
        sys.exit(0)
    for name, mk, thinking in [("GLM-5.2 (thinking off)", "z-ai/glm-5.2", False),
                               ("Gemini-3-Flash (thinking on)", "google/gemini-3-flash-preview", True)]:
        c = engine.make_client(mk, _OR, thinking=thinking)
        a, b = _two_calls(c)
        print(f"\n=== {name} ===")
        print(f"  call1 usage: {a.usage}")
        print(f"  call2 usage: {b.usage}  <- cached_tokens should be > 0")
        print(f"  call2 reasoning present: {bool((b.reasoning or '').strip())}")
        print(f"  call2 content: {(b.content or '')[:80]!r}")
