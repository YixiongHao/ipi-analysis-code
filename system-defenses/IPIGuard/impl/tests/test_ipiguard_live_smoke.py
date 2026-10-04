"""Live smoke test for IPIGuard's TDG construction against the real Qwen3-32B (:8000).

@pytest.mark.live; skipped via the live_qwen fixture when :8000 is down. Asserts ONLY the
STRUCTURE of plan() output (a dict with tool_calls / planned_tools / pending). Makes NO
claim about WHICH tools the model planned and NO blocking / efficacy claim.
"""
import pytest

from testlib import fixtures, pathsetup

ig = pathsetup.load_impl("IPIGuard")

pytestmark = pytest.mark.live


def _flat_tools(record):
    out = []
    for t in record["openai_tools"]:
        fn = t.get("function", t)
        out.append({"name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {})})
    return out


def test_plan_returns_well_formed_tdg(live_qwen):
    rec = fixtures.ipi_transcript()
    d = ig.Defense()  # points at :8000 by default
    plan = d.plan(rec["user_request"], _flat_tools(rec))

    assert isinstance(plan, dict)
    assert "tool_calls" in plan and isinstance(plan["tool_calls"], list)
    assert "planned_tools" in plan and isinstance(plan["planned_tools"], set)
    assert "pending" in plan and isinstance(plan["pending"], list)
    # planned_tools must be derived from tool_calls (names only); strings, not objects.
    assert all(isinstance(name, str) for name in plan["planned_tools"])
