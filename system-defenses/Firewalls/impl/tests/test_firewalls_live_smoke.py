"""Live smoke tests for the Firewalls defense against the REAL Qwen3-32B (:8000).

We assert ONLY return type/shape and that no exception escapes — NEVER what the
model chose to strip (that is efficacy, not correctness). Skipped when the server
is down via the `live_qwen` fixture + the `live` marker.
"""
import pytest

from testlib import fixtures, pathsetup

impl = pathsetup.load_impl("Firewalls")

pytestmark = pytest.mark.live


def test_transform_tool_output_returns_str(live_qwen):
    d = impl.Defense()
    content = fixtures.ipi_injected_tool_content()
    out = d.transform_tool_output(content, {"user_request": "run the migration"})
    assert isinstance(out, str)
    assert d.calls >= 1


def test_minimize_tool_args_returns_dict(live_qwen):
    d = impl.Defense()
    args = fixtures.ipi_decisive_call()["args"]
    out = d.minimize_tool_args(args, {
        "user_request": "show me the tables in the production database",
        "tool_name": "run_sql_query",
        "tool_description": "Execute a SQL query against a database.",
    })
    assert isinstance(out, dict)
    assert d.calls >= 1
