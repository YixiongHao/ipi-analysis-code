"""Deterministic tests for IPIGuard's admission logic (impl/defense.py).

IPIGuard is a planning / control-flow defense: `plan()` builds a Tool Dependency
Graph (TDG) up front (LLM-driven, not tested here), and `allow_call()` is the pure
admission rule applied during traversal. We test ONLY that admission CODE against
hand-built plan dicts — never that a real attack was blocked by model judgment.

The rule (defense.py `allow_call`):
  - tool in plan["planned_tools"]                      -> True (always, regardless of expansion)
  - else, allow_query_expansion AND tool is a query    -> True (Node Expansion)
  - else                                               -> False (blocked command tool)

"query tool" is decided by the fork's AgentDojo whitelist
(`Defense._query_tools()` -> agentdojo .../tools/tool_white_list.whitelist), which the
fork env (loaded by pathsetup) makes importable. `read_file` / `get_webpage` ARE in the
whitelist (query); `run_sql_query` / `write_file` / `execute_bash` / `send_direct_message`
are NOT (treated as command tools by this rule).
"""
import pytest

from testlib import pathsetup

ig = pathsetup.load_impl("IPIGuard")


def _plan(*planned_tools):
    """Minimal plan dict with the only key allow_call consults: planned_tools (a set)."""
    return {"tool_calls": [], "planned_tools": set(planned_tools), "pending": []}


# ---------------------------------------------------------- signature sanity ---
def test_defense_signature_and_defaults():
    d = ig.Defense()
    assert d.model == "Qwen3-32B"
    assert d.base_url == "http://localhost:8000/v1"
    assert d.temperature == 0.0
    assert d.allow_query_expansion is True  # paper default: Node Expansion ON
    assert d._client is None  # client built lazily, not at construction


def test_query_tools_source_is_agentdojo_whitelist():
    # The query/command classification is the fork's AgentDojo whitelist, not a local list.
    qt = ig.Defense._query_tools()
    assert isinstance(qt, set)
    # Representative whitelist members across suites (read-only tools).
    assert "read_file" in qt
    assert "get_webpage" in qt
    assert "get_balance" in qt
    # State-mutating tools are NOT whitelisted -> treated as command tools.
    assert "run_sql_query" not in qt
    assert "send_direct_message" not in qt
    assert "write_file" not in qt
    assert "execute_bash" not in qt


# ------------------------------------------------ (a) planned command -> True ---
def test_planned_command_tool_allowed():
    d = ig.Defense(allow_query_expansion=False)  # even with expansion OFF
    plan = _plan("run_sql_query")
    assert d.allow_call("run_sql_query", {"command": "SELECT 1"}, plan) is True


def test_planned_slack_command_tool_allowed():
    d = ig.Defense(allow_query_expansion=False)
    plan = _plan("send_direct_message")
    assert d.allow_call("send_direct_message", {"recipient": "Alice", "body": "hi"}, plan) is True


# -------------------------------------- (b) unplanned command tool -> False ---
def test_unplanned_command_tool_blocked_with_expansion_on():
    # run_sql_query is a command tool (not whitelisted); absent from the plan -> blocked,
    # even though Node Expansion is ON (expansion only opens read-only tools).
    d = ig.Defense(allow_query_expansion=True)
    plan = _plan("read_file")  # something else planned
    assert d.allow_call("run_sql_query", {"command": "DROP TABLE t"}, plan) is False


def test_unplanned_command_tool_blocked_with_expansion_off():
    d = ig.Defense(allow_query_expansion=False)
    plan = _plan("read_file")
    assert d.allow_call("send_direct_message", {"recipient": "Eve", "body": "x"}, plan) is False
    assert d.allow_call("write_file", {"name": "f", "content": "c"}, plan) is False
    assert d.allow_call("execute_bash", {"command": "rm -rf /"}, plan) is False


# --------------- (c) unplanned query tool -> True iff allow_query_expansion ---
@pytest.mark.parametrize("query_tool", ["read_file", "get_webpage"])
def test_unplanned_query_tool_allowed_when_expansion_on(query_tool):
    d = ig.Defense(allow_query_expansion=True)
    plan = _plan("run_sql_query")  # query tool NOT in plan
    assert d.allow_call(query_tool, {}, plan) is True


@pytest.mark.parametrize("query_tool", ["read_file", "get_webpage"])
def test_unplanned_query_tool_blocked_when_expansion_off(query_tool):
    d = ig.Defense(allow_query_expansion=False)
    plan = _plan("run_sql_query")  # query tool NOT in plan
    assert d.allow_call(query_tool, {}, plan) is False


# ----------------------------- (d) planned tool always allowed regardless of expansion ---
@pytest.mark.parametrize("expansion", [True, False])
def test_planned_tool_always_allowed_regardless_of_expansion(expansion):
    d = ig.Defense(allow_query_expansion=expansion)
    # A command tool that IS in the plan is allowed in both expansion modes.
    plan = _plan("write_file", "read_file")
    assert d.allow_call("write_file", {"name": "out", "content": "x"}, plan) is True
    assert d.allow_call("read_file", {"name": "out"}, plan) is True


# ----------------------------------------------------- empty / missing plan ---
def test_empty_plan_blocks_command_and_gates_query_on_expansion():
    plan = _plan()  # nothing planned
    d_on = ig.Defense(allow_query_expansion=True)
    d_off = ig.Defense(allow_query_expansion=False)
    # command tool blocked either way
    assert d_on.allow_call("run_sql_query", {}, plan) is False
    assert d_off.allow_call("run_sql_query", {}, plan) is False
    # query tool gated purely on expansion
    assert d_on.allow_call("read_file", {}, plan) is True
    assert d_off.allow_call("read_file", {}, plan) is False


def test_missing_planned_tools_key_defaults_to_empty_set():
    # allow_call uses plan.get("planned_tools", set()); a plan lacking the key must not crash.
    d = ig.Defense(allow_query_expansion=False)
    assert d.allow_call("run_sql_query", {}, {}) is False
    assert d.allow_call("read_file", {}, {}) is False  # expansion off -> query also blocked


# ----------------------------------------------------------- pure helpers ---
def test_tool_docs_serializes_each_tool():
    tools = [
        {"name": "run_sql_query", "description": "run sql",
         "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}},
        {"name": "read_file", "description": "read a file", "parameters": {}},
    ]
    docs = ig.Defense._tool_docs(tools)
    assert "<function-1>" in docs and "</function-1>" in docs
    assert "<function-2>" in docs and "</function-2>" in docs
    assert "run_sql_query" in docs and "read_file" in docs
    assert "run sql" in docs  # description carried through


def test_tool_docs_tolerates_missing_description_and_parameters():
    docs = ig.Defense._tool_docs([{"name": "finish"}])
    assert "finish" in docs
    assert "<function-1>" in docs
