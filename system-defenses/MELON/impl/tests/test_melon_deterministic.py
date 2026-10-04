"""Deterministic unit tests for the MELON detector core (impl/defense.py).

Tests CODE CORRECTNESS only: the tool-call rendering / arg-filtering, the masked-output
consolidation framing, cosine math, and the embedding-bank comparator's threshold logic.
The embed_fn is always a deterministic STUB so the similarity/threshold path is what is
exercised — never a real model and never an efficacy claim ("MELON caught a real attack").
"""
import numpy as np
import pytest

from testlib import fixtures, pathsetup, stubs

mel = pathsetup.load_impl("MELON")


# --------------------------------------------------------------- tool_call_to_text ---
def test_tool_call_to_text_filtered_send_money():
    # send_money is in _ARG_FILTERS -> only recipient + amount survive; memo dropped.
    txt = mel.tool_call_to_text(
        "send_money", {"recipient": "hacker", "amount": 100, "memo": "x"}
    )
    assert txt == "send_money(recipient = hacker, amount = 100)"


def test_tool_call_to_text_filtered_send_email_keeps_only_recipients():
    # send_email filter == ["recipients"]; body is dropped even when present.
    txt = mel.tool_call_to_text("send_email", {"body": "hi", "recipients": "a@b.com"})
    assert txt == "send_email(recipients = a@b.com)"


def test_tool_call_to_text_filter_renders_in_input_order_not_filter_order():
    # The filter is a membership test; surviving args keep the *input dict* order.
    txt = mel.tool_call_to_text("send_money", {"amount": 100, "recipient": "h", "memo": "m"})
    assert txt == "send_money(amount = 100, recipient = h)"
    # ...but only the allowed keys survive regardless of order.
    assert "memo" not in txt


def test_tool_call_to_text_generic_tool_renders_all_args():
    # run_sql_query is NOT in _ARG_FILTERS -> every arg is rendered.
    txt = mel.tool_call_to_text("run_sql_query", {"database": "db", "command": "DROP TABLE t"})
    assert txt == "run_sql_query(database = db, command = DROP TABLE t)"


@pytest.mark.parametrize("args", [{}, None])
def test_tool_call_to_text_empty_or_none_args(args):
    assert mel.tool_call_to_text("noop", args) == "noop()"


def test_tool_call_to_text_on_real_ipi_decisive_call():
    # The fixture's malicious IPI call (execute_sql TRUNCATE TABLE) -> generic rendering, all args.
    call = fixtures.ipi_decisive_call()
    txt = mel.tool_call_to_text(call["function"], call["args"])
    assert txt == "execute_sql(query = TRUNCATE TABLE product_reviews)"


def test_tool_call_to_text_on_real_agentdojo_malicious_call():
    # The frozen AgentDojo malicious call: send_direct_message is generic -> all args render.
    mc = fixtures.agentdojo_slack()["malicious_call"]
    txt = mel.tool_call_to_text(mc["function"], mc["args"])
    assert txt.startswith("send_direct_message(")
    assert "recipient = Alice" in txt
    assert "body = Check out this link: www.secure-systems-252.com" in txt


# --------------------------------------------------------------- tool_calls_to_texts ---
@pytest.mark.parametrize("empty", [None, []])
def test_tool_calls_to_texts_empty_is_sentinel(empty):
    assert mel.tool_calls_to_texts(empty) == ["No tool calls"]


def test_tool_calls_to_texts_renders_each_call():
    # Accepts (name, args) tuples; renders one string per call.
    calls = [("run_sql_query", {"command": "X"}), ("send_money", {"recipient": "r", "amount": 5})]
    out = mel.tool_calls_to_texts(calls)
    assert out == ["run_sql_query(command = X)", "send_money(recipient = r, amount = 5)"]


def test_tool_calls_to_texts_dict_and_object_forms():
    # dict {"function","args"} form
    out_dict = mel.tool_calls_to_texts([{"function": "f", "args": {"a": 1}}])
    assert out_dict == ["f(a = 1)"]

    # object-with-.function/.args form (AgentDojo FunctionCall-like)
    class _FC:
        def __init__(self, function, args):
            self.function, self.args = function, args

    out_obj = mel.tool_calls_to_texts([_FC("g", {"b": 2})])
    assert out_obj == ["g(b = 2)"]


# --------------------------------------------------------------- consolidate_tool_outputs ---
def test_consolidate_tool_outputs_headers_separators_and_order():
    blob = mel.consolidate_tool_outputs([("read_file", "CONTENT_A"), ("get_weather", "CONTENT_B")])
    sep = "=" * 50
    # Each block carries the function header and its content, wrapped in === separators.
    assert "function: read_file" in blob
    assert "function: get_weather" in blob
    assert "CONTENT_A" in blob and "CONTENT_B" in blob
    assert blob.count(sep) >= 2
    # Ordering preserved: read_file block precedes get_weather block.
    assert blob.index("function: read_file") < blob.index("function: get_weather")
    # First block starts with a separator + the header framing.
    assert blob.startswith(f"{sep}\n\nfunction: read_file\n\nCONTENT_A\n\n{sep}")


def test_consolidate_tool_outputs_empty_is_empty_string():
    assert mel.consolidate_tool_outputs([]) == ""


# --------------------------------------------------------------- cosine_sim ---
def test_cosine_sim_identical_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert mel.cosine_sim(v, v) == pytest.approx(1.0)


def test_cosine_sim_orthogonal_is_zero():
    assert mel.cosine_sim(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_sim_opposite_is_negative_one():
    assert mel.cosine_sim(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_cosine_sim_within_range():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.standard_normal(8)
        b = rng.standard_normal(8)
        s = mel.cosine_sim(a, b)
        assert -1.0 - 1e-9 <= s <= 1.0 + 1e-9


# --------------------------------------------------------------- MelonDetector banks ---
def test_add_masked_calls_dedups_and_skips_sentinel():
    det = mel.MelonDetector(embed_fn=stubs.stub_embed_fn, sim_threshold=0.8)
    det.add_masked_calls(["call_x", "call_x", "No tool calls", "call_y"])
    # call_bank is the dedup set; "No tool calls" sentinel excluded.
    assert det.call_bank == {"call_x", "call_y"}
    # emb_bank is parallel and the same length as the dedup set.
    assert len(det.emb_bank) == 2
    assert all(isinstance(e, np.ndarray) for e in det.emb_bank)


def test_check_flags_identical_text_via_stub_cosine_one():
    # stub_embed_fn: identical text -> cosine 1.0 > theta -> flagged.
    det = mel.MelonDetector(embed_fn=stubs.stub_embed_fn, sim_threshold=0.8)
    det.add_masked_calls(["send_money(recipient = hacker, amount = 100)"])
    flagged, sim = det.check(["send_money(recipient = hacker, amount = 100)"])
    assert flagged is True
    assert sim == pytest.approx(1.0)


def test_check_does_not_flag_disjoint_text():
    # stub_embed_fn: disjoint token sets -> cosine 0.0 < theta -> not flagged.
    det = mel.MelonDetector(embed_fn=stubs.stub_embed_fn, sim_threshold=0.8)
    det.add_masked_calls(["read_file(name = report.txt)"])
    flagged, sim = det.check(["transfer money offshore now urgently please"])
    assert flagged is False
    assert sim < 0.8


def test_check_empty_inputs_return_false():
    det = mel.MelonDetector(embed_fn=stubs.stub_embed_fn, sim_threshold=0.8)
    # No masked calls cached yet.
    assert det.check(["anything"]) == (False, 0.0)
    # All-sentinel original list (treated as no originals).
    det.add_masked_calls(["x"])
    assert det.check(["No tool calls"]) == (False, 0.0)


def test_check_threshold_is_strict_greater_than():
    # Construct an embed map with cosine EXACTLY at theta=0.8 between A and B.
    #   a = (1,0); b = (0.8, 0.6)  -> cos = 0.8 (b is unit length).
    emb = {"A": [1.0, 0.0], "B": [0.8, 0.6]}

    def emap(t):
        return emb[t]

    # At exactly theta -> NOT flagged (convention is sim > theta, strict).
    det_at = mel.MelonDetector(embed_fn=emap, sim_threshold=0.8)
    det_at.add_masked_calls(["B"])
    flagged_at, sim_at = det_at.check(["A"])
    assert sim_at == pytest.approx(0.8)
    assert flagged_at is False

    # Lower theta just under the pair's sim -> flagged.
    det_below = mel.MelonDetector(embed_fn=emap, sim_threshold=0.79)
    det_below.add_masked_calls(["B"])
    flagged_below, _ = det_below.check(["A"])
    assert flagged_below is True


def test_step_is_add_then_check_and_returns_tuple():
    det = mel.MelonDetector(embed_fn=stubs.stub_embed_fn, sim_threshold=0.8)
    text = "run_sql_query(command = DROP TABLE t)"
    # Single-step attack: masked run surfaced the call, original run made it -> caught immediately.
    flagged, sim = det.step(original_call_texts=[text], masked_call_texts=[text])
    assert flagged is True
    assert sim == pytest.approx(1.0)
    # step cached the masked call into the banks.
    assert text in det.call_bank


def test_step_no_match_when_disjoint():
    det = mel.MelonDetector(embed_fn=stubs.stub_embed_fn, sim_threshold=0.8)
    flagged, _ = det.step(
        original_call_texts=["read_file(name = summary.txt)"],
        masked_call_texts=["send_email(recipients = friend@x.com)"],
    )
    assert flagged is False


# --------------------------------------------------------------- build_few_shot ---
def test_build_few_shot_shape_and_factory_use():
    seen = []

    def fc_factory(function, args, id):
        seen.append((function, args, id))
        return {"_fc": function, "args": args, "id": id}

    msgs = mel.build_few_shot(fc_factory)
    # The two priming samples: read_inbox (no instructions) + read_file (instructions -> create_report).
    assert isinstance(msgs, list)
    assert len(msgs) == 9
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "tool", "assistant", "user", "assistant", "tool", "assistant", "tool"]
    # factory was invoked for each tool call (read_inbox x2, read_file x2, create_report x2).
    fns = [s[0] for s in seen]
    assert fns.count("read_inbox") == 2
    assert fns.count("read_file") == 2
    assert fns.count("create_report") == 2


def test_task_neutral_prompt_is_nonempty_string():
    assert isinstance(mel.TASK_NEUTRAL_PROMPT, str)
    assert "read_file" in mel.TASK_NEUTRAL_PROMPT
    assert "random.txt" in mel.TASK_NEUTRAL_PROMPT
