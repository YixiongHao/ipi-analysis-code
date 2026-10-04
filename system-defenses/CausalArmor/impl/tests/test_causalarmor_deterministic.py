"""Deterministic tests for the CausalArmor defense core (impl/defense.py).

We test CODE CORRECTNESS, never efficacy:
  * is_privileged  -> the keyword gate (verbs True, read/get/list False, case-insensitive).
  * serialize_action -> canonical `tool(sorted-json)` shape, stable across arg order.
  * analyze        -> the LOO attribution math (Eqs. 2/5/6) with `_score_batch` stubbed
                      to KNOWN logprobs, so Delta/Delta_norm/Delta_u and the flag set
                      B(tau) = {S : Delta_bar_S > Delta_bar_U - tau} are hand-computable.
                      We sweep tau (smaller tau flags fewer) and toggle length_normalize
                      and assert the boundary shifts exactly as the formula dictates.
  * detect         -> thin wrapper routes to analyze; returns (flagged, score) shape.

`_score_batch` is the LLM seam; stubbing it makes analyze fully deterministic AND avoids
loading the real tokenizer (analyze itself does no separate tokenization — all tokenizing
is inside _score_batch / _render_prefix, both bypassed by the stub).
"""
import pytest

from testlib import fixtures, pathsetup

ca = pathsetup.load_impl("CausalArmor")


# =============================================================== is_privileged ===
class TestIsPrivileged:
    @pytest.mark.parametrize("name", [
        "send_direct_message", "transfer_money", "make_payment", "create_event",
        "update_record", "delete_file", "write_file", "post_message", "send_email",
        "schedule_meeting", "share_document", "remove_user", "move_file", "execute_bash",
        "add_contact", "set_config",
    ])
    def test_privileged_verbs_true(self, name):
        assert ca.Defense().is_privileged(name) is True

    @pytest.mark.parametrize("name", [
        "read_file", "get_webpage", "list_directory", "get_balance", "read_inbox",
        "list_files", "search", "lookup_contact",
    ])
    def test_read_verbs_false(self, name):
        assert ca.Defense().is_privileged(name) is False

    def test_case_insensitive(self):
        d = ca.Defense()
        assert d.is_privileged("SEND_EMAIL") is True
        assert d.is_privileged("Write_File") is True
        assert d.is_privileged("GET_WEBPAGE") is False

    def test_none_and_empty(self):
        d = ca.Defense()
        assert d.is_privileged("") is False
        assert d.is_privileged(None) is False

    def test_real_fixture_tool_names(self):
        d = ca.Defense()
        # IPI fixture decisive call: run_sql_query is NOT a privileged keyword by name
        # (no state-changing verb substring), but write_file / send_direct_message are.
        assert d.is_privileged("run_sql_query") is False
        assert d.is_privileged("write_file") is True
        assert d.is_privileged("send_direct_message") is True
        assert d.is_privileged("read_file") is False
        assert d.is_privileged("list_directory") is False
        assert d.is_privileged("get_webpage") is False


# ============================================================== serialize_action ===
class TestSerializeAction:
    def test_shape_and_sorted_keys(self):
        s = ca.Defense.serialize_action("send_email", {"to": "a@x.com", "body": "hi"})
        assert s == 'send_email({"body": "hi", "to": "a@x.com"})'

    def test_stable_across_arg_order(self):
        a = ca.Defense.serialize_action("write_file", {"path": "/p", "content": "c"})
        b = ca.Defense.serialize_action("write_file", {"content": "c", "path": "/p"})
        assert a == b

    def test_empty_args(self):
        assert ca.Defense.serialize_action("list_dir", {}) == "list_dir({})"

    def test_unicode_preserved(self):
        s = ca.Defense.serialize_action("post", {"msg": "café"})
        assert "café" in s and "\\u" not in s   # ensure_ascii=False

    def test_is_staticmethod_callable_unbound(self):
        # serialize_action is a @staticmethod -> callable off the class without an instance.
        assert ca.Defense.serialize_action("x", {"k": 1}) == 'x({"k": 1})'


# ===================================================================== analyze ===
# Stub `_score_batch` so analyze sees KNOWN (sum_logprob, n_tokens) for each ablation.
# analyze builds ablate = [None, user_index, *tool_indices]; we key our canned returns by
# the ablate entry so the order is irrelevant to the test's correctness.

# Three-span prefill: roles [system,user,asst,tool,asst,tool,tool]; user=1; tool spans=3,5,6.
# Built from the IPI fixture (its system/user/assistant turns + injected review as span 6) with
# two benign tool results added, so the attribution math has benign spans to compare against.
def _three_span_prefill() -> list[dict]:
    system, user, assistant = fixtures.ipi_prefill_messages(include_injection=False)
    injected = fixtures.ipi_prefill_messages()[-1]
    return [system, user, assistant,
            {"role": "tool", "content": "[]"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "content": '{"sentiment": "positive"}'},
            injected]


N_Y = 10
FULL_LP = -20.0
# Per-ablation summed logprob of the action under the ablated context.
# Delta_X = FULL_LP - score[X];  Delta_bar_X = Delta_X / N_Y (when length_normalize).
ABLATED = {
    None: FULL_LP,   # full context (its n_tokens defines N_Y)
    1: -25.0,        # user ablation  -> Delta_u   = 5  -> norm 0.5
    3: -22.0,        # benign span    -> Delta=2   -> norm 0.2
    5: -23.0,        # benign span    -> Delta=3   -> norm 0.3
    6: -32.0,        # injected span  -> Delta=12  -> norm 1.2  (> 0.5 => flagged at tau=0)
}


def _fake_score_batch(messages, action, ablate):
    # Return (sum_logprob, n_tokens) per ablate entry, in the order given.
    return [(ABLATED[idx], N_Y) for idx in ablate]


def _stubbed(monkeypatch, **kwargs):
    d = ca.Defense(**kwargs)
    monkeypatch.setattr(d, "_score_batch", _fake_score_batch)
    return d


class TestAnalyzeMath:
    def test_fields_and_flag_at_tau0(self, monkeypatch):
        d = _stubbed(monkeypatch, tau=0.0, length_normalize=True)
        msgs = _three_span_prefill()
        action = "write_file({})"
        attr = d.analyze(action, msgs)

        assert attr.action == action
        assert attr.n_action_tokens == N_Y
        assert attr.user_index == 1
        # Delta_u = -20 - (-25) = 5 ; normed 0.5
        assert attr.delta_u == pytest.approx(5.0)
        assert attr.delta_u_norm == pytest.approx(0.5)
        # Three tool spans at indices 3,5,6.
        assert [s.index for s in attr.spans] == [3, 5, 6]
        by_idx = {s.index: s for s in attr.spans}
        assert by_idx[3].delta == pytest.approx(2.0)
        assert by_idx[3].delta_norm == pytest.approx(0.2)
        assert by_idx[5].delta == pytest.approx(3.0)
        assert by_idx[5].delta_norm == pytest.approx(0.3)
        assert by_idx[6].delta == pytest.approx(12.0)
        assert by_idx[6].delta_norm == pytest.approx(1.2)
        # span role recorded as the message role.
        assert all(s.role == "tool" for s in attr.spans)
        # Flag rule B(0) = {S : 1.2 > 0.5}, {0.2>0.5? no}, {0.3>0.5? no} -> only idx 6.
        assert attr.flagged == [6]
        # max_span_norm property.
        assert attr.max_span_norm == pytest.approx(1.2)

    def test_tau_sweep_smaller_flags_fewer(self, monkeypatch):
        # Boundary: flag iff delta_norm > delta_u_norm - tau, i.e. delta_norm > 0.5 - tau.
        # Span norms: {3:0.2, 5:0.3, 6:1.2}.
        msgs = _three_span_prefill()
        action = "write_file({})"

        # tau = -0.8 -> threshold 1.3 -> nothing (even 1.2 not > 1.3).
        d = _stubbed(monkeypatch, tau=-0.8)
        assert d.analyze(action, msgs).flagged == []

        # tau = 0.0 -> threshold 0.5 -> only 1.2.
        d = _stubbed(monkeypatch, tau=0.0)
        assert d.analyze(action, msgs).flagged == [6]

        # tau = 0.25 -> threshold 0.25 -> 0.3 and 1.2 (0.2 not > 0.25).
        d = _stubbed(monkeypatch, tau=0.25)
        assert d.analyze(action, msgs).flagged == [5, 6]

        # tau = 0.4 -> threshold 0.1 -> all three (0.2,0.3,1.2 all > 0.1).
        d = _stubbed(monkeypatch, tau=0.4)
        assert d.analyze(action, msgs).flagged == [3, 5, 6]

    def test_strict_greater_than_at_boundary(self, monkeypatch):
        # Eq. 5 uses STRICT > : a span exactly AT the threshold must NOT flag.
        # delta_u_norm = 0.5. Pick tau so threshold == one span's norm exactly: 0.5 - tau = 0.3
        # => tau = 0.2; span idx 5 has norm 0.3 == threshold -> excluded; idx 6 (1.2) included.
        d = _stubbed(monkeypatch, tau=0.2)
        attr = d.analyze("write_file({})", _three_span_prefill())
        assert 5 not in attr.flagged          # exactly-equal span NOT flagged (> not >=)
        assert attr.flagged == [6]

    def test_length_normalize_off_uses_raw_deltas(self, monkeypatch):
        # With length_normalize off, comparison is on RAW deltas: flag iff Delta_S > Delta_U - tau.
        # Delta_U = 5 ; span raw deltas {3:2, 5:3, 6:12}. At tau=0 only 12 > 5 -> idx 6.
        d = _stubbed(monkeypatch, tau=0.0, length_normalize=False)
        attr = d.analyze("write_file({})", _three_span_prefill())
        assert attr.delta_u == pytest.approx(5.0)
        assert attr.delta_u_norm == pytest.approx(5.0)        # norm == identity when off
        assert {s.index: s.delta_norm for s in attr.spans} == {3: 2.0, 5: 3.0, 6: 12.0}
        assert attr.flagged == [6]
        # Boundary shift: raw threshold is 5 (not 0.5). A tau that flagged idx 5 when normed
        # (tau=0.25 -> threshold 0.25) does NOT when raw (threshold 5 - 0.25 = 4.75 > 3).
        d2 = _stubbed(monkeypatch, tau=0.25, length_normalize=False)
        assert d2.analyze("write_file({})", _three_span_prefill()).flagged == [6]

    def test_n_action_tokens_floored_to_one(self, monkeypatch):
        # n_y = max(n_y, 1): a zero-token action must not divide-by-zero; norm uses 1.
        d = ca.Defense(tau=0.0, length_normalize=True)
        monkeypatch.setattr(d, "_score_batch",
                            lambda m, a, ab: [(ABLATED[i], 0) for i in ab])
        attr = d.analyze("write_file({})", _three_span_prefill())
        assert attr.n_action_tokens == 1                       # floored
        assert attr.delta_u_norm == pytest.approx(5.0)         # divided by 1

    def test_no_user_message(self, monkeypatch):
        # No user span -> delta_u = -inf, delta_u_norm = -inf; every positive-delta span flags.
        msgs = [{"role": "system", "content": "s"},
                {"role": "tool", "content": "t0"},
                {"role": "tool", "content": "t1"}]
        d = ca.Defense(tau=0.0)
        # ablate = [None, 1, 2] (no user message -> user_index None). Known scores per entry.
        scores = {None: -20.0, 1: -22.0, 2: -25.0}
        monkeypatch.setattr(d, "_score_batch",
                            lambda m, a, ab: [(scores[i], 10) for i in ab])
        attr = d.analyze("write_file({})", msgs)
        assert attr.user_index is None
        assert attr.delta_u == float("-inf")
        assert attr.delta_u_norm == float("-inf")
        assert attr.flagged == [1, 2]      # both spans > -inf

    def test_no_tool_spans_no_flags(self, monkeypatch):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        d = ca.Defense(tau=0.0)
        monkeypatch.setattr(d, "_score_batch",
                            lambda m, a, ab: [(-20.0, 10) for _ in ab])
        attr = d.analyze("write_file({})", msgs)
        assert attr.spans == []
        assert attr.flagged == []
        assert attr.max_span_norm == float("-inf")


# ====================================================================== detect ===
class TestDetect:
    def test_routes_to_analyze_and_shape(self, monkeypatch):
        d = _stubbed(monkeypatch, tau=0.0)
        msgs = _three_span_prefill()
        ctx = {"messages": msgs, "action": "write_file({})", "span_index": 6}
        flagged, score = d.detect("payload", ctx)
        assert flagged is True
        # score = Delta_bar_S - Delta_bar_U = 1.2 - 0.5 = 0.7
        assert score == pytest.approx(0.7)

    def test_specific_span_not_flagged(self, monkeypatch):
        d = _stubbed(monkeypatch, tau=0.0)
        ctx = {"messages": _three_span_prefill(),
               "action": "write_file({})", "span_index": 3}
        flagged, score = d.detect("payload", ctx)
        assert flagged is False
        assert score == pytest.approx(0.2 - 0.5)   # -0.3

    def test_default_span_is_max_attribution(self, monkeypatch):
        # No span_index -> picks the max delta_norm span (idx 6, norm 1.2).
        d = _stubbed(monkeypatch, tau=0.0)
        ctx = {"messages": _three_span_prefill(), "action": "write_file({})"}
        flagged, score = d.detect("payload", ctx)
        assert flagged is True
        assert score == pytest.approx(0.7)

    def test_no_spans_returns_false_neg_inf(self, monkeypatch):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        d = ca.Defense(tau=0.0)
        monkeypatch.setattr(d, "_score_batch", lambda m, a, ab: [(-20.0, 10) for _ in ab])
        flagged, score = d.detect("payload", {"messages": msgs, "action": "write_file({})"})
        assert flagged is False
        assert score == float("-inf")
