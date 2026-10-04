"""Live smoke tests for the real BGE embedding backend (make_bge_embed_fn -> :8001).

@pytest.mark.live; skipped when the BGE server is down (via the live_bge fixture). These assert
ONLY that embeddings come back well-formed (consistent dimensionality, finite, cosine in [-1, 1])
and that nothing crashes. NO claim about MELON flagging a real injection.
"""
import numpy as np
import pytest

from testlib import pathsetup

mel = pathsetup.load_impl("MELON")
adapter_mod = pathsetup.load_impl("MELON", "agentdojo_adapter.py", "_melon_agentdojo_adapter")

pytestmark = pytest.mark.live


def test_bge_embed_returns_consistent_dim_vectors(live_bge):
    embed = adapter_mod.make_bge_embed_fn()
    v1 = embed("send_money(recipient = hacker, amount = 100)")
    v2 = embed("read_file(name = report.txt)")
    assert len(v1) > 0
    assert len(v1) == len(v2)  # consistent dimensionality across inputs
    assert all(np.isfinite(v1)) and all(np.isfinite(v2))


def test_bge_cosine_in_range_and_no_crash(live_bge):
    embed = adapter_mod.make_bge_embed_fn()
    a = np.asarray(embed("run_sql_query(database = db, command = DROP TABLE t)"), dtype=float)
    b = np.asarray(embed("send_direct_message(recipient = Alice, body = hello)"), dtype=float)
    s = mel.cosine_sim(a, b)
    assert -1.0 - 1e-6 <= s <= 1.0 + 1e-6


def test_bge_identical_text_cosine_near_one(live_bge):
    # Same string embedded twice should be (near-)identical -> cosine ~ 1.0. Code-correctness
    # of the embed roundtrip, NOT an efficacy claim.
    embed = adapter_mod.make_bge_embed_fn()
    text = "transfer_funds(recipient = US133000000121212121212, amount = 100)"
    a = np.asarray(embed(text), dtype=float)
    b = np.asarray(embed(text), dtype=float)
    assert mel.cosine_sim(a, b) == pytest.approx(1.0, abs=1e-3)
