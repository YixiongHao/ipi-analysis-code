"""Live smoke test for FIDES.

FIDES's security mechanism (taint tracking + deterministic per-tool P-T policy) makes
NO model or embedding calls — see defense.py ("FIDES makes NO LLM calls") and
agentdojo_adapter.FidesMonitor (a pure BasePipelineElement). The only live dependency in
the AgentDojo wiring is the *agent* LLM (QwenLLM), which is the model under defense, not
part of the defense's decision. There is therefore nothing in the FIDES defense code to
exercise against a live server, so this suite is intentionally a single skip placeholder
(per the task brief: don't fabricate live tests).
"""
import pytest


@pytest.mark.skip(reason="FIDES has no live model/embedding dependency: its allow/deny "
                         "decision is a deterministic function of message role + tool policy "
                         "(no LLM/BGE calls). Nothing to smoke-test against a server.")
def test_fides_has_no_live_dependency():
    pass
