"""Shared test library for the IPI system-defense test suites.

Each defense's suite lives in ``system-defenses/<Name>/impl/tests/`` and imports
from here:

    from testlib import fixtures, stubs, pathsetup

``pathsetup`` makes ``ipi_eval``, ``agentdojo``, ``ipi_arena_bench`` and a given
defense's ``impl/`` importable; ``fixtures`` serves frozen samples from both the
IPI-transcript and AgentDojo datasets; ``stubs`` provides deterministic stand-ins
for the LLM / embedding backends plus server-reachability probes.

Guiding rule for every suite: assert what the *code* does (control flow, parsing,
routing, deterministic policy/threshold math). Never assert defense *efficacy*
(attack blocked / injection removed / ASR lowered) — that is non-deterministic and
measured separately by ipi_eval.
"""
from . import pathsetup  # noqa: F401  (ensures sys.path is set up on import)
