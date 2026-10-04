# CaMeL defense — modular, scaffold-callable wrapper.
#
# CaMeL (Debenedetti et al., 2503.18813) is a *planning + information-flow* defense,
# NOT an I/O transform or a detector. It cannot be expressed as
# `transform_tool_output()` / `detect()` over a single message, because it changes how
# the whole agent runs: a Privileged LLM writes restricted Python expressing the user
# query, a custom interpreter executes it while tracking a data-flow graph and
# capabilities, and a Quarantined LLM (no tool access) parses untrusted data into typed
# structured output. Security policies are enforced before each side-effecting tool call.
#
# Therefore the modular surface CaMeL exposes is `build_agentdojo_pipeline(...)`, which
# returns a composed AgentDojo `AgentPipeline`. The heavy lifting lives in the reference
# repo (../camel-prompt-injection); this module is glue only.
#
# --- Generic scaffold use -----------------------------------------------------------
# The defense is a whole agent loop, so you drive it through AgentDojo's pipeline:
#
#     from defense import Defense
#     d = Defense(model="Qwen3-32B", variant="camel")        # or "camel+secpol"
#     pipeline = d.build_agentdojo_pipeline(suite_name="banking")
#     # then run via agentdojo.benchmark.benchmark_suite_with_injections(pipeline, ...)
#
# Must be imported/run inside the reference repo's uv env (it provides `camel`,
# `pydantic_ai`, and a working `agentdojo`). See run_agentdojo.py for env setup.

from __future__ import annotations

VARIANTS = ("undefended", "camel", "camel+secpol")


class Defense:
    name = "camel"

    def __init__(
        self,
        model: str = "Qwen3-32B",
        variant: str = "camel",
        attack_name: str = "important_instructions",
        q_llm: str | None = None,
        strict: bool = False,
    ) -> None:
        """CaMeL defense.

        Args:
            model: agent model name as served by the local vLLM endpoint (P-LLM, and
                Q-LLM unless `q_llm` is set). Paper used frontier models; we standardize
                on the local Qwen3-32B.
            variant: one of VARIANTS. "camel" = isolation + data-flow only (no security
                policies, `ADNoSecurityPolicyEngine`); "camel+secpol" = additionally
                enforce per-suite security policies via the replay path; "undefended" =
                native tool-calling baseline (no CaMeL).
            attack_name: AgentDojo attack to defend against (paper headline:
                `important_instructions`). Only used to locate replay logs for secpol.
            q_llm: optional separate Quarantined-LLM model id; defaults to `model`.
            strict: use the interpreter's STRICT data-flow eval mode (control-dependency
                edges added). Paper default is NORMAL; STRICT is an ablation.
        """
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.model = model
        self.variant = variant
        self.attack_name = attack_name
        self.q_llm = q_llm
        self.strict = strict

    def build_agentdojo_pipeline(self, suite_name: str):
        """Return a composed AgentDojo AgentPipeline for this variant + suite."""
        from agentdojo_adapter import build_pipeline

        return build_pipeline(
            suite_name=suite_name,
            model=self.model,
            variant=self.variant,
            attack_name=self.attack_name,
            q_llm=self.q_llm,
            strict=self.strict,
        )
