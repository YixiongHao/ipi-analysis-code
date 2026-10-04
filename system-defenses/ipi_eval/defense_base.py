"""DefenseAdapter interface + rollout context.

Every defense adapter delegates to its existing system-defenses/<Name>/impl/defense.py
core methods so the transformations match the AgentDojo versions at the method level.
The engine calls the hooks below at fixed seams; full-loop defenses (CaMeL, IPIGuard)
instead set drives_loop()=True and implement run_loop().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class RolloutContext:
    behavior: dict                      # behavior_doc from behaviors.get_behavior()
    system_prompt: str
    user_request: str
    tools: list[dict]                   # OpenAI function-calling tool list
    command_tools: set[str]             # state-mutating tool names (for IPIGuard etc.)
    target_client: Any                  # ipi_arena_bench.llm_client.LLMClient (defended model)
    worldsim_client: Any                # LLMClient for tool-response simulation
    worldsim_prompt: str                # prebuilt world-sim system prompt
    tool_policies: dict[str, str] = field(default_factory=dict)  # tool name -> "P-T" | "P-F or P-T" (FIDES)
    prefill_end_index: int | None = None  # index of the injected (last) prefill tool message in messages
    messages: list[dict] = field(default_factory=list)   # running conversation
    generated: list[dict] = field(default_factory=list)  # assistant turns authored this rollout
    max_steps: int = 5
    scratch: dict = field(default_factory=dict)          # adapter-private state

    def tool_description(self, name: str) -> str:
        for t in self.tools:
            if t["function"]["name"] == name:
                return t["function"].get("description", "")
        return ""


# review_tool_call returns (allow, new_args, replacement_output):
#   allow=False           -> tool not executed; replacement_output (or a default) used as result
#   new_args not None     -> execute with these args instead (e.g. Firewalls minimizer)
ReviewResult = tuple[bool, dict | None, str | None]


class DefenseAdapter:
    name: str = "base"

    def on_rollout_start(self, ctx: RolloutContext) -> None:
        """Called once before the loop. Use to plan / seed state from task context."""

    def preprocess_prefill(self, ctx: RolloutContext) -> None:
        """Act on the replayed prefill (esp. the injected tool message at
        ctx.prefill_end_index) before the rollout begins. Default: do nothing.
        I/O-transform defenses override to sanitize in place. Mutate ctx.messages
        in place (it's already a fresh copy); fail-open on any error."""

    def preprocess_agent_input(self, messages: list[dict], ctx: RolloutContext) -> list[dict]:
        """Seam: rewrite the message list the agent sees *for this generation only*, before
        the original-run LLM call. Default: identity. Return a (possibly new) list; do NOT
        mutate ctx.messages — the returned list is transient and not persisted. Used by the
        repeat-prompt family (MELON-Aug re-appends the user request here)."""
        return messages

    def transform_tool_output(self, content: str, ctx: RolloutContext) -> str:
        """Seam (a): rewrite a tool observation before the model sees it (sanitizers)."""
        return content

    def review_tool_call(self, tool_name: str, args: dict, ctx: RolloutContext) -> ReviewResult:
        """Seam (b): inspect/block/rewrite a proposed tool call before execution."""
        return (True, None, None)

    def inspect_assistant(self, message: dict, ctx: RolloutContext) -> str | None:
        """Seam (c): inspect a generated assistant message. Return 'abort' to stop the rollout."""
        return None

    def drives_loop(self) -> bool:
        return False

    def run_loop(self, ctx: RolloutContext) -> dict:
        """Full-loop defenses (CaMeL, IPIGuard) implement this.

        Must drive the rollout (executing tools via ctx.worldsim_*) and return
        {"messages": [...], "generated_messages": [...]}.
        """
        raise NotImplementedError
