"""Scaffold-callable wrapper around the IPIGuard reference implementation.

IPIGuard is a *planning / control-flow* defense: it builds a Tool Dependency Graph
(TDG) from the trusted user request + tool list up front, then constrains execution to
that plan — new *command* (state-mutating) tool calls introduced by tool outputs are
never executed; only read-only *query* tools may be added at run time (Node Expansion).

This module exposes the two planning-archetype methods from the skill interface:

    d = Defense()                                  # paper defaults baked in
    plan = d.plan(user_request, tools)             # build the TDG
    ok   = d.allow_call("send_money", args, plan)  # would IPIGuard let this call run?

`tools` is a list of dicts: {"name", "description", "parameters"} where "parameters" is
a JSON-schema dict (e.g. agentdojo `tool.parameters.model_json_schema()`).

The full end-to-end defense (argument estimation, node expansion, fake tool invocation)
runs through AgentDojo via `agentdojo_adapter.build_ipiguard_pipeline()`; this wrapper
exposes the *decision surface* (plan + admissibility) for generic scaffolds and for the
IPI post-hoc check, reusing the fork's exact prompt/parsing/whitelist as ground truth.
"""
from __future__ import annotations

import json


class Defense:
    name = "ipiguard"

    def __init__(
        self,
        model: str = "Qwen3-32B",
        base_url: str = "http://localhost:8000/v1",
        temperature: float = 0.0,
        allow_query_expansion: bool = True,
    ):
        # paper defaults: temperature 0 for replicability; Node Expansion ON (allows
        # read-only query tools to be added during execution).
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.allow_query_expansion = allow_query_expansion
        self._client = None

    # -- lazy helpers ---------------------------------------------------------
    def _client_(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(api_key="EMPTY", base_url=self.base_url)
        return self._client

    @staticmethod
    def _query_tools() -> set[str]:
        import _fork_env  # noqa: F401  (sets fork sys.path)
        from agentdojo.default_suites.v1.tools.tool_white_list import whitelist

        return set(whitelist)

    @staticmethod
    def _tool_docs(tools: list[dict]) -> str:
        docs = ""
        for i, t in enumerate(tools, start=1):
            docs += f"<function-{i}>\n"
            docs += json.dumps(
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}),
                },
                indent=4,
            )
            docs += f"\n</function-{i}>\n\n"
        return docs

    # -- planning archetype ---------------------------------------------------
    def plan(self, user_request: str, tools: list[dict], ctx: dict | None = None) -> dict:
        """Construct the Tool Dependency Graph for the trusted user request.

        Returns {"tool_calls": [...], "planned_tools": set, "pending": [...]} where each
        tool_call has function_name/args/depends_on (the fork's TDG schema).
        """
        import _fork_env  # noqa: F401
        from agentdojo.agent_pipeline.llms.ipiguard_llm import (
            OpenAIConstructLLM,
            chat_completion_request,
            get_pre_plan,
        )

        client = self._client_()
        tool_docs = self._tool_docs(tools)

        # Step 1: pre-plan (task understanding) — same as the fork's construct query().
        pre = get_pre_plan(
            client=client,
            model=self.model,
            temperature=self.temperature,
            user_instruction=user_request,
            tool_docs=tool_docs,
        )
        pre_plan = pre.choices[0].message.content

        # Step 2: TDG construction with the fork's exact prompt.
        system_content = (
            f"<function-definitions>\n{tool_docs}\n</function-definitions>\n"
            f"{OpenAIConstructLLM._construct_dag_prompt}"
        )
        user_content = (
            f"{user_request}\nThese information maybe helpful for you to complete the DAG:\n{pre_plan}"
        )
        completion = chat_completion_request(
            client,
            self.model,
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            self.temperature,
            json_format=True,
        )
        tcs = json.loads(completion.choices[0].message.content).get("tool_calls", [])
        planned_tools = {tc["function_name"] for tc in tcs}
        pending = [
            tc for tc in tcs
            if any(isinstance(v, str) and "<unknown>" in v for v in (tc.get("args") or {}).values())
        ]
        return {"tool_calls": tcs, "planned_tools": planned_tools, "pending": pending,
                "pre_plan": pre_plan}

    def allow_call(self, tool_name: str, args: dict, plan: dict, ctx: dict | None = None) -> bool:
        """Would IPIGuard execute this tool call during traversal?

        Allowed iff the tool is in the pre-approved plan, OR (Node Expansion is on AND
        it is a read-only query tool). A *command* tool absent from the plan — the path
        an injected instruction needs to trigger a sensitive action — is blocked.
        """
        if tool_name in plan.get("planned_tools", set()):
            return True
        if self.allow_query_expansion and tool_name in self._query_tools():
            return True
        return False
