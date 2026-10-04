"""AgentDojo pipeline-element wrappers for the Firewalls defense.

Two elements, both go INSIDE the ToolsExecutionLoop:

  SanitizerElement -- placed AFTER ToolsExecutor(): rewrites each tool-result
    message's text via Defense.transform_tool_output. The trusted user task is the
    `query` arg (AgentDojo passes the user request as `query` throughout).

  MinimizerElement -- placed BEFORE ToolsExecutor(): rewrites the args of the tool
    calls in the last assistant message via Defense.minimize_tool_args, using the
    tool descriptions from the runtime.

Compose like the built-in PI detector, e.g.:

    loop = ToolsExecutionLoop([ToolsExecutor(fmt), SanitizerElement(d), llm])
    pipeline = AgentPipeline([sys_msg, init_query, llm, loop])
"""

from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, get_text_content_as_str, text_content_block_from_string

from defense import Defense


class SanitizerElement(BasePipelineElement):
    """Tool-Output Firewall (Sanitizer). Sanitizes the most recent run of tool
    messages (the ones ToolsExecutor just appended)."""

    name = "firewalls-sanitizer"

    def __init__(self, defense: Defense):
        self.defense = defense

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ):
        if not messages or messages[-1]["role"] != "tool":
            return query, runtime, env, messages, extra_args
        ctx = {"user_request": query}
        messages = list(messages)
        # sanitize the trailing run of tool messages
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] != "tool":
                break
            text = get_text_content_as_str(messages[i]["content"]) or ""
            clean = self.defense.transform_tool_output(text, ctx)
            if clean != text:
                messages[i] = {**messages[i], "content": [text_content_block_from_string(clean)]}
        return query, runtime, env, messages, extra_args


class MinimizerElement(BasePipelineElement):
    """Tool-Input Firewall (Minimizer). Filters the args of tool calls in the last
    assistant message before they are executed."""

    name = "firewalls-minimizer"

    def __init__(self, defense: Defense):
        self.defense = defense

    def _tool_desc(self, runtime: FunctionsRuntime, name: str) -> str:
        fn = runtime.functions.get(name)
        return getattr(fn, "description", "") if fn else ""

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ):
        if not messages or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        tool_calls = messages[-1].get("tool_calls")
        if not tool_calls:
            return query, runtime, env, messages, extra_args
        for tc in tool_calls:
            if not tc.args:
                continue
            ctx = {
                "user_request": query,
                "tool_name": tc.function,
                "tool_description": self._tool_desc(runtime, tc.function),
            }
            tc.args = self.defense.minimize_tool_args(dict(tc.args), ctx)
        return query, runtime, env, messages, extra_args
