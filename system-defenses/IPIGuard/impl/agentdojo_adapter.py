"""Compose AgentDojo pipelines that run the reference fork's `ipiguard` defense on a
local Qwen3-32B vLLM endpoint, without editing the fork.

`build_ipiguard_pipeline()` mirrors, line-for-line, the `config.defense == "ipiguard"`
branch of the fork's `AgentPipeline.from_config` — except the LLM client points at our
local vLLM server instead of the fork's hardcoded provider. `build_baseline_pipeline()`
is the fork's no-defense pipeline, for the comparable baseline arm.
"""
import _fork_env  # noqa: F401  (must precede agentdojo imports; sets sys.path + stubs)

import openai

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.agent_pipeline.llms.ipiguard_llm import OpenAIConstructLLM, OpenAITraverseLLM
from agentdojo.agent_pipeline.tool_execution import (
    DagToolsExecutionLoop,
    DagToolsExecutor,
    ToolsExecutionLoop,
    ToolsExecutor,
)

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen3-32B"


def make_client(base_url: str = DEFAULT_BASE_URL) -> openai.OpenAI:
    # Generous timeout: traversal makes long generations on a shared, contended GPU;
    # the client default (~10 min but per-attempt) can trip APITimeoutError under load.
    return openai.OpenAI(api_key="EMPTY", base_url=base_url, timeout=600.0, max_retries=2)


def _pipeline_name(model: str) -> str:
    """Pipeline.name must contain a key registered in agentdojo MODEL_NAMES so the
    `important_instructions` attack can resolve a prose model name. Qwen3 ids aren't
    registered; embed the Qwen2.5 key (display name "Qwen created by Alibaba Cloud."
    is generic and correct for Qwen3 too) while keeping the real model id visible.
    """
    if "Qwen" in model:
        return f"{model} [Qwen/Qwen2.5-7B-Instruct]"
    return model


def build_ipiguard_pipeline(
    model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL
) -> AgentPipeline:
    """The fork's `ipiguard` pipeline (TDG construction + traversal) on a local model.

    The model string must contain 'Qwen'/'gpt'/'llama' (the fork's ipiguard branch
    asserts this); 'Qwen3-32B' satisfies it.
    """
    assert any(k in model for k in ("gpt", "Qwen", "llama")), (
        f"ipiguard branch only supports gpt/Qwen/llama model ids, got {model!r}"
    )
    client = make_client(base_url)
    system_message = SystemMessage(load_system_message(None))
    construct_llm = OpenAIConstructLLM(client, model)
    traverse_llm = OpenAITraverseLLM(client, model)
    executor = DagToolsExecutor(traverse_llm)
    tools_loop = DagToolsExecutionLoop(executor)
    pipeline = AgentPipeline([system_message, InitQuery(), construct_llm, tools_loop])
    pipeline.name = _pipeline_name(model)
    return pipeline


def build_baseline_pipeline(
    model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL
) -> AgentPipeline:
    """The fork's no-defense pipeline on the same local model (comparable baseline)."""
    client = make_client(base_url)
    llm: BasePipelineElement = OpenAILLM(client, model)
    system_message = SystemMessage(load_system_message(None))
    tools_loop = ToolsExecutionLoop([ToolsExecutor(), llm])
    pipeline = AgentPipeline([system_message, InitQuery(), llm, tools_loop])
    pipeline.name = _pipeline_name(model)
    return pipeline


def build_pipeline(defense: str | None, **kw) -> AgentPipeline:
    if defense in (None, "None", "none"):
        return build_baseline_pipeline(**kw)
    if defense == "ipiguard":
        return build_ipiguard_pipeline(**kw)
    raise ValueError(f"unsupported defense: {defense!r}")
