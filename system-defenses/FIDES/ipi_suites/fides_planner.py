"""FIDES labeled planner — the *real* reference planning loop, vendored from the notebook.

Paper: "Securing AI Agents with Information-Flow Control" (Costa et al., arXiv 2505.23643).
This module vendors the FIDES **labeled planner** (the actual agent architecture the paper
evaluates) from the reference repo's tutorial notebook
(`system-defenses/FIDES/fides/Tutorial.ipynb`), which is the ground truth. It is a faithful
copy of the generic, reusable pieces:

    cell 5   typed-tool abstraction  -> Tool, call_tool, CustomSchemaGenerator, make_custom_generator
    cell 7   planning loop           -> Action/Query/ToolCall/Response, Planner, PlanningLoop
    cell 9   bare planner            -> BasicPlanner
    cell 11  variable passing        -> VariablePassingPlanner, read_variable  (Variable-Passing* arm)
    cell 23  metadata wrapper        -> MetaValue
    cell 25  generic label plumbing  -> readers_label, metadata_to_label, Lattice alias
    cell 29  labeled planning loop   -> PolicyViolation, LabeledPlanner, LabeledPlanningLoop
    cell 31  labeled bare planner    -> LabeledBasicPlanner

The IFC lattice algebra (`IntegrityLabel`, `PowersetLattice`, `InverseLattice`, `ProductLabel`,
…) is imported from the sibling `../impl/defense.py`, which already vendors it verbatim from
notebook cell 21 (single source of truth). Behavior-/example-specific tools and policies
(e.g. the email assistant + its three policies) live in the caller, NOT here.

Out of scope (the FIDES* utility-recovery layer — not in the repo): quarantined_llm/query_llm,
expand_variables/inspect, constrained decoding, the type-capacity lattice.
"""
from __future__ import annotations

import inspect
import json
import sys
import uuid

import llm_parse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, FrozenSet

import openai
from docstring_parser import parse
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from pydantic import BaseModel, Field
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode
from pydantic_core import CoreSchema

# --- IFC lattice core: reuse the verbatim cell-21 vendoring in ../impl/defense.py -----------
_IMPL_DIR = Path(__file__).resolve().parent.parent / "impl"
if str(_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(_IMPL_DIR))
from defense import (  # noqa: E402  (path-based import of the sibling impl)
    IntegrityLabel,
    InverseLattice,
    PowersetLattice,
    ProductLabel,
)

# The FIDES label is a product of an integrity label and a confidentiality (readers) label.
# Confidentiality is an InverseLattice over the powerset of authorized readers (notebook cell 25).
Lattice = ProductLabel  # type alias (ProductLabel[IntegrityLabel, InverseLattice[PowersetLattice[str]]])


# ============================================================================================
# cell 23 — MetaValue: wraps a value with an IFC-label metadata dict {"integrity", "confidentiality"}
# ============================================================================================
from typing import Generic, TypeVar, get_args  # noqa: E402
from pydantic import GetCoreSchemaHandler  # noqa: E402
from pydantic_core import core_schema  # noqa: E402

_T = TypeVar("_T")


class MetaValue(Generic[_T]):
    def __init__(self, value: _T, metadata: dict[str, Any] | None = None):
        self.value = value
        self.metadata = metadata or {}

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)

    def __lt__(self, other: Any) -> bool:
        return self.value < (other.value if isinstance(other, MetaValue) else other)  # type: ignore

    def __gt__(self, other: Any) -> bool:
        return self.value > (other.value if isinstance(other, MetaValue) else other)  # type: ignore

    def __getattr__(self, name: str):
        # delegate all other attribute access to the inner value
        return getattr(self.value, name)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        args = get_args(source)
        if not args:
            raise TypeError("MetaValue must be parameterized with a type, e.g., MetaValue[int]")
        inner_schema = handler.generate_schema(args[0])
        instance_schema = core_schema.is_instance_schema(cls)
        wrap_schema = core_schema.no_info_after_validator_function(cls, inner_schema)
        union = core_schema.union_schema([instance_schema, wrap_schema])
        return core_schema.json_or_python_schema(json_schema=union, python_schema=union)


# ============================================================================================
# cell 5 — typed-tool abstraction (pydantic param/result models -> OpenAI tool schema)
# ============================================================================================
class CustomSchemaGenerator(GenerateJsonSchema):
    def generate(self, schema: CoreSchema, mode: JsonSchemaMode = "validation"):
        json_schema = super().generate(schema, mode=mode)
        json_schema.pop("title", None)
        for prop in json_schema.get("properties", {}).values():
            prop.pop("title", None)
        json_schema["additionalProperties"] = False
        return json_schema


def make_custom_generator(variables: list[str]) -> type[GenerateJsonSchema]:
    """Schema generator for the Variable-Passing planner: every argument becomes an anyOf of a
    literal {kind:value} or a {kind:variable_name} referencing one of `variables` (cell 5)."""

    class CustomSchemaGeneratorVar(GenerateJsonSchema):
        def generate(self, schema: CoreSchema, mode: JsonSchemaMode = "validation"):
            json_schema = super().generate(schema, mode=mode)
            json_schema.pop("title", None)
            for prop in json_schema.get("properties", {}).values():
                prop.pop("title", None)
            json_schema["additionalProperties"] = False
            assert "properties" in json_schema, "No properties found in JSON schema"
            new_properties = {}
            for prop_name, prop_schema in json_schema["properties"].items():
                new_properties[prop_name] = {
                    "description": prop_schema.get("description", ""),
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "value"},
                                "value": {"type": prop_schema.get("type", "string")},
                            },
                            "required": ["kind", "value"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "variable_name"},
                                "value": {"type": "string", **({"enum": variables} if variables else {})},
                            },
                            "required": ["kind", "value"],
                            "additionalProperties": False,
                        },
                    ],
                }
                json_schema["properties"] = new_properties
                json_schema["additionalProperties"] = False
            return json_schema

    return CustomSchemaGeneratorVar


@dataclass
class Tool:
    name: str
    description: str
    callable: Callable[[type[BaseModel]], type[BaseModel]]
    parameter_model: type[BaseModel]
    result_model: type[BaseModel]

    @classmethod
    def from_callable(cls, callable: Callable[[type[BaseModel]], type[BaseModel]]) -> "Tool":
        if not callable.__doc__:
            raise ValueError(f"Callable {callable.__name__} has no docstring")
        doc = parse(callable.__doc__)
        if not doc.short_description:
            raise ValueError(f"Callable {callable.__name__} has no short description")
        sig = inspect.signature(callable)
        param = next(iter(sig.parameters.values()))
        model_cls = param.annotation
        if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
            raise TypeError(f"Parameter of {callable.__name__} must be a subclass of BaseModel")
        if sig.return_annotation is None:
            raise TypeError(f"Callable {callable.__name__} must have a return type annotation")
        result_cls = sig.return_annotation
        if not (isinstance(result_cls, type) and issubclass(result_cls, BaseModel)):
            raise TypeError(f"Return type of {callable.__name__} must be a subclass of BaseModel")
        return cls(
            name=callable.__name__,
            description=doc.short_description,
            callable=callable,
            parameter_model=model_cls,
            result_model=result_cls,
        )

    def to_dict_openai(self) -> ChatCompletionToolParam:
        return openai.pydantic_function_tool(self.parameter_model, name=self.name, description=self.description)

    def to_dict(self) -> ChatCompletionToolParam:
        model_json_schema = self.parameter_model.model_json_schema(schema_generator=CustomSchemaGenerator)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": model_json_schema,
                "strict": True,
            },
        }

    def to_dict_var(self, variables: list[str]) -> ChatCompletionToolParam:
        model_json_schema = self.parameter_model.model_json_schema(schema_generator=make_custom_generator(variables))
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": model_json_schema,
                "strict": True,
            },
        }


def call_tool(tools: list[Tool], name: str, json_args: str) -> type[BaseModel]:
    """Call the named tool with JSON-serialized arguments; returns its (labeled) result model."""
    if name not in {t.name for t in tools}:
        raise ValueError(f"Unknown tool: {name}")
    tool = {t.name: t for t in tools}[name]
    args = llm_parse.safe_args(json_args)  # tolerate fenced/preamble/malformed/None arg JSON
    params = tool.parameter_model(**args)
    return tool.callable(params)  # type: ignore


# ============================================================================================
# cell 7 — planning loop primitives
# ============================================================================================
@dataclass
class Action:
    pass


@dataclass
class Query(Action):
    messages: list[ChatCompletionMessageParam]
    tools: list[ChatCompletionToolParam]


@dataclass
class ToolCall(Action):
    id: str
    name: str
    arguments: str


@dataclass
class Response(Action):
    response: str


class Planner(ABC):
    @abstractmethod
    def next_action(self, message: ChatCompletionMessage | ChatCompletionMessageParam) -> Action:
        """Given a message, determine the next action in the planning loop."""


class PlanningLoop:
    def __init__(self, planner: Planner, client: "openai.Client", model: str, tools: list[Tool]):
        self.planner = planner
        self.client = client
        self.model = model
        self.tools = tools
        self.turn = 0

    def loop(self, msg: ChatCompletionMessage | ChatCompletionMessageParam) -> str:
        current_msg = msg
        while True:
            self.turn += 1
            action = self.planner.next_action(current_msg)
            match action:
                case Query(messages, tools):
                    response = self.client.chat.completions.create(
                        model=self.model, messages=messages, tools=tools, parallel_tool_calls=False
                    )
                    current_msg = response.choices[0].message
                case ToolCall(id, name, arguments):
                    result = call_tool(self.tools, name, arguments)
                    current_msg = ChatCompletionToolMessageParam(role="tool", tool_call_id=id, content=str(result))
                case Response(response):
                    return response
                case _:
                    raise ValueError("Invalid action")


# ============================================================================================
# cell 9 — BasicPlanner (no taint)
# ============================================================================================
class BasicPlanner(Planner):
    def __init__(self, state: list[ChatCompletionMessageParam], tools: list[Tool]):
        self.tools = tools
        self.history = state

    def next_action(self, message: ChatCompletionMessage | ChatCompletionMessageParam) -> Action:
        match message:
            case {"role": "user"} | {"role": "tool"}:
                self.history.append(message)
                return Query(messages=self.history, tools=[t.to_dict() for t in self.tools])
            case ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls) if tool_calls:
                assert len(tool_calls) == 1, "Only one tool call is supported"
                self.history.append(_assistant_param(content, tool_calls))
                return ToolCall(
                    id=tool_calls[0].id,
                    name=tool_calls[0].function.name,
                    arguments=tool_calls[0].function.arguments,
                )
            case ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls) if content:
                assert not tool_calls, "Tool calls are not supported in this context"
                self.history.append(ChatCompletionAssistantMessageParam(role="assistant", content=content, tool_calls=[]))
                return Response(response=content)
            case _:
                raise ValueError("Invalid message format")


def _assistant_param(content, tool_calls) -> ChatCompletionAssistantMessageParam:
    """Convert an SDK assistant message's tool_calls into the param dict the history expects."""
    tcs: list[ChatCompletionMessageToolCallParam] = [
        {
            "id": tc.id,
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            "type": "function",
        }
        for tc in tool_calls
    ]
    return ChatCompletionAssistantMessageParam(role="assistant", content=content, tool_calls=tcs)


# ============================================================================================
# cell 11 — VariablePassingPlanner (the Variable-Passing* arm) + read_variable tool
# ============================================================================================
class ReadVariableParams(BaseModel):
    variable_name: str = Field(..., description="The name of the variable to read.")


class ReadVariableResult(BaseModel):
    value: str = Field(..., description="The value of the variable read.")


def read_variable(params: ReadVariableParams) -> ReadVariableResult:
    """Reads the value of a variable."""
    # Dummy body: the real logic lives in VariablePassingPlanner.next_action.
    return ReadVariableResult(value="")


class VariablePassingPlanner(Planner):
    def __init__(self, state: list[ChatCompletionMessageParam], tools: list[Tool]):
        self.tools = tools
        self.history = state
        self.memory: dict[str, Any] = {}

    def _expand_args(self, args: dict[str, dict[str, str]]) -> str:
        actual_args = {}
        for a, v in args.items():
            if v["kind"] == "value":
                actual_args[a] = v["value"]
            else:
                assert v["kind"] == "variable", f"Invalid kind for argument {a}: {v['kind']}"
                actual_args[a] = self.memory[v["variable"]]
        return json.dumps(actual_args)

    def next_action(self, message: ChatCompletionMessage | ChatCompletionMessageParam) -> Action:
        tools = [t.to_dict_var(list(self.memory.keys())) for t in self.tools]
        match message:
            case {"role": "user"}:
                self.history.append(message)
                return Query(messages=self.history, tools=tools)
            case {"role": "tool", "content": content, "tool_call_id": tool_call_id}:
                var_name = str(uuid.uuid4())
                self.memory[var_name] = content
                self.history.append(ChatCompletionToolMessageParam(role="tool", content=var_name, tool_call_id=tool_call_id))
                return Query(messages=self.history, tools=tools)
            case ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls) if tool_calls:
                assert len(tool_calls) == 1, "Only one tool call is supported"
                self.history.append(_assistant_param(content, tool_calls))
                if tool_calls[0].function.name == "read_variable":
                    args = json.loads(tool_calls[0].function.arguments)
                    assert "variable_name" in args, "read_variable requires a 'variable_name' argument"
                    var_name = args["variable_name"]["value"]
                    assert var_name in self.memory, f"Variable {var_name} not found in memory"
                    value = self.memory[var_name]
                    self.history.append(ChatCompletionToolMessageParam(role="tool", content=value, tool_call_id=tool_calls[0].id))
                    return Query(messages=self.history, tools=tools)
                return ToolCall(
                    id=tool_calls[0].id,
                    name=tool_calls[0].function.name,
                    arguments=self._expand_args(json.loads(tool_calls[0].function.arguments)),
                )
            case ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls) if content:
                assert not tool_calls, "Tool calls are not supported in this context"
                self.history.append(ChatCompletionAssistantMessageParam(role="assistant", content=content, tool_calls=[]))
                return Response(response=content)
            case _:
                raise ValueError("Invalid message format")


# ============================================================================================
# cell 25 — generic label plumbing (the email-specific universe/label_email stay in the caller)
# ============================================================================================
def readers_label(readers: FrozenSet[str], universe: FrozenSet[str]) -> "InverseLattice[PowersetLattice[str]]":
    """Confidentiality label for a set of authorized readers (InverseLattice over the powerset).

    Higher in the lattice = fewer readers = more secret; join = intersection of reader sets.
    """
    assert readers.issubset(universe), "Readers must be a subset of the universe"
    return InverseLattice(PowersetLattice(subset=readers, universe=universe))


def metadata_to_label(metadata: dict[str, Any]) -> Lattice:
    """Convert a MetaValue metadata dict {"integrity","confidentiality"} into the product label."""
    return ProductLabel(metadata["integrity"], metadata["confidentiality"])


# ============================================================================================
# cell 29 — the labeled planning loop with dynamic taint-tracking + policy enforcement
# ============================================================================================
class PolicyViolation(Exception):
    def __init__(self, reason: str):
        super().__init__(f"Policy violation: {reason}")
        self.reason = reason


Policy = Callable[[list[tuple[Action, Lattice]]], None]  # may raise PolicyViolation


class LabeledPlanner(ABC):
    @abstractmethod
    def next_action(
        self, message: ChatCompletionMessage | ChatCompletionMessageParam, label: Lattice
    ) -> tuple[Action, Lattice]:
        """Given a message + the current context label, return the next action and its label."""


class LabeledPlanningLoop:
    def __init__(
        self,
        planner: LabeledPlanner,
        client: "openai.Client",
        model: str,
        tools: list[Tool],
        policy: Policy,
    ):
        self.planner = planner
        self.client = client
        self.model = model
        self.tools = tools
        self.policy = policy
        self.turn = 0

    def loop(
        self,
        msg: ChatCompletionMessage | ChatCompletionMessageParam,
        label: Lattice,
        max_turns: int = 16,
    ) -> tuple[str, Lattice]:
        """Run the planning loop with dynamic taint-tracking. May raise PolicyViolation.

        `max_turns` is the ONLY deviation from the notebook (an unbounded `while True`): a safety
        cap so a live model that never emits a final Response can't spin forever. On hitting the
        cap we return the latest assistant text (or "") — it does not affect the taint/policy
        mechanism, only bounds wall-clock.
        """
        trace: list[tuple[Action, Lattice]] = []
        current_msg = msg
        current_label = label
        while self.turn < max_turns:
            self.turn += 1
            action, label = self.planner.next_action(current_msg, current_label)
            trace.append((action, label))
            self.policy(trace)  # enforce per-tool policy BEFORE the action executes

            match action:
                case Query(messages, tools):
                    response = self.client.chat.completions.create(
                        model=self.model, messages=messages, tools=tools, parallel_tool_calls=False
                    )
                    current_msg = response.choices[0].message
                case ToolCall(id, name, arguments):
                    result = call_tool(self.tools, name, arguments)
                    current_msg = ChatCompletionToolMessageParam(role="tool", tool_call_id=id, content=str(result))
                    current_label = metadata_to_label(result.root.metadata).join(current_label)
                case Response(response):
                    return response, current_label
                case _:
                    raise ValueError("Invalid action")
        return "", current_label  # max_turns reached without a final Response


# ============================================================================================
# cell 31 — LabeledBasicPlanner (taint-tracking; carries the context label onto each action)
# ============================================================================================
class LabeledBasicPlanner(LabeledPlanner):
    def __init__(self, state: list[ChatCompletionMessageParam], tools: list[Tool]):
        self.tools = tools
        self.history = state

    def next_action(
        self, message: ChatCompletionMessage | ChatCompletionMessageParam, label: Lattice
    ) -> tuple[Action, Lattice]:
        match message:
            case {"role": "user"} | {"role": "tool"}:
                self.history.append(message)
                return Query(messages=self.history, tools=[t.to_dict() for t in self.tools]), label
            case ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls) if tool_calls:
                assert len(tool_calls) == 1, "Only one tool call is supported"
                self.history.append(_assistant_param(content, tool_calls))
                return ToolCall(
                    id=tool_calls[0].id,
                    name=tool_calls[0].function.name,
                    arguments=tool_calls[0].function.arguments,
                ), label
            case ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls) if content:
                assert not tool_calls, "Tool calls are not supported in this context"
                self.history.append(ChatCompletionAssistantMessageParam(role="assistant", content=content, tool_calls=[]))
                return Response(response=content), label
            case _:
                raise ValueError("Invalid message format")
