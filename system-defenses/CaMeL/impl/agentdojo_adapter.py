# AgentDojo adapter for CaMeL on the local Qwen3-32B vLLM endpoint.
#
# Reuses the reference repo's `make_tools_pipeline` (camel.models) verbatim for the
# pipeline composition (P-LLM code-gen element, interpreter, Q-LLM, security-policy
# replay). The ONLY glue here is:
#   1. routing both the OpenAI SDK (P-LLM) and pydantic_ai (Q-LLM) to localhost vLLM
#      via OPENAI_BASE_URL / OPENAI_API_KEY env vars, and
#   2. disabling Qwen3 "thinking" mode globally via a chat.completions.create patch
#      (the paper's primary results use a non-reasoning model; skill default = OFF).
#
# Must run inside the reference repo's uv env. Import order matters: agentdojo top-level
# must be imported before camel submodules (circular import otherwise) — mirrors main.py.

from __future__ import annotations

import os
from pathlib import Path

# --- 1. Route LLM clients to the local vLLM endpoint --------------------------------
# Honored by the openai SDK (P-LLM) and by pydantic_ai's OpenAIProvider (Q-LLM).
_PORT = os.environ.get("LOCAL_LLM_PORT", "8000")
os.environ.setdefault("OPENAI_BASE_URL", f"http://localhost:{_PORT}/v1")
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-for-local-vllm")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _read_secret(name: str) -> str | None:
    from ipi_arena_compat import read_secret   # env var first, then secrets.md at the repo root
    return read_secret(name) or None


def route_openrouter(model: str) -> None:
    """Point the P-LLM (openai SDK) and Q-LLM (pydantic_ai) at OpenRouter for a deepseek target.

    Overrides (not setdefault) OPENAI_BASE_URL/KEY so it wins over the localhost defaults set at
    import. Called from build_pipeline when the model id is a provider/model string (contains '/').
    """
    key = _read_secret("openrouter")
    if not key:
        raise SystemExit("no OpenRouter key: set OPENROUTER_API_KEY or add an `openrouter = ...` "
                         "line to secrets.md")
    os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE
    os.environ["OPENAI_API_KEY"] = key


def _patch_openrouter_extra_body(safe_model: str, api_model: str) -> None:
    """Inject the deepseek provider pin + reasoning-off, and remap the slash-free model alias.

    The OpenRouter analog of _patch_disable_thinking (which adds a vLLM-only chat_template_kwargs
    OpenRouter rejects). We name the pipeline with a slash-free alias (``safe_model``) so the
    AgentDojo logger's write path and the reference replayer's read path agree (the logger
    sanitizes '/'->'_' but the replayer reads the raw id -> mismatch on a provider/model id). Here
    we remap that alias back to the real OpenRouter id (``api_model``) at the API boundary and add
    the same openrouter_extra_body as every other arm. Patches both sync and async
    Completions.create (P-LLM openai SDK + Q-LLM pydantic_ai AsyncOpenAI). Idempotent.
    """
    from openai.resources.chat import completions as _c
    from ipi_arena_compat import openrouter_extra_body

    if getattr(_c.Completions.create, "_camel_openrouter", False):
        return
    eb_base = openrouter_extra_body(api_model, thinking=False)

    def _wrap(orig):
        def create(self, *args, **kwargs):
            if kwargs.get("model") == safe_model:
                kwargs["model"] = api_model
            eb = dict(kwargs.get("extra_body") or {})
            for k, v in eb_base.items():
                eb.setdefault(k, v)
            kwargs["extra_body"] = eb
            return orig(self, *args, **kwargs)

        create._camel_openrouter = True
        return create

    _c.Completions.create = _wrap(_c.Completions.create)
    _c.AsyncCompletions.create = _wrap(_c.AsyncCompletions.create)


# --- 2. Disable Qwen3 thinking globally ---------------------------------------------
def _patch_disable_thinking() -> None:
    """Force `chat_template_kwargs.enable_thinking=False` on every chat completion.

    Patches both sync and async Completions.create so it covers the P-LLM (openai SDK)
    and the Q-LLM (pydantic_ai's AsyncOpenAI). Idempotent.
    """
    from openai.resources.chat import completions as _c

    if getattr(_c.Completions.create, "_camel_nothink", False):
        return

    def _wrap(orig):
        def create(self, *args, **kwargs):
            eb = dict(kwargs.get("extra_body") or {})
            ctk = dict(eb.get("chat_template_kwargs") or {})
            ctk.setdefault("enable_thinking", False)
            eb["chat_template_kwargs"] = ctk
            kwargs["extra_body"] = eb
            return orig(self, *args, **kwargs)

        create._camel_nothink = True
        return create

    _c.Completions.create = _wrap(_c.Completions.create)
    _c.AsyncCompletions.create = _wrap(_c.AsyncCompletions.create)


def _patch_interpreter_constant_bug() -> None:
    """Fix a crash-bug in the reference interpreter (README warns of such bugs).

    `interpreter._eval_constant`'s unsupported-constant path (bytes/complex/Ellipsis)
    builds its error message with `type(node.value).__type__` — a typo for `__name__`.
    So when the P-LLM emits e.g. an ellipsis literal, the *error path itself* raises
    AttributeError, which escapes CaMeL's graceful retry loop and crashes the whole run.
    We intercept only that broken path and return the intended graceful Error, so the
    P-LLM is told to fix its code (the repo's designed behavior). No repo file edited.
    """
    from camel.interpreter import interpreter as I

    orig = I._eval_constant
    if getattr(orig, "_camel_fixed", False):
        return

    def _eval_constant(node, namespace, tool_calls_chain, dependencies, eval_args):
        try:
            return orig(node, namespace, tool_calls_chain, dependencies, eval_args)
        except AttributeError:
            # Only the unsupported-constant branch can land here (supported constants
            # construct CaMeL values and never touch `.__type__`).
            return I.EvalResult(
                I.result.Error(
                    I.CaMeLException(
                        NotImplementedError(f"unsupported constant type {type(node.value).__name__}"),
                        (node,),
                        (),
                    )
                ),
                namespace,
                tool_calls_chain,
                dependencies,
            )

    _eval_constant._camel_fixed = True
    I._eval_constant = _eval_constant


def _patch_pydantic_undefined() -> None:
    """Handle pydantic's `PydanticUndefined` sentinel in the interpreter.

    The Q-LLM returns pydantic models via structured output; an unset field can carry
    the `PydanticUndefined` sentinel, whose type the interpreter's `value_from_raw`
    doesn't recognize -> it hits the catch-all and raises `UndefinedClassError`,
    crashing both the error-formatting path AND security-policy `is_trusted` checks.
    The sentinel means "no value", so we map it to `CaMeLNone`. Recursive calls inside
    the original go through the module global, so nested occurrences are caught too.
    """
    from camel.interpreter import value as V

    orig = V.value_from_raw
    if getattr(orig, "_camel_pu_fixed", False):
        return

    def value_from_raw(raw_value, metadata, namespace, dependencies):
        if type(raw_value).__name__ == "PydanticUndefinedType":
            return V.CaMeLNone(metadata, dependencies)
        return orig(raw_value, metadata, namespace, dependencies)

    value_from_raw._camel_pu_fixed = True
    V.value_from_raw = value_from_raw


_AUTHORED_PROGRAM: str | None = None


def set_authored_program(code: str | None) -> None:
    """Force the P-LLM to emit `code` as its program on the FIRST turn (clean single fragment).

    Used by the plan-authoring path: instead of live P-LLM generation (which can emit
    error-correction fragments), we inject a pre-authored self-contained restricted-Python program
    so phase-1 runs it ONCE with a LIVE Q-LLM (recording query_ai_assistant outputs), and phase-2
    replays that single fragment under policy. Set None to restore live generation."""
    global _AUTHORED_PROGRAM
    _AUTHORED_PROGRAM = code


def _patch_force_authored_program() -> None:
    """When an authored program is set, return it as the P-LLM's first-turn code (no model call).

    Wraps PrivilegedLLM._generate_and_interpret_code: on the first turn (empty privileged history)
    with an authored program set, temporarily stub self.llm.query to yield that program. If it
    interprets without error the outer loop stops -> a clean single-fragment cache. Idempotent."""
    from camel.pipeline_elements.privileged_llm import PrivilegedLLM
    from agentdojo import types as ad_types

    orig = PrivilegedLLM._generate_and_interpret_code
    if getattr(orig, "_force_authored", False):
        return

    def patched(self, query, runtime, namespace, env, messages, privileged_llm_messages,
                system_prompt, previous_printed_output, dependencies):
        if _AUTHORED_PROGRAM and len(privileged_llm_messages) == 0:
            real = self.llm.query

            def stub(query, runtime, messages):
                # The P-LLM protocol + the replayer's make_turns both expect exactly one markdown
                # code block; wrap raw authored code in a ```python fence if not already fenced.
                code = _AUTHORED_PROGRAM.strip()
                if not code.startswith("```"):
                    code = f"```python\n{code}\n```"
                msg = ad_types.ChatAssistantMessage(
                    role="assistant",
                    content=[ad_types.text_content_block_from_string(code)],
                    tool_calls=None)
                return query, runtime, None, [*messages, msg], {}

            self.llm.query = stub
            try:
                return orig(self, query, runtime, namespace, env, messages, privileged_llm_messages,
                            system_prompt, previous_printed_output, dependencies)
            finally:
                self.llm.query = real
        return orig(self, query, runtime, namespace, env, messages, privileged_llm_messages,
                    system_prompt, previous_printed_output, dependencies)

    patched._force_authored = True
    PrivilegedLLM._generate_and_interpret_code = patched


def _patch_safe_exception_formatting() -> None:
    """Resilience: never let error-message *formatting* crash the P-LLM retry loop.

    `format_camel_exception` walks a failed value's capability graph; an unexpected
    value there would crash the whole run instead of just failing one task. Wrap it so
    any formatting failure degrades to a generic redacted message (the P-LLM then
    retries / the task fails gracefully). Patches both the live and replay copies.
    """
    from camel.pipeline_elements import privileged_llm as P
    from camel.pipeline_elements import replay_privileged_llm as R

    for mod in (P, R):
        orig = mod.format_camel_exception
        if getattr(orig, "_camel_safe", False):
            continue

        def safe(camel_exception, code, _orig=orig):
            try:
                return _orig(camel_exception, code)
            except Exception:
                exc = getattr(camel_exception, "exception", camel_exception)
                return f"\n{type(exc).__name__}: <error redacted: message could not be formatted>\n"

        safe._camel_safe = True
        mod.format_camel_exception = safe


# --- 3. Pipeline builder (reuses repo's make_tools_pipeline) -------------------------
def build_pipeline(
    suite_name: str,
    model: str = "Qwen3-32B",
    variant: str = "camel",
    attack_name: str = "important_instructions",
    q_llm: str | None = None,
    strict: bool = False,
):
    """Build a composed AgentPipeline for the given CaMeL variant on the local model.

    variant: "undefended" | "camel" | "camel+secpol".
    """
    openrouter = "/" in model  # provider/model id (e.g. deepseek/deepseek-v4-pro) -> OpenRouter
    api_model = model
    if openrouter:
        # Slash-free alias for pipeline naming so the logger write path (it sanitizes '/'->'_')
        # and the reference replayer read path agree; remapped to api_model at the API boundary.
        model = model.replace("/", "_")
        route_openrouter(api_model)
        _patch_openrouter_extra_body(model, api_model)
    else:
        _patch_disable_thinking()

    # Import order mirrors the repo's main.py to avoid agentdojo's circular import.
    import camel.custom_yaml  # noqa: F401
    from camel.interpreter.interpreter import MetadataEvalMode
    from camel.models import make_tools_pipeline

    _patch_interpreter_constant_bug()
    _patch_pydantic_undefined()
    _patch_safe_exception_formatting()
    _patch_force_authored_program()

    # The `important_instructions` attack fills `{model}` in its injection from the
    # pipeline's registered identity (AgentDojo's MODEL_NAMES), keyed by pipeline.name.
    # Qwen3-32B isn't a built-in, so register all variant names → a generic identity.
    from agentdojo.models import MODEL_NAMES
    for _suffix in ("", "+camel", "+camel+secpol", "+camel+secpol+strict"):
        MODEL_NAMES[f"{model}{_suffix}"] = "AI assistant"

    eval_mode = MetadataEvalMode.STRICT if strict else MetadataEvalMode.NORMAL
    model_id = f"openai:{model}"  # routed to local vLLM via OPENAI_BASE_URL
    q_llm_id = f"openai:{q_llm}" if q_llm else None

    if variant == "undefended":
        use_original, replay = True, False
    elif variant == "camel":
        use_original, replay = False, False
    elif variant == "camel+secpol":
        use_original, replay = False, True
    else:
        raise ValueError(f"unknown variant {variant!r}")

    return make_tools_pipeline(
        model_id,            # model
        use_original,        # use_original
        replay,              # replay_with_policies
        attack_name,         # attack_name
        "medium",            # reasoning_effort (unused for non-reasoning model)
        None,                # thinking_budget_tokens
        suite_name,          # suite
        None,                # ad_defense
        eval_mode,           # eval_mode
        q_llm_id,            # q_llm
    )
