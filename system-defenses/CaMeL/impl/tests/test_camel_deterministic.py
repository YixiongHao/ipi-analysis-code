"""Deterministic unit tests for the CaMeL defense core (impl/defense.py) and the
standalone patch helpers in impl/agentdojo_adapter.py.

CORE PRINCIPLE: test CODE CORRECTNESS, not efficacy. CaMeL is a full-loop
isolation defense (P-LLM planner + quarantined Q-LLM) whose security logic lives in
the reference repo; the impl module is glue. So the only pure, model-free logic worth
testing deterministically is:
  (a) `Defense.__init__` config / variant validation, and
  (b) the standalone patch helpers (what they patch, and idempotency).

We NEVER assert ASR / blocking. The `_patch_disable_thinking` helper only needs the
openai SDK and is fully testable here; the other three patch helpers need the reference
`camel` package (and its deps, e.g. pydantic_ai), which is NOT importable under the
master .venv — those tests gate on a successful import and skip with a clear reason.
"""
import pytest

from testlib import pathsetup

impl = pathsetup.load_impl("CaMeL")


def _load_adapter():
    """Load CaMeL's agentdojo_adapter.py by file path (bare import collides with the
    other five defenses' adapters of the same name)."""
    return pathsetup.load_impl("CaMeL", "agentdojo_adapter.py")


# ============================================================ Defense.__init__ ===
class TestDefenseConfig:
    def test_name_is_camel(self):
        assert impl.Defense.name == "camel"
        assert impl.Defense().name == "camel"

    def test_variants_constant(self):
        # The exact allowed set, asserted from the code (not hardcoded guesses).
        assert impl.VARIANTS == ("undefended", "camel", "camel+secpol")

    @pytest.mark.parametrize("variant", ["undefended", "camel", "camel+secpol"])
    def test_each_valid_variant_accepted_and_stored(self, variant):
        d = impl.Defense(variant=variant)
        assert d.variant == variant

    def test_invalid_variant_raises_valueerror(self):
        # The code raises ValueError (not assert) on an unknown variant.
        with pytest.raises(ValueError):
            impl.Defense(variant="bogus")

    def test_default_model_is_qwen(self):
        assert impl.Defense().model == "Qwen3-32B"

    def test_default_variant_is_camel(self):
        assert impl.Defense().variant == "camel"

    def test_constructor_fields_stored(self):
        d = impl.Defense(
            model="some-model",
            variant="camel+secpol",
            attack_name="custom_attack",
            q_llm="q-model",
            strict=True,
        )
        assert d.model == "some-model"
        assert d.variant == "camel+secpol"
        assert d.attack_name == "custom_attack"
        assert d.q_llm == "q-model"
        assert d.strict is True

    def test_defaults_for_optional_fields(self):
        d = impl.Defense()
        assert d.attack_name == "important_instructions"
        assert d.q_llm is None
        assert d.strict is False

    def test_build_agentdojo_pipeline_present(self):
        # We do NOT call it here (it needs the reference env); just confirm the
        # modular surface exists, since it is CaMeL's only callable seam.
        assert hasattr(impl.Defense(), "build_agentdojo_pipeline")
        assert callable(impl.Defense().build_agentdojo_pipeline)


# ============================================ _patch_disable_thinking (pure) ===
# This helper only needs the openai SDK (no `camel`), so it is testable here. It
# wraps both sync and async chat Completions.create to force
# extra_body.chat_template_kwargs.enable_thinking=False, and is idempotent.
class TestPatchDisableThinking:
    def test_marks_and_injects_enable_thinking_false(self):
        adapter = _load_adapter()
        adapter._patch_disable_thinking()

        from openai.resources.chat import completions as _c

        # Idempotency marker set on both wrapped methods.
        assert getattr(_c.Completions.create, "_camel_nothink", False) is True
        assert getattr(_c.AsyncCompletions.create, "_camel_nothink", False) is True

    def test_idempotent_does_not_double_wrap(self):
        adapter = _load_adapter()
        adapter._patch_disable_thinking()
        from openai.resources.chat import completions as _c

        first = _c.Completions.create
        adapter._patch_disable_thinking()  # second call is a no-op
        assert _c.Completions.create is first  # same object, not re-wrapped

    def test_real_wrapper_injects_flag_into_extra_body(self, monkeypatch):
        # Exercise the ACTUAL helper: install a probe as the unpatched create, then let
        # the helper wrap it, then call the wrapped method. Assert the helper's wrapper
        # set enable_thinking=False without clobbering sibling extra_body keys, and
        # forwarded to the original. monkeypatch restores the real create after the test.
        adapter = _load_adapter()
        from openai.resources.chat import completions as _c

        captured = {}

        def probe(self, *args, **kwargs):
            captured.update(kwargs)
            return "ok"

        # Reset to an unwrapped probe (no `_camel_nothink` marker) so the helper wraps it.
        monkeypatch.setattr(_c.Completions, "create", probe, raising=True)
        monkeypatch.setattr(_c.AsyncCompletions, "create", probe, raising=True)

        adapter._patch_disable_thinking()  # wraps our probe

        ret = _c.Completions.create(object(), model="m", extra_body={"foo": "bar"})
        assert ret == "ok"  # forwarded to the original probe
        assert captured["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        assert captured["extra_body"]["foo"] == "bar"  # sibling key preserved


# ===================================== camel-dependent patch helpers (gated) ===
# These three patch reference-`camel` internals. The reference package (and deps like
# pydantic_ai) is not importable under the master .venv, so each test first attempts to
# import the target module and skips with a clear reason if it is unavailable.
def _camel_importable(module_path: str) -> bool:
    """True iff the named reference module imports cleanly under the current env.
    Triggers the adapter's path inserts first (camel-prompt-injection/src)."""
    pathsetup.get_ipi_adapter("camel")  # ensures camel-prompt-injection/src on sys.path
    import importlib

    try:
        importlib.import_module(module_path)
        return True
    except Exception:
        return False


class TestCamelDependentPatches:
    def test_patch_interpreter_constant_bug(self):
        if not _camel_importable("camel.interpreter.interpreter"):
            pytest.skip("reference `camel` package not importable under master .venv "
                        "(needs the reference uv env)")
        adapter = _load_adapter()
        adapter._patch_interpreter_constant_bug()
        from camel.interpreter import interpreter as I

        assert getattr(I._eval_constant, "_camel_fixed", False) is True
        first = I._eval_constant
        adapter._patch_interpreter_constant_bug()  # idempotent
        assert I._eval_constant is first

    def test_patch_pydantic_undefined(self):
        if not _camel_importable("camel.interpreter.value"):
            pytest.skip("reference `camel` package not importable under master .venv "
                        "(needs the reference uv env)")
        adapter = _load_adapter()
        adapter._patch_pydantic_undefined()
        from camel.interpreter import value as V

        assert getattr(V.value_from_raw, "_camel_pu_fixed", False) is True
        first = V.value_from_raw
        adapter._patch_pydantic_undefined()  # idempotent
        assert V.value_from_raw is first

    def test_patch_safe_exception_formatting(self):
        if not _camel_importable("camel.pipeline_elements.privileged_llm"):
            pytest.skip("reference `camel` package not importable under master .venv "
                        "(needs the reference uv env)")
        adapter = _load_adapter()
        adapter._patch_safe_exception_formatting()
        from camel.pipeline_elements import privileged_llm as P
        from camel.pipeline_elements import replay_privileged_llm as R

        assert getattr(P.format_camel_exception, "_camel_safe", False) is True
        assert getattr(R.format_camel_exception, "_camel_safe", False) is True
        first_p = P.format_camel_exception
        adapter._patch_safe_exception_formatting()  # idempotent
        assert P.format_camel_exception is first_p


# ============================================ local-vLLM env wiring (pure) =======
class TestEnvWiring:
    def test_openai_base_url_and_key_set(self):
        # Importing the adapter module sets OPENAI_BASE_URL / OPENAI_API_KEY (setdefault)
        # so both the P-LLM (openai SDK) and Q-LLM (pydantic_ai) route to local vLLM.
        import os

        _load_adapter()
        assert os.environ.get("OPENAI_BASE_URL", "").startswith("http://localhost:")
        assert os.environ.get("OPENAI_BASE_URL", "").endswith("/v1")
        assert os.environ.get("OPENAI_API_KEY")  # non-empty
