from mcpgraphy.models import EffectClass, OutcomeKind
from mcpgraphy.observe.normalize import normalize_error, redact
from mcpgraphy.safety import Budget, SafetyPolicy


def test_error_prefix_grammar_maps_to_outcome_kind():
    o = normalize_error("Error executing tool publish_release: STATE_CONFLICT: nope")
    assert o.kind is OutcomeKind.ERR_STATE and o.code == "STATE_CONFLICT" and o.message == "nope"
    assert normalize_error("NOT_FOUND: x").kind is OutcomeKind.ERR_NOT_FOUND
    assert normalize_error("PERMISSION_DENIED: x").kind is OutcomeKind.ERR_PERMISSION
    assert normalize_error("INVALID_ARGUMENT: x").kind is OutcomeKind.ERR_VALIDATION


def test_validation_and_unknown_prefixes():
    v = normalize_error(
        "Error executing tool get_release: 1 validation error for get_releaseArguments\n"
        "release_id\n  Input should be a valid string"
    )
    assert v.kind is OutcomeKind.ERR_VALIDATION and v.code == "SCHEMA_VALIDATION"
    u = normalize_error("Error executing tool x: something exploded")
    assert u.kind is OutcomeKind.ERR_OTHER and u.code is None
    # an unknown ALLCAPS prefix is still ERR_OTHER, never guessed into a category
    assert normalize_error("TEAPOT: short and stout").kind is OutcomeKind.ERR_OTHER


def test_redact_masks_secret_keys_and_bearer_tokens():
    out = redact(
        {
            "api_key": "abc",
            "nested": {"Authorization": "x", "note": "Bearer abc.def"},
            "items": ["sk-ABCDEFGHIJKLMNOPQRS", "fine"],
        }
    )
    assert out["api_key"] == "[REDACTED]"
    assert out["nested"]["Authorization"] == "[REDACTED]"
    assert out["nested"]["note"] == "[REDACTED]"
    assert out["items"] == ["[REDACTED]", "fine"]


def test_budget_destructive_cap_and_exhaustion():
    b = Budget(total=10, destructive_cap_fraction=0.2)
    assert b.destructive_cap == 2
    assert b.would_exceed(EffectClass.READ_ONLY) is None
    b.consume(EffectClass.DESTRUCTIVE)
    b.consume(EffectClass.DESTRUCTIVE)
    assert "destructive cap" in b.would_exceed(EffectClass.DESTRUCTIVE)
    assert b.would_exceed(EffectClass.MUTATING) is None
    for _ in range(8):
        b.consume(EffectClass.MUTATING)
    assert b.remaining == 0
    assert b.would_exceed(EffectClass.MUTATING) == "budget exhausted"
    assert b.would_exceed(EffectClass.READ_ONLY) is None


def test_policy_refuses_unlisted_tools_and_effects_without_reset():
    p = SafetyPolicy(allowed_tools=frozenset({"get_release"}))
    b = Budget(total=5)
    assert "allow-list" in p.check("drop_db", EffectClass.MUTATING, b, True)
    assert p.check("get_release", EffectClass.READ_ONLY, b, False) is None
    p2 = SafetyPolicy(allowed_tools=frozenset({"publish_release"}))
    assert "proven reset" in p2.check("publish_release", EffectClass.MUTATING, b, False)
    assert p2.check("publish_release", EffectClass.MUTATING, b, True) is None
