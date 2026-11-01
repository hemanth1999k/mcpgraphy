"""Canonical ids are deterministic, order-independent and scoped."""

from mcpgraphy.models import (
    AbstractState,
    ActionClass,
    AuthStatus,
    Availability,
    BehaviorStatus,
    EffectClass,
    Offer,
    OfferAuthorization,
    OfferBehavior,
    OfferUncertainty,
    ResourceRef,
    ServerVersion,
    ToolSpec,
    canonical_json,
    content_id,
)


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": [1, {"d": 2, "c": 3}]}) == canonical_json(
        {"a": [1, {"c": 3, "d": 2}], "b": 1}
    )


def test_state_id_deterministic_across_key_order():
    a = AbstractState.build("release_sim", "release", {"status": "DRAFT", "tests_passed": True})
    b = AbstractState.build("release_sim", "release", {"tests_passed": True, "status": "DRAFT"})
    assert a.state_id == b.state_id
    assert a.state_id.startswith("st_") and len(a.state_id) == 3 + 16


def test_state_id_differs_by_server_id():
    a = AbstractState.build("release_sim", "release", {"status": "DRAFT"})
    b = AbstractState.build("other", "release", {"status": "DRAFT"})
    assert a.state_id != b.state_id


def test_resource_ref_differs_by_tenant_same_local_id():
    a = ResourceRef.build("release_sim", "t1", "release", "demo-7")
    b = ResourceRef.build("release_sim", "t2", "release", "demo-7")
    assert a.resource_ref_id != b.resource_ref_id
    assert a.render() == b.render() == "release:demo-7"


def test_action_class_label_and_id_stable():
    a = ActionClass.build(
        "release_sim", "publish_release", {"release_id": "EXISTING"}, EffectClass.MUTATING
    )
    b = ActionClass.build(
        "release_sim", "publish_release", {"release_id": "EXISTING"}, EffectClass.DESTRUCTIVE
    )
    assert a.label == "publish_release(release_id=EXISTING)"
    # effect class is metadata, not identity
    assert a.action_class_id == b.action_class_id


def test_server_fingerprint_changes_with_schema_only():
    t1 = ToolSpec(name="run_tests", description="x", input_schema={"properties": {"id": {}}})
    t2 = ToolSpec(name="run_tests", description="different words", input_schema=t1.input_schema)
    t3 = ToolSpec(
        name="run_tests", description="x", input_schema={"properties": {"id": {}, "suite": {}}}
    )
    assert ServerVersion.fingerprint_of([t1]) == ServerVersion.fingerprint_of([t2])
    assert ServerVersion.fingerprint_of([t1]) != ServerVersion.fingerprint_of([t3])


def test_content_id_prefix_and_length():
    cid = content_id("exp", {"a": 1})
    assert cid.startswith("exp_") and len(cid) == 4 + 16


def test_offer_matches_spec_5_4_keys():
    offer = Offer(
        offer_id="offer_x",
        server_id="release_sim_v1",
        resource_ref="release:demo-7",
        tool="publish_release",
        argument_template={"release_id": "demo-7"},
        behavior=OfferBehavior(
            status=BehaviorStatus.OBSERVED_SUPPORTED,
            observed_state={"status": "DRAFT", "tests_passed": True},
            required_predicates=("status == DRAFT", "tests_passed == true"),
            predicted_effects=("status becomes PUBLISHED",),
            evidence_ids=("exp_041", "exp_053"),
            model_version="graph_2026_10_31_01",
        ),
        authorization=OfferAuthorization(required=True, granted=False, status=AuthStatus.REQUIRED),
        availability=Availability.CANDIDATE_REQUIRES_APPROVAL,
        uncertainty=OfferUncertainty(
            untested_parameter_regions=(), possible_unobserved_effects=True
        ),
    )
    d = offer.to_dict()
    spec_keys = {
        "offer_id",
        "server_id",
        "resource_ref",
        "tool",
        "argument_template",
        "behavior",
        "authorization",
        "availability",
        "uncertainty",
    }
    assert spec_keys <= set(d)
    assert {
        "status",
        "observed_state",
        "required_predicates",
        "predicted_effects",
        "evidence_ids",
        "model_version",
    } <= set(d["behavior"])
    assert {"required", "granted"} <= set(d["authorization"])
    assert {"untested_parameter_regions", "possible_unobserved_effects"} <= set(d["uncertainty"])
    assert d["availability"] == "CANDIDATE_REQUIRES_APPROVAL"
