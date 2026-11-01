"""Guard induction and status assignment on synthetic evidence.

These tests validate both directions: nothing unprobed becomes OBSERVED_SUPPORTED (no false
positives) and a reproduced behavior is not left UNKNOWN (no false negatives).
"""

from mcpgraphy.learn.guards import induce
from mcpgraphy.models import (
    AbstractState,
    ActionClass,
    BehaviorStatus,
    EffectClass,
    Experiment,
    Outcome,
    OutcomeKind,
    Provenance,
    Snapshot,
)

SERVER = "release_sim"
MV = "graph_test"


def _state(**features) -> AbstractState:
    return AbstractState.build(SERVER, "release", features)


DRAFT_F = _state(status="DRAFT", tests_passed=False)
DRAFT_T = _state(status="DRAFT", tests_passed=True)
PUB_T = _state(status="PUBLISHED", tests_passed=True)
ARCH_F = _state(status="ARCHIVED", tests_passed=False)
STATES = {s.state_id: s for s in (DRAFT_F, DRAFT_T, PUB_T, ARCH_F)}

PUBLISH = ActionClass.build(
    SERVER, "publish_release", {"release_id": "EXISTING"}, EffectClass.MUTATING
)
PUBLISH_STALE = ActionClass.build(
    SERVER, "publish_release", {"release_id": "STALE_DELETED"}, EffectClass.MUTATING
)


def _exp(seq, src, ac, kind, dst=None, fixture="fx_a"):
    snap = Snapshot(
        tool="get_release",
        raw={"status": src.features["status"]},
        outcome=Outcome(kind=OutcomeKind.OK),
    )
    return Experiment(
        experiment_id=f"exp_{seq:04d}",
        run_id="run",
        seq=seq,
        server_version_id="sv",
        fixture_id=fixture,
        action_class_id=ac.action_class_id,
        resource_ref="release:x",
        args={"release_id": "x"},
        before=snap,
        before_state_id=src.state_id,
        result=Outcome(kind=kind),
        after=snap,
        after_state_id=(dst or src).state_id,
        reset_confirmed=True,
        phase="tier0",
    )


def _rows(ind, src, ac):
    return [
        t
        for t in ind.transitions
        if t.source_state_id == src.state_id and t.action_class_id == ac.action_class_id
    ]


def test_unprobed_state_is_unknown_not_supported():
    exps = [
        _exp(1, DRAFT_T, PUBLISH, OutcomeKind.OK, PUB_T),
        _exp(2, DRAFT_F, PUBLISH, OutcomeKind.ERR_STATE),
    ]
    ind = induce(MV, exps, dict(STATES), [PUBLISH.action_class_id])
    # ARCHIVED never probed, does not satisfy the guard (status == DRAFT) -> no row at all
    assert _rows(ind, ARCH_F, PUBLISH) == []
    assert all(
        t.status is not BehaviorStatus.OBSERVED_SUPPORTED
        for t in ind.transitions
        if t.source_state_id == ARCH_F.state_id
    )


def test_reproduced_behavior_is_observed_supported_not_unknown():
    exps = [
        _exp(1, DRAFT_T, PUBLISH, OutcomeKind.OK, PUB_T, "fx_a"),
        _exp(2, DRAFT_T, PUBLISH, OutcomeKind.OK, PUB_T, "fx_b"),
        _exp(3, ARCH_F, PUBLISH, OutcomeKind.ERR_STATE),
    ]
    ind = induce(MV, exps, dict(STATES), [PUBLISH.action_class_id])
    ok = _rows(ind, DRAFT_T, PUBLISH)
    assert len(ok) == 1 and ok[0].status is BehaviorStatus.OBSERVED_SUPPORTED
    assert ok[0].evidence_ids == ("exp_0001", "exp_0002")
    assert ok[0].predicted_effects == ("status becomes PUBLISHED",)
    err = _rows(ind, ARCH_F, PUBLISH)
    assert err[0].status is BehaviorStatus.OBSERVED_SUPPORTED  # a reproduced failure
    assert err[0].outcome_kind is OutcomeKind.ERR_STATE


def test_guard_covered_unprobed_state_is_conditional_with_hypothesized_provenance():
    # success in DRAFT/T, failure in ARCHIVED/F: one feature suffices to separate them, so the
    # guard is minimized to a single predicate (deterministically, in sorted feature order).
    exps = [
        _exp(1, DRAFT_T, PUBLISH, OutcomeKind.OK, PUB_T),
        _exp(2, ARCH_F, PUBLISH, OutcomeKind.ERR_STATE),
    ]
    ind = induce(MV, exps, dict(STATES), [PUBLISH.action_class_id])
    guard = ind.guards[PUBLISH.action_class_id]
    assert len(guard.all_of) == 1
    assert guard.holds(DRAFT_T.features) and not guard.holds(ARCH_F.features)
    covered_unprobed = [s for s in (DRAFT_F, PUB_T) if guard.holds(s.features)]
    assert covered_unprobed  # at least one unprobed state is covered by the hypothesis
    for s in covered_unprobed:
        hyp = _rows(ind, s, PUBLISH)
        assert len(hyp) == 1
        assert hyp[0].status is BehaviorStatus.CONDITIONAL
        assert hyp[0].provenance is Provenance.HYPOTHESIZED_BY_GUARD
        assert hyp[0].evidence_ids == ("exp_0001",)
    assert guard.coverage_total == 1 + len(covered_unprobed) and guard.coverage_tested == 1


def test_perfect_separation_keeps_needed_features():
    # OK only in DRAFT/T, failures in DRAFT/F and PUBLISHED: both features needed
    exps = [
        _exp(1, DRAFT_T, PUBLISH, OutcomeKind.OK, PUB_T),
        _exp(2, DRAFT_F, PUBLISH, OutcomeKind.ERR_STATE),
        _exp(3, PUB_T, PUBLISH, OutcomeKind.ERR_STATE),
    ]
    ind = induce(MV, exps, dict(STATES), [PUBLISH.action_class_id])
    guard = ind.guards[PUBLISH.action_class_id]
    assert [str(p) for p in guard.all_of] == ["status == DRAFT", "tests_passed == true"]


def test_no_negative_evidence_keeps_full_conjunction():
    exps = [_exp(1, DRAFT_T, PUBLISH, OutcomeKind.OK, PUB_T)]
    ind = induce(MV, exps, dict(STATES), [PUBLISH.action_class_id])
    guard = ind.guards[PUBLISH.action_class_id]
    assert len(guard.all_of) == 2  # nothing dropped without a counterexample to justify it


def test_mixed_outcomes_record_multiple_successors_no_overwrite():
    exps = [
        _exp(1, DRAFT_T, PUBLISH, OutcomeKind.OK, PUB_T, "fx_a"),
        _exp(2, DRAFT_T, PUBLISH, OutcomeKind.ERR_STATE, None, "fx_b"),
    ]
    ind = induce(MV, exps, dict(STATES), [PUBLISH.action_class_id])
    rows = _rows(ind, DRAFT_T, PUBLISH)
    assert {r.outcome_kind for r in rows} == {OutcomeKind.OK, OutcomeKind.ERR_STATE}
    earliest = next(r for r in rows if r.outcome_kind is OutcomeKind.OK)
    later = next(r for r in rows if r.outcome_kind is OutcomeKind.ERR_STATE)
    assert earliest.status is BehaviorStatus.CONTRADICTED
    assert later.status is BehaviorStatus.CONDITIONAL
    assert len(ind.counterexamples) == 1
    cx = ind.counterexamples[0]
    assert cx.transition_id == earliest.transition_id and cx.experiment_id == "exp_0002"
    assert earliest.counterexample_ids == (cx.counterexample_id,)
    assert "hidden condition suspected" in ind.guards[PUBLISH.action_class_id].note


def test_candidate_guard_contradicted_by_new_fixture():
    base = [_exp(1, DRAFT_T, PUBLISH, OutcomeKind.OK, PUB_T, "fx_a")]
    before = induce(MV, base, dict(STATES), [PUBLISH.action_class_id])
    assert _rows(before, DRAFT_T, PUBLISH)[0].status is BehaviorStatus.OBSERVED_SUPPORTED
    after = induce(
        MV,
        base + [_exp(2, DRAFT_T, PUBLISH, OutcomeKind.ERR_STATE, None, "fx_c")],
        dict(STATES),
        [PUBLISH.action_class_id],
    )
    ok_row = next(r for r in _rows(after, DRAFT_T, PUBLISH) if r.outcome_kind is OutcomeKind.OK)
    assert ok_row.status is BehaviorStatus.CONTRADICTED
    assert ok_row.counterexample_ids


def test_stale_id_class_not_merged_with_existing():
    exps = [
        _exp(1, DRAFT_T, PUBLISH, OutcomeKind.OK, PUB_T),
        _exp(2, DRAFT_T, PUBLISH_STALE, OutcomeKind.ERR_NOT_FOUND),
    ]
    ind = induce(MV, exps, dict(STATES), [PUBLISH.action_class_id, PUBLISH_STALE.action_class_id])
    assert _rows(ind, DRAFT_T, PUBLISH)[0].status is BehaviorStatus.OBSERVED_SUPPORTED
    stale = _rows(ind, DRAFT_T, PUBLISH_STALE)
    assert stale[0].outcome_kind is OutcomeKind.ERR_NOT_FOUND
    assert ind.guards[PUBLISH_STALE.action_class_id] is None


def test_unknown_or_hypothesized_rows_never_claim_observed_effects():
    exps = [
        _exp(1, DRAFT_T, PUBLISH, OutcomeKind.OK, PUB_T),
        _exp(2, ARCH_F, PUBLISH, OutcomeKind.ERR_STATE),
    ]
    ind = induce(MV, exps, dict(STATES), [PUBLISH.action_class_id])
    for t in ind.transitions:
        if t.provenance is Provenance.HYPOTHESIZED_BY_GUARD:
            assert t.status is BehaviorStatus.CONDITIONAL
            assert t.evidence_ids  # linked to the experiments the guard came from
