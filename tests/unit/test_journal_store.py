"""Journal durability and store idempotency."""

from __future__ import annotations

import json

from mcpgraphy.graph.store import GraphStore
from mcpgraphy.models import (
    AbstractState,
    ActionClass,
    BehaviorStatus,
    EffectClass,
    ModelRecord,
    OutcomeKind,
    Provenance,
    Transition,
)
from mcpgraphy.observe.journal import Journal


def test_append_is_readable_line_by_line_and_seq_continues(tmp_journal):
    with Journal(tmp_journal) as j:
        j.append("run_start", {"run_id": "r", "api_key": "sekrit"})
        j.append("experiment", {"run_id": "r", "n": 1})
    with Journal(tmp_journal) as j:
        assert j.seq == 2
        j.append("run_end", {"run_id": "r"})
    events, skipped = Journal.read(tmp_journal)
    assert [e["kind"] for e in events] == ["run_start", "experiment", "run_end"]
    assert [e["seq"] for e in events] == [1, 2, 3]
    assert events[0]["api_key"] == "[REDACTED]"
    assert skipped == 0


def test_truncated_last_line_is_skipped_and_reported(tmp_journal):
    with Journal(tmp_journal) as j:
        j.append("run_start", {"run_id": "r"})
    with open(tmp_journal, "a") as fh:
        fh.write('{"kind": "experiment", "seq": 2, "trunc')  # power loss mid-write
    events, skipped = Journal.read(tmp_journal)
    assert len(events) == 1 and skipped == 1


def test_migrations_apply_once(tmp_db):
    with GraphStore(tmp_db) as store:
        assert store.migrations_applied == [1, 2]
    with GraphStore(tmp_db) as store:
        assert store.migrations_applied == []


def _transition(mv, status, retest=False):
    s = AbstractState.build("release_sim", "release", {"status": "DRAFT"})
    ac = ActionClass.build(
        "release_sim", "publish_release", {"release_id": "EXISTING"}, EffectClass.MUTATING
    )
    return (
        s,
        ac,
        Transition(
            transition_id=Transition.make_id(
                mv, s.state_id, ac.action_class_id, OutcomeKind.OK, None
            ),
            model_version=mv,
            source_state_id=s.state_id,
            action_class_id=ac.action_class_id,
            outcome_kind=OutcomeKind.OK,
            target_state_id=None,
            status=status,
            provenance=Provenance.INHERITED_NEEDS_RETEST,
            retest_required=retest,
        ),
    )


def test_unknown_survives_store_round_trip(tmp_db):
    mv = "graph_x"
    s, ac, t = _transition(mv, BehaviorStatus.UNKNOWN, retest=True)
    with GraphStore(tmp_db) as store, store.conn:
        store.save_states([s])
        store.save_action_classes([ac])
        store.save_model(
            ModelRecord(
                model_version=mv,
                server_id="release_sim",
                server_version_id="sv",
                run_id="r",
                seed=0,
            )
        )
        store.replace_transitions(mv, [t], [])
    with GraphStore(tmp_db) as store:
        rows = store.transitions(mv)
        assert rows[0].status is BehaviorStatus.UNKNOWN
        assert rows[0].retest_required is True
        assert rows[0].predicted_effects == ()


def test_replace_transitions_is_idempotent(tmp_db):
    mv = "graph_x"
    s, ac, t = _transition(mv, BehaviorStatus.OBSERVED_SUPPORTED)
    with GraphStore(tmp_db) as store:
        with store.conn:
            store.save_states([s])
            store.save_action_classes([ac])
            store.replace_transitions(mv, [t], [])
        with store.conn:
            store.replace_transitions(mv, [t], [])
        assert len(store.transitions(mv)) == 1
        assert (
            json.loads(store.conn.execute("SELECT guard_json FROM transitions").fetchone()[0])
            is None
        )
