"""Server version drift: inherited edges are flagged for retest and contradicted."""

from __future__ import annotations

import pytest

from mcpgraphy.graph.export import export_json
from mcpgraphy.graph.store import GraphStore
from mcpgraphy.learn.pipeline import run_learn
from mcpgraphy.models import AbstractState, BehaviorStatus, Provenance

pytestmark = pytest.mark.anyio


def _sid(**features):
    return AbstractState.build("release_sim", "release", features).state_id


async def test_v1_model_on_v2_flags_edges_for_retest_and_contradicts_archive(tmp_path):
    db = tmp_path / "drift.sqlite"
    v1 = await run_learn("release_sim", "v1", 0, 12, 90, db, tmp_path / "v1.jsonl")
    # drift: retest with a tiny budget so most inherited rows stay UNKNOWN/retest_required
    v2 = await run_learn(
        "release_sim", "v2", 0, 12, 12, db, tmp_path / "v2.jsonl", base_model=v1.model_version
    )
    with GraphStore(db) as store:
        data = export_json(store, v2.model_version)
    assert data["model"]["parent_model_version"] == v1.model_version
    assert data["model"]["fingerprint"] != export_json_fingerprint(db, v1.model_version)
    inherited = [
        t for t in data["transitions"] if t["provenance"] == Provenance.INHERITED_NEEDS_RETEST
    ]
    assert inherited
    pending = [t for t in inherited if t["retest_required"]]
    assert pending and all(t["status"] == BehaviorStatus.UNKNOWN for t in pending)
    # run_tests schema changed in v2 -> its v1 rows are dropped, not inherited
    acs = {a["action_class_id"]: a for a in data["action_classes"]}
    assert not any(acs.get(t["action_class_id"], {}).get("tool") == "run_tests" for t in inherited)


async def test_retest_with_budget_contradicts_v1_archive_from_draft(tmp_path):
    db = tmp_path / "drift.sqlite"
    v1 = await run_learn("release_sim", "v1", 0, 12, 90, db, tmp_path / "v1.jsonl")
    v2 = await run_learn(
        "release_sim", "v2", 0, 12, 90, db, tmp_path / "v2.jsonl", base_model=v1.model_version
    )
    assert v2.summary.phases["retest"] > 0
    with GraphStore(db) as store:
        data = export_json(store, v2.model_version)
    acs = {a["action_class_id"]: a for a in data["action_classes"]}
    draft = _sid(status="DRAFT", tests_passed=False)
    rows = [
        t
        for t in data["transitions"]
        if t["source_state_id"] == draft
        and acs[t["action_class_id"]]["tool"] == "archive_release"
        and acs[t["action_class_id"]]["arg_classes"]["release_id"] == "EXISTING"
    ]
    kinds = {(t["outcome_kind"], t["status"], t["provenance"]) for t in rows}
    # the old "archive from DRAFT fails" edge is contradicted by fresh v2 evidence ...
    assert ("ERR_STATE", "CONTRADICTED", Provenance.INHERITED_NEEDS_RETEST) in kinds
    # ... and the new behavior is observed directly
    assert any(k[0] == "OK" and k[2] == Provenance.SANDBOX_PROBE for k in kinds)
    cx = [
        c for c in data["counterexamples"] if c["predicted"].get("from_model") == v1.model_version
    ]
    assert cx


def export_json_fingerprint(db, model_version):
    with GraphStore(db) as store:
        return export_json(store, model_version)["model"]["fingerprint"]
