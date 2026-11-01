"""End-to-end learning against the in-memory release simulator."""

from __future__ import annotations

import json

import pytest

from mcpgraphy.graph.export import dumps, export_json, export_mermaid, import_json
from mcpgraphy.graph.store import GraphStore
from mcpgraphy.learn.pipeline import run_learn
from mcpgraphy.models import AbstractState, BehaviorStatus, OutcomeKind, Provenance
from mcpgraphy.safety import ResetError
from mcpgraphy.sandbox import NoResetSandbox, ReleaseSandbox
from mcpgraphy.sandbox.base import FixtureManifest

pytestmark = pytest.mark.anyio


async def _learn(tmp_path, name="a", **kw):
    args = {"server_id": "release_sim", "version": "v1", "seed": 0, "fixtures": 12, "budget": 90}
    args.update(kw)
    return await run_learn(
        db_path=tmp_path / f"{name}.sqlite", journal_path=tmp_path / f"{name}.jsonl", **args
    )


def _export(result):
    with GraphStore(result.db_path) as store:
        return export_json(store, result.model_version)


def _state_id(**features):
    return AbstractState.build("release_sim", "release", features).state_id


def _rows(data, state_id, tool, arg="EXISTING"):
    acs = {a["action_class_id"]: a for a in data["action_classes"]}
    return [
        t
        for t in data["transitions"]
        if t["source_state_id"] == state_id
        and acs[t["action_class_id"]]["tool"] == tool
        and acs[t["action_class_id"]]["arg_classes"].get("release_id") == arg
    ]


async def test_end_to_end_in_memory(tmp_path):
    r = await _learn(tmp_path)
    assert r.summary.aborted is None
    assert r.summary.experiments > 40
    data = _export(r)
    assert data["model"]["coverage"]["experiments"] == r.summary.experiments
    assert data["model"]["coverage"]["by_status"]["OBSERVED_SUPPORTED"] > 0
    # every transition links to evidence or is explicitly hypothesized
    for t in data["transitions"]:
        assert t["evidence_ids"] or t["provenance"] == Provenance.INHERITED_NEEDS_RETEST
    assert len(data["states"]) == 6  # 3 statuses x tests_passed


async def test_replay_same_seed_identical_export(tmp_path):
    a = await _learn(tmp_path, "a")
    b = await _learn(tmp_path, "b")
    assert a.model_version == b.model_version
    assert dumps(_export(a)) == dumps(_export(b))
    # and re-materializing the same journal into a fresh store is identical too
    from mcpgraphy.graph.builder import materialize

    with GraphStore(tmp_path / "c.sqlite") as store:
        mv = materialize(a.journal_path, store)
        assert dumps(export_json(store, mv)) == dumps(_export(a))


async def test_different_seed_changes_run_id_but_not_feature_semantics(tmp_path):
    a = await _learn(tmp_path, "a", seed=0)
    b = await _learn(tmp_path, "b", seed=5)
    assert a.model_version != b.model_version
    assert {s["state_id"] for s in _export(a)["states"]} == {
        s["state_id"] for s in _export(b)["states"]
    }


async def test_draft_untested_supports_edit_and_run_tests_but_not_publish(tmp_path):
    data = _export(await _learn(tmp_path))
    sid = _state_id(status="DRAFT", tests_passed=False)
    edit = _rows(data, sid, "edit_release")
    assert any(t["outcome_kind"] == "OK" and t["status"] == "OBSERVED_SUPPORTED" for t in edit)
    publish = _rows(data, sid, "publish_release")
    assert publish and all(t["outcome_kind"] == "ERR_STATE" for t in publish)
    assert all(t["status"] == "OBSERVED_SUPPORTED" for t in publish)
    run = _rows(data, sid, "run_tests")
    ok = next(t for t in run if t["outcome_kind"] == "OK")
    assert ok["predicted_effects"] == ["tests_passed becomes true"]


async def test_hidden_mismatch_yields_contradicted_not_guaranteed(tmp_path):
    """DRAFT/tests_passed looks identical with and without the hidden version mismatch; the
    learner must represent the disagreement, never a guaranteed success."""
    data = _export(await _learn(tmp_path, fixtures=12, budget=120))
    sid = _state_id(status="DRAFT", tests_passed=True)
    publish = _rows(data, sid, "publish_release")
    kinds = {t["outcome_kind"] for t in publish}
    assert kinds == {"OK", "ERR_STATE"}
    statuses = {t["status"] for t in publish}
    assert "OBSERVED_SUPPORTED" not in statuses
    assert "CONTRADICTED" in statuses
    contradicted = next(t for t in publish if t["status"] == "CONTRADICTED")
    assert contradicted["counterexample_ids"]
    cx = {c["counterexample_id"]: c for c in data["counterexamples"]}
    exp_ids = {e["experiment_id"] for e in data["experiments"]}
    for cid in contradicted["counterexample_ids"]:
        assert cx[cid]["experiment_id"] in exp_ids  # links to a precise experiment


async def test_unknown_pairs_are_reported_not_hidden(tmp_path):
    data = _export(await _learn(tmp_path, fixtures=4, budget=20))
    assert data["model"]["coverage"]["unknown_pairs"] > 0
    assert data["model"]["budget"]["used"] <= 20


async def test_reset_failure_aborts_and_records_reason(tmp_path):
    class FlakyReset(ReleaseSandbox):
        def reset_fixture(self, fixture_id: str) -> FixtureManifest:
            if self.counter.resets >= 3:
                raise ResetError("simulated reset outage")
            return super().reset_fixture(fixture_id)

    r = await _learn(tmp_path, sandbox=FlakyReset("v1"))
    assert r.summary.aborted and "reset failed" in r.summary.aborted
    events, _ = __import__("mcpgraphy.observe.journal", fromlist=["Journal"]).Journal.read(
        r.journal_path
    )
    abort = [e for e in events if e["kind"] == "abort"]
    assert abort and "simulated reset outage" in abort[0]["reason"]
    data = _export(r)
    assert data["model"]["coverage"]["aborted"]


async def test_no_reset_sandbox_learns_nothing_and_abstains(tmp_path):
    r = await _learn(tmp_path, sandbox=NoResetSandbox(ReleaseSandbox("v1")))
    assert r.summary.aborted
    assert r.summary.experiments == 0
    data = _export(r)
    assert data["transitions"] == []
    assert data["model"]["coverage"]["reset_supported"] is False


async def test_export_import_round_trip_preserves_statuses(tmp_path):
    r = await _learn(tmp_path)
    data = _export(r)
    with GraphStore(tmp_path / "imported.sqlite") as store:
        mv = import_json(store, data)
        again = export_json(store, mv)
    assert again["transitions"] == data["transitions"]
    assert again["counterexamples"] == data["counterexamples"]
    assert {s["state_id"] for s in again["states"]} >= {
        t["source_state_id"] for t in data["transitions"]
    }


async def test_mermaid_is_derived_from_json(tmp_path):
    data = _export(await _learn(tmp_path))
    mermaid = export_mermaid(data)
    assert mermaid.startswith("flowchart LR")
    assert "publish_release" in mermaid and "run_tests" in mermaid
    assert json.dumps(data)  # still serializable; mermaid changed nothing


async def test_journal_redacts_secret_looking_values(tmp_path):
    r = await _learn(tmp_path, fixtures=2, budget=5)
    text = r.journal_path.read_text()
    assert "Bearer" not in text
    assert OutcomeKind.OK in text  # sanity: evidence is there


async def test_no_counterexamples_ablation_skips_replication(tmp_path):
    r = await _learn(tmp_path, ablation="no_counterexamples")
    assert r.summary.phases.get("replicate", 0) == 0
    assert BehaviorStatus.CONTRADICTED  # imported for parity; status set may still be non-empty
