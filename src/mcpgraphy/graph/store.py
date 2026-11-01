"""SQLite graph store: content-id upserts, one transaction per materialization."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mcpgraphy.graph.schema import apply_migrations
from mcpgraphy.models import (
    AbstractState,
    ActionClass,
    BehaviorStatus,
    Counterexample,
    Experiment,
    Guard,
    ModelRecord,
    ServerVersion,
    ToolSpec,
    Transition,
    canonical_json,
)


class GraphStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = ":memory:" if path is None else str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL") if self.path != ":memory:" else None
        self.migrations_applied = apply_migrations(self.conn)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> GraphStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ writes
    def save_server_version(self, sv: ServerVersion) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO server_versions VALUES (?, ?, ?, ?, ?)",
            (
                sv.server_version_id,
                sv.server_id,
                sv.version,
                sv.fingerprint,
                canonical_json([t.to_dict() for t in sv.tools]),
            ),
        )
        for t in sv.tools:
            self.conn.execute(
                "INSERT OR REPLACE INTO tools VALUES (?, ?, ?, ?, ?)",
                (
                    sv.server_version_id,
                    t.name,
                    t.description,
                    canonical_json(t.input_schema),
                    canonical_json(t.annotations),
                ),
            )

    def save_states(self, states: Iterable[AbstractState]) -> None:
        self.conn.executemany(
            "INSERT OR IGNORE INTO abstract_states VALUES (?, ?, ?, ?)",
            [
                (s.state_id, s.server_id, s.resource_class, canonical_json(s.features))
                for s in states
            ],
        )

    def save_action_classes(self, acs: Iterable[ActionClass]) -> None:
        self.conn.executemany(
            "INSERT OR IGNORE INTO action_classes VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    a.action_class_id,
                    a.server_id,
                    a.tool,
                    canonical_json(a.arg_classes),
                    a.label,
                    str(a.effect_class),
                )
                for a in acs
            ],
        )

    def save_experiments(self, exps: Iterable[Experiment]) -> None:
        rows = []
        obs = []
        for e in exps:
            rows.append(
                (
                    e.experiment_id,
                    e.run_id,
                    e.seq,
                    e.fixture_id,
                    e.server_version_id,
                    e.action_class_id,
                    e.resource_ref,
                    canonical_json(e.args),
                    canonical_json(e.before.to_dict()),
                    e.before_state_id,
                    canonical_json(e.result.to_dict()),
                    str(e.result.kind),
                    canonical_json(e.after.to_dict()) if e.after else None,
                    e.after_state_id,
                    int(e.reset_confirmed),
                    e.phase,
                    e.created_at,
                )
            )
            obs.append(
                (
                    f"{e.experiment_id}:before",
                    e.experiment_id,
                    "before",
                    e.before.tool,
                    canonical_json(e.before.raw) if e.before.raw is not None else None,
                    e.before_state_id,
                )
            )
            if e.after is not None:
                obs.append(
                    (
                        f"{e.experiment_id}:after",
                        e.experiment_id,
                        "after",
                        e.after.tool,
                        canonical_json(e.after.raw) if e.after.raw is not None else None,
                        e.after_state_id,
                    )
                )
        self.conn.executemany(
            "INSERT OR REPLACE INTO experiments VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.executemany("INSERT OR REPLACE INTO observations VALUES (?, ?, ?, ?, ?, ?)", obs)

    def save_events(self, run_id: str, events: Iterable[dict[str, Any]]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO journal_events VALUES (?, ?, ?, ?)",
            [(run_id, ev["seq"], ev["kind"], canonical_json(ev)) for ev in events],
        )

    def save_model(self, m: ModelRecord) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                m.model_version,
                m.server_id,
                m.server_version_id,
                m.run_id,
                m.seed,
                canonical_json(m.budget),
                canonical_json(m.coverage),
                m.parent_model_version,
                m.created_at,
            ),
        )

    def replace_transitions(
        self,
        model_version: str,
        transitions: Iterable[Transition],
        counterexamples: Iterable[Counterexample],
    ) -> None:
        self.conn.execute("DELETE FROM transitions WHERE model_version = ?", (model_version,))
        self.conn.execute(
            "DELETE FROM transition_evidence WHERE transition_id IN "
            "(SELECT transition_id FROM transitions WHERE model_version = ?)",
            (model_version,),
        )
        self.conn.execute("DELETE FROM counterexamples WHERE model_version = ?", (model_version,))
        for t in transitions:
            self.conn.execute(
                "INSERT INTO transitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    t.transition_id,
                    t.model_version,
                    t.source_state_id,
                    t.action_class_id,
                    str(t.outcome_kind),
                    t.target_state_id,
                    str(t.status),
                    canonical_json(t.guard.to_dict()) if t.guard else "null",
                    str(t.provenance),
                    len(t.evidence_ids),
                    canonical_json(list(t.evidence_ids)),
                    len(t.counterexample_ids),
                    canonical_json(list(t.counterexample_ids)),
                    canonical_json(list(t.predicted_effects)),
                    int(t.retest_required),
                    t.created_at,
                ),
            )
            self.conn.executemany(
                "INSERT OR IGNORE INTO transition_evidence VALUES (?, ?, ?)",
                [(t.transition_id, eid, "supports") for eid in t.evidence_ids],
            )
        for c in counterexamples:
            self.conn.execute(
                "INSERT OR REPLACE INTO counterexamples VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    c.counterexample_id,
                    c.model_version,
                    c.transition_id,
                    c.experiment_id,
                    canonical_json(c.predicted),
                    canonical_json(c.observed),
                    c.created_at,
                ),
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO transition_evidence VALUES (?, ?, ?)",
                (c.transition_id, c.experiment_id, "contradicts"),
            )

    def save_execution(self, rec: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO executions VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rec["execution_id"],
                rec["idempotency_key"],
                rec["offer_id"],
                rec["server_id"],
                rec["tool"],
                canonical_json(rec["args"]),
                rec["decision"],
                rec["reason"],
                canonical_json(rec["before_state"])
                if rec.get("before_state") is not None
                else None,
                canonical_json(rec["after_state"]) if rec.get("after_state") is not None else None,
                rec.get("outcome_kind"),
                canonical_json(rec["predicted_effects"]),
                canonical_json(rec["observed_effects"])
                if rec.get("observed_effects") is not None
                else None,
                None if rec.get("prediction_matched") is None else int(rec["prediction_matched"]),
                rec["model_version"],
                canonical_json(rec["transition_ids"]),
                rec.get("counterexample_id"),
                rec["created_at"],
            ),
        )

    def execution(self, execution_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM executions WHERE execution_id = ?", (execution_id,)
        ).fetchone()
        return _execution(row) if row else None

    def execution_by_key(self, idempotency_key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM executions WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        return _execution(row) if row else None

    def executions_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0])

    def contradict_transition(self, transition_id: str, cx: Counterexample) -> None:
        """Conservative update only: a transition may lose trust, never gain it."""
        row = self.conn.execute(
            "SELECT counterexamples_json FROM transitions WHERE transition_id = ?", (transition_id,)
        ).fetchone()
        if row is None:
            return
        ids = json.loads(row["counterexamples_json"])
        if cx.counterexample_id not in ids:
            ids.append(cx.counterexample_id)
        self.conn.execute(
            "UPDATE transitions SET status = ?, counterexamples_json = ?, contradiction_count = ? "
            "WHERE transition_id = ?",
            (str(BehaviorStatus.CONTRADICTED), canonical_json(ids), len(ids), transition_id),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO counterexamples VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                cx.counterexample_id,
                cx.model_version,
                cx.transition_id,
                cx.experiment_id,
                canonical_json(cx.predicted),
                canonical_json(cx.observed),
                cx.created_at,
            ),
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO transition_evidence VALUES (?, ?, ?)",
            (transition_id, cx.experiment_id, "contradicts"),
        )

    # ------------------------------------------------------------------ reads
    def model(self, model_version: str) -> ModelRecord | None:
        row = self.conn.execute(
            "SELECT * FROM models WHERE model_version = ?", (model_version,)
        ).fetchone()
        return _model(row) if row else None

    def latest_model(self, server_id: str) -> ModelRecord | None:
        row = self.conn.execute(
            "SELECT * FROM models WHERE server_id = ? ORDER BY rowid DESC LIMIT 1", (server_id,)
        ).fetchone()
        return _model(row) if row else None

    def models(self, server_id: str | None = None) -> list[ModelRecord]:
        q = (
            "SELECT * FROM models"
            + (" WHERE server_id = ?" if server_id else "")
            + " ORDER BY rowid"
        )
        rows = self.conn.execute(q, (server_id,) if server_id else ()).fetchall()
        return [_model(r) for r in rows]

    def resolve_model_version(self, server_id: str, model: str | None) -> str:
        if model in (None, "latest"):
            m = self.latest_model(server_id)
            if m is None:
                raise LookupError(f"no model for {server_id}; run `learn` first")
            return m.model_version
        if self.model(model) is None:
            raise LookupError(f"unknown model version {model}")
        return model

    def server_version(self, server_version_id: str) -> ServerVersion | None:
        row = self.conn.execute(
            "SELECT * FROM server_versions WHERE server_version_id = ?", (server_version_id,)
        ).fetchone()
        if not row:
            return None
        tools = tuple(ToolSpec(**t) for t in json.loads(row["tools_json"]))
        return ServerVersion(
            server_version_id=row["server_version_id"],
            server_id=row["server_id"],
            version=row["version"],
            fingerprint=row["fingerprint"],
            tools=tools,
        )

    def state(self, state_id: str) -> AbstractState | None:
        row = self.conn.execute(
            "SELECT * FROM abstract_states WHERE state_id = ?", (state_id,)
        ).fetchone()
        return _state(row) if row else None

    def states_for_model(self, model_version: str) -> dict[str, AbstractState]:
        rows = self.conn.execute(
            """
            SELECT DISTINCT s.* FROM abstract_states s
            WHERE s.state_id IN (
              SELECT source_state_id FROM transitions WHERE model_version = :mv
              UNION SELECT target_state_id FROM transitions
                    WHERE model_version = :mv AND target_state_id IS NOT NULL
              UNION SELECT before_state_id FROM experiments e JOIN models m ON m.run_id = e.run_id
                    WHERE m.model_version = :mv
              UNION SELECT after_state_id FROM experiments e JOIN models m ON m.run_id = e.run_id
                    WHERE m.model_version = :mv AND after_state_id IS NOT NULL
            )
            ORDER BY s.state_id
            """,
            {"mv": model_version},
        ).fetchall()
        return {r["state_id"]: _state(r) for r in rows}

    def action_classes(self, server_id: str) -> dict[str, ActionClass]:
        rows = self.conn.execute(
            "SELECT * FROM action_classes WHERE server_id = ? ORDER BY action_class_id",
            (server_id,),
        ).fetchall()
        return {r["action_class_id"]: _action_class(r) for r in rows}

    def transitions(self, model_version: str) -> list[Transition]:
        rows = self.conn.execute(
            "SELECT * FROM transitions WHERE model_version = ? ORDER BY transition_id",
            (model_version,),
        ).fetchall()
        return [_transition(r) for r in rows]

    def transitions_for(
        self, model_version: str, source_state_id: str, action_class_id: str
    ) -> list[Transition]:
        rows = self.conn.execute(
            "SELECT * FROM transitions WHERE model_version = ? AND source_state_id = ? "
            "AND action_class_id = ? ORDER BY transition_id",
            (model_version, source_state_id, action_class_id),
        ).fetchall()
        return [_transition(r) for r in rows]

    def counterexamples(self, model_version: str) -> list[Counterexample]:
        rows = self.conn.execute(
            "SELECT * FROM counterexamples WHERE model_version = ? ORDER BY counterexample_id",
            (model_version,),
        ).fetchall()
        return [
            Counterexample(
                counterexample_id=r["counterexample_id"],
                model_version=r["model_version"],
                transition_id=r["transition_id"],
                experiment_id=r["experiment_id"],
                predicted=json.loads(r["predicted_json"]),
                observed=json.loads(r["observed_json"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def experiments(self, run_id: str) -> list[Experiment]:
        rows = self.conn.execute(
            "SELECT * FROM experiments WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
        return [_experiment(r) for r in rows]

    def experiment(self, experiment_id: str) -> Experiment | None:
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        return _experiment(row) if row else None

    def events(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT payload_json FROM journal_events WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]


# ------------------------------------------------------------------ row mappers


def _execution(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "execution_id": r["execution_id"],
        "idempotency_key": r["idempotency_key"],
        "offer_id": r["offer_id"],
        "server_id": r["server_id"],
        "tool": r["tool"],
        "args": json.loads(r["args_json"]),
        "decision": r["decision"],
        "reason": r["reason"],
        "before_state": json.loads(r["before_json"]) if r["before_json"] else None,
        "after_state": json.loads(r["after_json"]) if r["after_json"] else None,
        "outcome_kind": r["outcome_kind"],
        "predicted_effects": json.loads(r["predicted_json"]),
        "observed_effects": json.loads(r["observed_json"]) if r["observed_json"] else None,
        "prediction_matched": None
        if r["prediction_matched"] is None
        else bool(r["prediction_matched"]),
        "model_version": r["model_version"],
        "transition_ids": json.loads(r["transition_ids_json"]),
        "counterexample_id": r["counterexample_id"],
        "created_at": r["created_at"],
    }


def _model(r: sqlite3.Row) -> ModelRecord:
    return ModelRecord(
        model_version=r["model_version"],
        server_id=r["server_id"],
        server_version_id=r["server_version_id"],
        run_id=r["run_id"],
        seed=r["seed"],
        budget=json.loads(r["budget_json"]),
        coverage=json.loads(r["coverage_json"]),
        parent_model_version=r["parent_model_version"],
        created_at=r["created_at"],
    )


def _state(r: sqlite3.Row) -> AbstractState:
    return AbstractState(
        state_id=r["state_id"],
        server_id=r["server_id"],
        resource_class=r["resource_class"],
        features=json.loads(r["features_json"]),
    )


def _action_class(r: sqlite3.Row) -> ActionClass:
    return ActionClass(
        action_class_id=r["action_class_id"],
        server_id=r["server_id"],
        tool=r["tool"],
        arg_classes=json.loads(r["arg_classes_json"]),
        effect_class=r["effect_class"],
    )


def _transition(r: sqlite3.Row) -> Transition:
    guard_raw = json.loads(r["guard_json"])
    return Transition(
        transition_id=r["transition_id"],
        model_version=r["model_version"],
        source_state_id=r["source_state_id"],
        action_class_id=r["action_class_id"],
        outcome_kind=r["outcome_kind"],
        target_state_id=r["target_state_id"],
        status=r["status"],
        guard=Guard(**guard_raw) if guard_raw else None,
        provenance=r["provenance"],
        evidence_ids=tuple(json.loads(r["evidence_json"])),
        counterexample_ids=tuple(json.loads(r["counterexamples_json"])),
        predicted_effects=tuple(json.loads(r["predicted_effects_json"])),
        retest_required=bool(r["retest_required"]),
        created_at=r["created_at"],
    )


def _experiment(r: sqlite3.Row) -> Experiment:
    return Experiment(
        experiment_id=r["experiment_id"],
        run_id=r["run_id"],
        seq=r["seq"],
        server_version_id=r["server_version_id"],
        fixture_id=r["fixture_id"],
        action_class_id=r["action_class_id"],
        resource_ref=r["resource_ref"],
        args=json.loads(r["args_json"]),
        before=json.loads(r["before_json"]),
        before_state_id=r["before_state_id"],
        result=json.loads(r["result_json"]),
        after=json.loads(r["after_json"]) if r["after_json"] else None,
        after_state_id=r["after_state_id"],
        reset_confirmed=bool(r["reset_confirmed"]),
        phase=r["phase"],
        created_at=r["created_at"],
    )
