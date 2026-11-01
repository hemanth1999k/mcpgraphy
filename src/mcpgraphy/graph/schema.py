"""SQLite schema with explicit migrations."""

from __future__ import annotations

import sqlite3

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS server_versions (
  server_version_id TEXT PRIMARY KEY,
  server_id TEXT NOT NULL,
  version TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  tools_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tools (
  server_version_id TEXT NOT NULL,
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  input_schema_json TEXT NOT NULL,
  annotations_json TEXT NOT NULL,
  PRIMARY KEY (server_version_id, name)
);
CREATE TABLE IF NOT EXISTS abstract_states (
  state_id TEXT PRIMARY KEY,
  server_id TEXT NOT NULL,
  resource_class TEXT NOT NULL,
  features_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS action_classes (
  action_class_id TEXT PRIMARY KEY,
  server_id TEXT NOT NULL,
  tool TEXT NOT NULL,
  arg_classes_json TEXT NOT NULL,
  label TEXT NOT NULL,
  effect_class TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predicates (
  predicate_id TEXT PRIMARY KEY,
  feature TEXT NOT NULL,
  op TEXT NOT NULL,
  value_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
  model_version TEXT PRIMARY KEY,
  server_id TEXT NOT NULL,
  server_version_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  seed INTEGER NOT NULL,
  budget_json TEXT NOT NULL,
  coverage_json TEXT NOT NULL,
  parent_model_version TEXT NULL,
  created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
  experiment_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  fixture_id TEXT NOT NULL,
  server_version_id TEXT NOT NULL,
  action_class_id TEXT NOT NULL,
  resource_ref TEXT NOT NULL,
  args_json TEXT NOT NULL,
  before_json TEXT NOT NULL,
  before_state_id TEXT NOT NULL,
  result_json TEXT NOT NULL,
  outcome_kind TEXT NOT NULL,
  after_json TEXT NULL,
  after_state_id TEXT NULL,
  reset_confirmed INTEGER NOT NULL,
  phase TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  UNIQUE (run_id, seq)
);
CREATE INDEX IF NOT EXISTS ix_experiments_run_seq ON experiments (run_id, seq);
CREATE TABLE IF NOT EXISTS observations (
  observation_id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL,
  phase TEXT NOT NULL,
  tool TEXT NOT NULL,
  raw_json TEXT NULL,
  state_id TEXT NULL
);
CREATE TABLE IF NOT EXISTS transitions (
  transition_id TEXT PRIMARY KEY,
  model_version TEXT NOT NULL,
  source_state_id TEXT NOT NULL,
  action_class_id TEXT NOT NULL,
  outcome_kind TEXT NOT NULL,
  target_state_id TEXT NULL,
  status TEXT NOT NULL,
  guard_json TEXT NOT NULL,
  provenance TEXT NOT NULL,
  evidence_count INTEGER NOT NULL,
  evidence_json TEXT NOT NULL,
  contradiction_count INTEGER NOT NULL,
  counterexamples_json TEXT NOT NULL,
  predicted_effects_json TEXT NOT NULL,
  retest_required INTEGER NOT NULL,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_transitions_lookup
  ON transitions (model_version, source_state_id, action_class_id);
CREATE TABLE IF NOT EXISTS transition_evidence (
  transition_id TEXT NOT NULL,
  experiment_id TEXT NOT NULL,
  role TEXT NOT NULL,
  PRIMARY KEY (transition_id, experiment_id, role)
);
CREATE TABLE IF NOT EXISTS counterexamples (
  counterexample_id TEXT PRIMARY KEY,
  model_version TEXT NOT NULL,
  transition_id TEXT NOT NULL,
  experiment_id TEXT NOT NULL,
  predicted_json TEXT NOT NULL,
  observed_json TEXT NOT NULL,
  created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS journal_events (
  run_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY (run_id, seq)
);
""",
    ),
    (
        2,
        """
CREATE TABLE IF NOT EXISTS executions (
  execution_id TEXT PRIMARY KEY,
  idempotency_key TEXT NOT NULL UNIQUE,
  offer_id TEXT NOT NULL,
  server_id TEXT NOT NULL,
  tool TEXT NOT NULL,
  args_json TEXT NOT NULL,
  decision TEXT NOT NULL,
  reason TEXT NOT NULL,
  before_json TEXT NULL,
  after_json TEXT NULL,
  outcome_kind TEXT NULL,
  predicted_json TEXT NOT NULL,
  observed_json TEXT NULL,
  prediction_matched INTEGER NULL,
  model_version TEXT NOT NULL,
  transition_ids_json TEXT NOT NULL,
  counterexample_id TEXT NULL,
  created_at INTEGER NOT NULL
);
""",
    ),
]


def apply_migrations(conn: sqlite3.Connection) -> list[int]:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    newly: list[int] = []
    for version, ddl in MIGRATIONS:
        if version in applied:
            continue
        with conn:
            conn.executescript(ddl)
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, "logical"),
            )
        newly.append(version)
    return newly
