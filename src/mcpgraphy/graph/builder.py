"""Materialize a model from the journal.

`materialize(journal_path, store)` is idempotent and runs in a single SQLite transaction, so an
interruption leaves either the previous state or the complete new one. The journal is the only
input; re-running it reproduces byte-identical exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcpgraphy.graph.store import GraphStore
from mcpgraphy.learn.counterexamples import inherit_for_retest
from mcpgraphy.learn.guards import induce
from mcpgraphy.models import (
    AbstractState,
    ActionClass,
    BehaviorStatus,
    Experiment,
    ModelRecord,
    ServerVersion,
    ToolSpec,
)
from mcpgraphy.observe.journal import Journal


class MaterializeError(RuntimeError):
    pass


def materialize(journal_path: str | Path, store: GraphStore, run_id: str | None = None) -> str:
    """Rebuild the model for one run from the journal; return its model_version."""
    events, skipped = Journal.read(journal_path)
    runs = _split_runs(events)
    if not runs:
        raise MaterializeError("journal has no run_start event")
    if run_id is None:
        run_id = list(runs)[-1]
    if run_id not in runs:
        raise MaterializeError(f"run {run_id} not in journal")
    run_events = runs[run_id]
    start = run_events[0]
    end = next((e for e in run_events if e["kind"] == "run_end"), None)

    server_id, version = start["server_id"], start["version"]
    model_version = start["model_version"]
    sv = ServerVersion.build(server_id, version, [ToolSpec(**t) for t in start["tools"]])
    if sv.server_version_id != start["server_version_id"]:
        raise MaterializeError("server fingerprint in journal does not match its tool list")

    experiments: list[Experiment] = []
    states: dict[str, AbstractState] = {}
    action_classes: dict[str, ActionClass] = {}
    for ev in run_events:
        if ev["kind"] != "experiment":
            continue
        payload = {k: v for k, v in ev.items() if k not in _EXTRA_KEYS}
        exp = Experiment(**payload)
        experiments.append(exp)
        states[exp.before_state_id] = AbstractState(
            state_id=exp.before_state_id,
            server_id=server_id,
            resource_class=_resource_class(exp),
            features=ev["before_features"],
        )
        if exp.after_state_id:
            states[exp.after_state_id] = AbstractState(
                state_id=exp.after_state_id,
                server_id=server_id,
                resource_class=_resource_class(exp),
                features=ev["after_features"],
            )
        ac = ActionClass(**ev["action_class"])
        action_classes[ac.action_class_id] = ac
    # states known from fixtures even if never probed (e.g. aborted runs)
    for sid in (end or {}).get("known_states", []):
        states.setdefault(sid, None)  # type: ignore[arg-type]
    states = {k: v for k, v in states.items() if v is not None}

    ac_ids = sorted(action_classes)
    induced = induce(model_version, experiments, states, ac_ids)
    transitions = list(induced.transitions)
    counterexamples = list(induced.counterexamples)

    parent = start.get("base_model_version")
    if parent:
        inherited = inherit_for_retest(
            store, parent, model_version, sv, states, action_classes, transitions
        )
        transitions.extend(inherited.transitions)
        counterexamples.extend(inherited.counterexamples)
        transitions.sort(key=lambda t: t.transition_id)
        counterexamples.sort(key=lambda c: c.counterexample_id)

    status_counts: dict[str, int] = {}
    for t in transitions:
        status_counts[str(t.status)] = status_counts.get(str(t.status), 0) + 1
    coverage: dict[str, Any] = {
        "experiments": len(experiments),
        "states": len(states),
        "action_classes": len(action_classes),
        "transitions": len(transitions),
        "by_status": dict(sorted(status_counts.items())),
        "counterexamples": len(counterexamples),
        "skipped_journal_lines": skipped,
        "aborted": (end or {}).get("aborted"),
        "phases": (end or {}).get("phases", {}),
        "reset_supported": start.get("reset_supported", True),
        "unknown_pairs": _unknown_pairs(states, action_classes, transitions),
    }
    record = ModelRecord(
        model_version=model_version,
        server_id=server_id,
        server_version_id=sv.server_version_id,
        run_id=run_id,
        seed=start["seed"],
        budget=(end or {}).get("budget", {"total": start["budget"]}),
        coverage=coverage,
        parent_model_version=parent,
        created_at=len(run_events),
    )
    with store.conn:
        store.save_server_version(sv)
        store.save_states(states.values())
        store.save_action_classes(action_classes.values())
        store.save_experiments(experiments)
        store.save_events(run_id, run_events)
        store.save_model(record)
        store.replace_transitions(model_version, transitions, counterexamples)
    return model_version


_EXTRA_KEYS = {"kind", "before_features", "after_features", "action_class"}


def _split_runs(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    current: str | None = None
    for ev in events:
        if ev["kind"] == "run_start":
            current = ev["run_id"]
            runs[current] = []
        if current is not None and ev.get("run_id", current) == current:
            runs[current].append(ev)
    return runs


def _resource_class(exp: Experiment) -> str:
    return exp.resource_ref.split(":", 1)[0]


def _unknown_pairs(
    states: dict[str, AbstractState], acs: dict[str, ActionClass], transitions: list
) -> int:
    covered = {
        (t.source_state_id, t.action_class_id)
        for t in transitions
        if t.status is not BehaviorStatus.UNKNOWN
    }
    return sum(1 for s in states for a in acs if (s, a) not in covered)
