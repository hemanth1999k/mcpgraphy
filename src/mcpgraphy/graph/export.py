"""JSON is the authoritative export; Mermaid is a presentation derived from it."""

from __future__ import annotations

import json
from typing import Any

from mcpgraphy.graph.store import GraphStore
from mcpgraphy.models import (
    AbstractState,
    ActionClass,
    Counterexample,
    Experiment,
    ModelRecord,
    Transition,
)

FORMAT_VERSION = 1


def export_json(
    store: GraphStore, model_version: str, strip_timestamps: bool = True
) -> dict[str, Any]:
    m = store.model(model_version)
    if m is None:
        raise LookupError(model_version)
    sv = store.server_version(m.server_version_id)
    states = store.states_for_model(model_version)
    acs = store.action_classes(m.server_id)
    transitions = store.transitions(model_version)
    used_acs = {t.action_class_id for t in transitions} | {
        e.action_class_id for e in store.experiments(m.run_id)
    }
    data: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model": {
            "model_version": m.model_version,
            "server_id": m.server_id,
            "server_version": sv.version if sv else None,
            "server_version_id": m.server_version_id,
            "fingerprint": sv.fingerprint if sv else None,
            "seed": m.seed,
            "budget": m.budget,
            "coverage": m.coverage,
            "parent_model_version": m.parent_model_version,
        },
        "states": [s.to_dict() for _, s in sorted(states.items())],
        "action_classes": [
            {**a.to_dict(), "label": a.label} for k, a in sorted(acs.items()) if k in used_acs
        ],
        "transitions": [_strip(t.to_dict(), strip_timestamps) for t in transitions],
        "counterexamples": [
            _strip(c.to_dict(), strip_timestamps) for c in store.counterexamples(model_version)
        ],
        "experiments": [
            {
                "experiment_id": e.experiment_id,
                "seq": e.seq,
                "fixture_id": e.fixture_id,
                "action_class_id": e.action_class_id,
                "args": e.args,
                "before_state_id": e.before_state_id,
                "outcome_kind": str(e.result.kind),
                "outcome_code": e.result.code,
                "after_state_id": e.after_state_id,
                "reset_confirmed": e.reset_confirmed,
                "phase": e.phase,
            }
            for e in store.experiments(m.run_id)
        ],
        "events": [
            {k: v for k, v in ev.items() if k in ("seq", "kind", "reason", "phase", "fixture_id")}
            for ev in store.events(m.run_id)
            if ev["kind"] in ("skipped", "abort", "budget_exhausted")
        ],
    }
    if strip_timestamps:
        data["model"].pop("created_at", None)
    return data


def _strip(d: dict[str, Any], strip: bool) -> dict[str, Any]:
    if strip:
        d.pop("created_at", None)
    return d


def dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def import_json(store: GraphStore, data: dict[str, Any]) -> str:
    """Load an export into a store (round trip). Experiments are restored only as summaries."""
    m = data["model"]
    record = ModelRecord(
        model_version=m["model_version"],
        server_id=m["server_id"],
        server_version_id=m["server_version_id"],
        run_id=f"import_{m['model_version']}",
        seed=m["seed"],
        budget=m["budget"],
        coverage=m["coverage"],
        parent_model_version=m.get("parent_model_version"),
        created_at=0,
    )
    states = [AbstractState(**s) for s in data["states"]]
    acs = [
        ActionClass(**{k: v for k, v in a.items() if k != "label"}) for a in data["action_classes"]
    ]
    transitions = [Transition(**t) for t in data["transitions"]]
    cxs = [Counterexample(**c) for c in data["counterexamples"]]
    with store.conn:
        store.save_states(states)
        store.save_action_classes(acs)
        store.save_model(record)
        store.replace_transitions(record.model_version, transitions, cxs)
    return record.model_version


def export_mermaid(data: dict[str, Any]) -> str:
    """Flowchart of OK transitions plus dashed hypothesized edges. Derived from the JSON only."""
    states = {s["state_id"]: s for s in data["states"]}
    acs = {a["action_class_id"]: a for a in data["action_classes"]}
    lines = ["flowchart LR"]
    for sid, s in sorted(states.items()):
        label = ", ".join(f"{k}={v}" for k, v in s["features"].items())
        lines.append(f'  {_node(sid)}["{label}"]')
    for t in data["transitions"]:
        if t["outcome_kind"] != "OK" or not t["target_state_id"]:
            continue
        ac = acs.get(t["action_class_id"], {})
        label = f"{ac.get('tool', t['action_class_id'])} [{t['status']}]"
        arrow = "-.->" if t["status"] in ("CONDITIONAL", "UNKNOWN") else "-->"
        if t["status"] == "CONTRADICTED":
            label += " !"
        lines.append(
            f'  {_node(t["source_state_id"])} {arrow}|"{label}"| {_node(t["target_state_id"])}'
        )
    return "\n".join(lines) + "\n"


def _node(state_id: str) -> str:
    return state_id.replace("-", "_")


__all__ = ["Experiment", "dumps", "export_json", "export_mermaid", "import_json"]
