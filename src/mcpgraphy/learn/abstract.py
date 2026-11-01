"""State abstraction: raw read-backs -> bounded typed feature maps.

Identifiers are never features (they would make every resource its own state). Only the
observable properties that could plausibly gate behavior are kept. Feature extractors are a
registry keyed by resource class so new servers can plug in.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mcpgraphy.models import AbstractState, Features, Snapshot

MISSING: Features = {"exists": False}


def _release_features(raw: dict[str, Any]) -> Features:
    return {"status": str(raw.get("status")), "tests_passed": bool(raw.get("tests_passed"))}


def _ticket_features(raw: dict[str, Any]) -> Features:
    return {"state": str(raw.get("state")), "has_assignee": raw.get("assignee") is not None}


EXTRACTORS: dict[str, Callable[[dict[str, Any]], Features]] = {
    "release": _release_features,
    "ticket": _ticket_features,
}


def features_of(resource_class: str, snapshot: Snapshot) -> Features:
    if snapshot.raw is None or not snapshot.outcome.kind.is_ok:
        return dict(MISSING)
    return EXTRACTORS[resource_class](snapshot.raw)


def features_of_raw(resource_class: str, raw: dict[str, Any]) -> Features:
    return EXTRACTORS[resource_class](raw)


def state_of(server_id: str, resource_class: str, snapshot: Snapshot) -> AbstractState:
    return AbstractState.build(server_id, resource_class, features_of(resource_class, snapshot))


def diff_effects(source: Features, target: Features | None) -> tuple[str, ...]:
    """Human-readable, sorted 'feature becomes value' list; ('no state change',) for a self-loop."""
    if target is None:
        return ("successor state unobserved",)
    changes = []
    for key in sorted(set(source) | set(target)):
        if source.get(key) != target.get(key):
            changes.append(f"{key} becomes {_fmt(target.get(key))}")
    return tuple(changes) if changes else ("no state change",)


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return "null" if v is None else str(v)
