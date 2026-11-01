"""Private ground truth for release_sim.

Two independent sources of truth, checked against each other by a property test:
  * `true_outcome` simulates the real rules on a deep copy of the world (correct by construction);
  * `oracle_outcome` is the handwritten guard table over visible + hidden features (what a human
    author believes the rules are). B2 uses the latter, with full hidden state.

`mcpgraphy` must never import this package. tests/unit/test_oracle_isolation.py enforces it.
"""

from __future__ import annotations

import copy
from typing import Any

from mcpgraphy.models import OutcomeKind
from mcpgraphy.sandbox.release_sim import ReleaseSandbox, RuleError

HIDDEN_FEATURES = ("locked", "mismatch")
VISIBLE_FEATURES = ("status", "tests_passed")


def full_features(sb: ReleaseSandbox, release_id: str) -> dict[str, Any] | None:
    r = sb.world.releases.get(release_id)
    if r is None or r.tenant != "t1":
        return None
    return {
        "status": r.status,
        "tests_passed": r.tests_passed,
        "locked": r.locked,
        "mismatch": r.tested_version != r.version,
    }


def true_outcome(
    sb: ReleaseSandbox, tool: str, args: dict[str, Any]
) -> tuple[OutcomeKind, dict[str, Any] | None]:
    """Apply the real rules to a copy of the world; return (outcome, visible-after of the focal
    argument resource if any). Never mutates the sandbox."""
    world = copy.deepcopy(sb.world)
    fn = getattr(sb.rules, tool)
    try:
        fn(world, **args)
    except RuleError as e:
        kind = {
            "NOT_FOUND": OutcomeKind.ERR_NOT_FOUND,
            "STATE_CONFLICT": OutcomeKind.ERR_STATE,
            "PERMISSION_DENIED": OutcomeKind.ERR_PERMISSION,
            "INVALID_ARGUMENT": OutcomeKind.ERR_VALIDATION,
        }[e.code]
        return kind, None
    rid = args.get("release_id")
    after = world.releases[rid].visible() if rid in world.releases else None
    return OutcomeKind.OK, after


def oracle_outcome(
    version: str, tool: str, arg_classes: dict[str, str], full: dict[str, Any] | None
) -> OutcomeKind:
    """Handwritten guard table. `full` is the focal resource's visible+hidden features, or None
    when the id argument does not resolve to an acting-tenant resource."""
    id_class = arg_classes.get("release_id")
    if tool == "create_release":
        return OutcomeKind.ERR_VALIDATION if arg_classes["name"] == "EMPTY" else OutcomeKind.OK
    if id_class != "EXISTING" or full is None:
        return OutcomeKind.ERR_NOT_FOUND
    if tool == "get_release":
        return OutcomeKind.OK
    if tool == "edit_release":
        if arg_classes["title"] == "EMPTY":
            return OutcomeKind.ERR_VALIDATION
        if full["status"] != "DRAFT" or full["locked"]:
            return OutcomeKind.ERR_STATE
        return OutcomeKind.OK
    if tool == "run_tests":
        return OutcomeKind.ERR_STATE if full["status"] == "ARCHIVED" else OutcomeKind.OK
    if tool == "publish_release":
        ok = (
            full["status"] == "DRAFT"
            and full["tests_passed"]
            and not full["locked"]
            and not full["mismatch"]
        )
        return OutcomeKind.OK if ok else OutcomeKind.ERR_STATE
    if tool == "archive_release":
        allowed = {"PUBLISHED"} | ({"DRAFT"} if version == "v2" else set())
        ok = full["status"] in allowed and not full["locked"]
        return OutcomeKind.OK if ok else OutcomeKind.ERR_STATE
    raise KeyError(tool)


def visible_cells() -> list[dict[str, Any]]:
    return [
        {"status": s, "tests_passed": t}
        for s in ("DRAFT", "PUBLISHED", "ARCHIVED")
        for t in (False, True)
    ]


def hidden_cells() -> list[dict[str, Any]]:
    return [{"locked": lk, "mismatch": mm} for lk in (False, True) for mm in (False, True)]
