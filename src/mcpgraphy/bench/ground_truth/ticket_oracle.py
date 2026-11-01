"""Private ground truth for ticket_sim. See release_oracle.py."""

from __future__ import annotations

import copy
from typing import Any

from mcpgraphy.models import OutcomeKind
from mcpgraphy.sandbox.release_sim import RuleError
from mcpgraphy.sandbox.ticket_sim import LEGAL_MOVES, PROJECTS, TicketSandbox

HIDDEN_FEATURES = ("write_denied",)
VISIBLE_FEATURES = ("state", "has_assignee")


def full_features(sb: TicketSandbox, ticket_id: str) -> dict[str, Any] | None:
    t = sb.world.tickets.get(ticket_id)
    if t is None:
        return None
    return {
        "state": t.state,
        "has_assignee": t.assignee is not None,
        "write_denied": t.write_denied,
        "project_id": t.project_id,
    }


def true_outcome(
    sb: TicketSandbox, tool: str, args: dict[str, Any]
) -> tuple[OutcomeKind, dict[str, Any] | None]:
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
    tid = args.get("ticket_id")
    after = world.tickets[tid].visible() if tid in world.tickets else None
    return OutcomeKind.OK, after


def oracle_outcome(
    version: str, tool: str, arg_classes: dict[str, str], full: dict[str, Any] | None
) -> OutcomeKind:
    if tool == "create_ticket":
        if arg_classes["project_id"] == "NONEXISTENT_PROJECT":
            return OutcomeKind.ERR_NOT_FOUND
        return OutcomeKind.ERR_VALIDATION if arg_classes["title"] == "EMPTY" else OutcomeKind.OK
    if tool == "list_users":
        return (
            OutcomeKind.ERR_NOT_FOUND
            if arg_classes["project_id"] == "NONEXISTENT_PROJECT"
            else OutcomeKind.OK
        )
    id_class = arg_classes.get("ticket_id")
    if id_class in ("NONEXISTENT", "STALE_DELETED", "MALFORMED") or full is None:
        return OutcomeKind.ERR_NOT_FOUND
    # OTHER_TENANT tickets exist (other project) and are readable
    if tool == "get_ticket":
        return OutcomeKind.OK
    if tool == "move_ticket" and arg_classes["state"] == "TARGET_INVALID":
        return OutcomeKind.ERR_VALIDATION
    if full["write_denied"]:
        return OutcomeKind.ERR_PERMISSION
    if tool == "assign_ticket":
        user_class = arg_classes["user_id"]
        members = PROJECTS[full["project_id"]]["users"]
        user = {
            "USER_IN_PROJECT": "u-alpha-2",
            "USER_OTHER_PROJECT": "u-ops-1",
            "USER_NONEXISTENT": "u-nope",
        }[user_class]
        if user not in members:
            return OutcomeKind.ERR_VALIDATION
        return OutcomeKind.ERR_STATE if full["state"] == "CLOSED" else OutcomeKind.OK
    if tool == "move_ticket":
        target = arg_classes["state"].removeprefix("TARGET_")
        if (full["state"], target) not in LEGAL_MOVES:
            return OutcomeKind.ERR_STATE
        if target == "IN_PROGRESS" and not full["has_assignee"]:
            return OutcomeKind.ERR_STATE
        return OutcomeKind.OK
    if tool == "close_ticket":
        allowed = {"DONE"} | ({"IN_PROGRESS"} if version == "v2" else set())
        return OutcomeKind.OK if full["state"] in allowed else OutcomeKind.ERR_STATE
    raise KeyError(tool)


def visible_cells() -> list[dict[str, Any]]:
    return [
        {"state": s, "has_assignee": a}
        for s in ("OPEN", "IN_PROGRESS", "DONE", "CLOSED")
        for a in (False, True)
    ]


def hidden_cells() -> list[dict[str, Any]]:
    return [{"write_denied": w} for w in (False, True)]
