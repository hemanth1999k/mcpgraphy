"""Private benchmark ground truth. Never imported by `mcpgraphy` (enforced by tests)."""

from __future__ import annotations

import copy
from collections import deque
from types import ModuleType
from typing import Any

from mcpgraphy.bench.ground_truth import release_oracle, ticket_oracle
from mcpgraphy.models import OutcomeKind
from mcpgraphy.sandbox.base import SandboxControl


def oracle_for(server_id: str) -> ModuleType:
    return {"release_sim": release_oracle, "ticket_sim": ticket_oracle}[server_id]


def true_outcome(sandbox: SandboxControl, tool: str, args: dict[str, Any]) -> OutcomeKind:
    return oracle_for(sandbox.server_id).true_outcome(sandbox, tool, args)[0]  # type: ignore[arg-type]


def focal_visible(sandbox: Any, local_id: str) -> dict[str, Any] | None:
    """Visible features of a resource straight from the world (benchmark-side convenience)."""
    if sandbox.server_id == "release_sim":
        r = sandbox.world.releases.get(local_id)
        return (
            None
            if r is None or r.tenant != "t1"
            else {"status": r.status, "tests_passed": r.tests_passed}
        )
    t = sandbox.world.tickets.get(local_id)
    return None if t is None else {"state": t.state, "has_assignee": t.assignee is not None}


def reachable(
    sandbox: Any,
    local_id: str,
    goal: dict[str, Any],
    actions: list[tuple[str, dict[str, Any]]],
    max_depth: int,
) -> bool:
    """Breadth-first search over the TRUE world (hidden state included): is there a sequence of
    at most `max_depth` actions after which the resource's visible features satisfy `goal`?"""
    start = copy.deepcopy(sandbox.world)
    rules = sandbox.rules

    def satisfied(world) -> bool:
        sandbox_like = type("W", (), {"world": world, "server_id": sandbox.server_id})()
        vis = focal_visible(sandbox_like, local_id)
        return vis is not None and all(vis.get(k) == v for k, v in goal.items())

    if satisfied(start):
        return True
    seen = {_key(sandbox.server_id, start, local_id)}
    queue = deque([(start, 0)])
    while queue:
        world, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for tool, args in actions:
            nxt = copy.deepcopy(world)
            try:
                getattr(rules, tool)(nxt, **args)
            except Exception:
                continue
            if satisfied(nxt):
                return True
            k = _key(sandbox.server_id, nxt, local_id)
            if k not in seen:
                seen.add(k)
                queue.append((nxt, depth + 1))
    return False


def _key(server_id: str, world: Any, local_id: str) -> tuple:
    if server_id == "release_sim":
        r = world.releases.get(local_id)
        return (r.status, r.tests_passed, r.version, r.locked, r.tested_version) if r else ()
    t = world.tickets.get(local_id)
    return (t.state, t.assignee, t.write_denied) if t else ()
