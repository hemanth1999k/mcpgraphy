"""`ticket_sim`: a parameterized, partially observable task service.

Visible (via `get_ticket`): state, assignee, project_id, title.
Hidden: `write_denied` (the acting principal lacks write permission on this ticket).

Benchmark conditions covered: similar project names in different namespaces (`p-alpha` vs
`p-alpha-ops`), users valid only within their own project, transitions that require assignment,
hidden permission flags, deleted/stale ids.

Version "v2" differs from "v1" in exactly two ways (drift tests):
  * `close_ticket` also succeeds from IN_PROGRESS (behavior change, no schema change);
  * `list_users` gains an optional `include_inactive` argument (schema change, no behavior change).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

from mcpgraphy.models import ResourceRef
from mcpgraphy.sandbox.base import CallCounter, FixtureManifest, counted, fixture_id_for
from mcpgraphy.sandbox.release_sim import RuleError

SERVER_ID = "ticket_sim"
ACTING_TENANT = "p-alpha"
STATES = ("OPEN", "IN_PROGRESS", "DONE", "CLOSED")
PROJECTS: dict[str, dict[str, Any]] = {
    "p-alpha": {"name": "Alpha", "users": ["u-alpha-1", "u-alpha-2"]},
    "p-alpha-ops": {"name": "Alpha Ops", "users": ["u-ops-1"]},
    "p-beta": {"name": "Beta", "users": ["u-beta-1", "u-beta-2"]},
}
LEGAL_MOVES = {("OPEN", "IN_PROGRESS"), ("IN_PROGRESS", "DONE"), ("DONE", "OPEN")}


@dataclass
class Ticket:
    id: str
    project_id: str
    title: str
    state: str = "OPEN"
    assignee: str | None = None
    # hidden
    write_denied: bool = False

    def visible(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "assignee": self.assignee,
            "project_id": self.project_id,
            "title": self.title,
        }


@dataclass
class TicketWorld:
    tickets: dict[str, Ticket] = field(default_factory=dict)
    deleted: set[str] = field(default_factory=set)
    next_id: int = 1


class TicketView(BaseModel):
    id: str
    state: str
    assignee: str | None
    project_id: str
    title: str


class CreateResult(BaseModel):
    id: str


class OkResult(BaseModel):
    ok: bool
    id: str


class UsersResult(BaseModel):
    project_id: str
    users: list[str]


class TicketRules:
    def __init__(self, version: Literal["v1", "v2"] = "v1") -> None:
        self.version = version

    def _get(self, world: TicketWorld, ticket_id: str) -> Ticket:
        t = world.tickets.get(ticket_id)
        if t is None:
            raise RuleError("NOT_FOUND", f"ticket {ticket_id!r} not found")
        return t

    @staticmethod
    def _writable(t: Ticket) -> None:
        if t.write_denied:
            raise RuleError("PERMISSION_DENIED", "you cannot modify this ticket")

    def create_ticket(self, world: TicketWorld, project_id: str, title: str) -> dict[str, Any]:
        if project_id not in PROJECTS:
            raise RuleError("NOT_FOUND", f"project {project_id!r} not found")
        if not isinstance(title, str) or not title.strip():
            raise RuleError("INVALID_ARGUMENT", "title must be a non-empty string")
        tid = f"tk-new-{world.next_id}"
        world.next_id += 1
        world.tickets[tid] = Ticket(id=tid, project_id=project_id, title=title)
        return {"id": tid}

    def get_ticket(self, world: TicketWorld, ticket_id: str) -> dict[str, Any]:
        return self._get(world, ticket_id).visible()

    def list_users(
        self, world: TicketWorld, project_id: str, include_inactive: bool = False
    ) -> dict[str, Any]:
        if project_id not in PROJECTS:
            raise RuleError("NOT_FOUND", f"project {project_id!r} not found")
        return {"project_id": project_id, "users": list(PROJECTS[project_id]["users"])}

    def assign_ticket(self, world: TicketWorld, ticket_id: str, user_id: str) -> dict[str, Any]:
        t = self._get(world, ticket_id)
        self._writable(t)
        if user_id not in PROJECTS[t.project_id]["users"]:
            raise RuleError("INVALID_ARGUMENT", "user is not a member of the ticket's project")
        if t.state == "CLOSED":
            raise RuleError("STATE_CONFLICT", "closed tickets cannot be assigned")
        t.assignee = user_id
        return {"ok": True, "id": t.id}

    def move_ticket(self, world: TicketWorld, ticket_id: str, state: str) -> dict[str, Any]:
        t = self._get(world, ticket_id)
        if state not in STATES:
            raise RuleError("INVALID_ARGUMENT", f"unknown state {state!r}")
        self._writable(t)
        if (t.state, state) not in LEGAL_MOVES:
            raise RuleError("STATE_CONFLICT", f"cannot move from {t.state} to {state}")
        if state == "IN_PROGRESS" and t.assignee is None:
            raise RuleError("STATE_CONFLICT", "ticket must be assigned before work starts")
        t.state = state
        return {"ok": True, "id": t.id}

    def close_ticket(self, world: TicketWorld, ticket_id: str) -> dict[str, Any]:
        t = self._get(world, ticket_id)
        self._writable(t)
        allowed = {"DONE"} | ({"IN_PROGRESS"} if self.version == "v2" else set())
        if t.state not in allowed:
            raise RuleError("STATE_CONFLICT", "ticket cannot be closed in its current state")
        t.state = "CLOSED"
        return {"ok": True, "id": t.id}


@dataclass(frozen=True)
class _FixtureSpec:
    fixture_id: str
    seed: int
    index: int
    state: str
    has_assignee: bool
    write_denied: bool
    title: str


def _grid() -> list[tuple[str, bool, bool]]:
    """(state, has_assignee, write_denied). Permission denial is hidden and rare: it only exists
    on unassigned OPEN and assigned DONE tickets. 10 cells."""
    cells = [(s, a, False) for s in STATES for a in (False, True)]
    cells += [("OPEN", False, True), ("DONE", True, True)]
    return cells


class TicketSandbox:
    server_id = SERVER_ID

    def __init__(self, version: Literal["v1", "v2"] = "v1") -> None:
        self.version = version
        self.rules = TicketRules(version)
        self.world = TicketWorld()
        self.counter = CallCounter()
        self._specs: dict[str, _FixtureSpec] = {}
        self.mcp = _build_mcp(self)

    @property
    def reset_supported(self) -> bool:
        return True

    def fixtures(self, seed: int, n: int) -> list[str]:
        rng = random.Random(f"{SERVER_ID}|{self.version}|{seed}")
        grid = _grid()
        order = list(range(len(grid)))
        rng.shuffle(order)
        ids = []
        for i in range(n):
            state, has_assignee, write_denied = grid[order[i % len(grid)]]
            fid = fixture_id_for(SERVER_ID, self.version, seed, i)
            title = f"{rng.choice(['Fix', 'Ship', 'Review', 'Refactor'])} item {rng.randint(1, 99)}"
            self._specs[fid] = _FixtureSpec(fid, seed, i, state, has_assignee, write_denied, title)
            ids.append(fid)
        return ids

    def reset_fixture(self, fixture_id: str) -> FixtureManifest:
        spec = self._specs.get(fixture_id)
        if spec is None:
            raise KeyError(f"unknown fixture {fixture_id}")
        w = TicketWorld()
        focal_id = f"tk-{spec.seed}-{spec.index}"
        w.tickets[focal_id] = Ticket(
            id=focal_id,
            project_id=ACTING_TENANT,
            title=spec.title,
            state=spec.state,
            assignee="u-alpha-1" if spec.has_assignee else None,
            write_denied=spec.write_denied,
        )
        extra = f"tk-{spec.seed}-{spec.index}-x"
        w.tickets[extra] = Ticket(id=extra, project_id=ACTING_TENANT, title="Extra open ticket")
        other = f"tk-{spec.seed}-{spec.index}-ops"
        w.tickets[other] = Ticket(id=other, project_id="p-alpha-ops", title="Ops ticket")
        deleted = f"tk-{spec.seed}-{spec.index}-del"
        w.deleted.add(deleted)
        self.world = w
        self.counter.resets += 1
        focal = w.tickets[focal_id]
        return FixtureManifest(
            fixture_id=fixture_id,
            server_id=SERVER_ID,
            version=self.version,
            focal_ref=ResourceRef.build(SERVER_ID, ACTING_TENANT, "ticket", focal_id),
            expected_visible=focal.visible(),
            id_pools={
                "EXISTING": [focal_id],
                "EXISTING_OTHER": [extra],
                "OTHER_TENANT": [other],
                "STALE_DELETED": [deleted],
                "NONEXISTENT": [f"tk-{spec.seed}-{spec.index}-nope"],
                "MALFORMED": ["", "not an id"],
                "USER_IN_PROJECT": ["u-alpha-2"],
                "USER_OTHER_PROJECT": ["u-ops-1"],
                "USER_NONEXISTENT": ["u-nope"],
                "EXISTING_PROJECT": ["p-alpha"],
                "SIMILAR_NAME_OTHER_PROJECT": ["p-alpha-ops"],
                "NONEXISTENT_PROJECT": ["p-gamma"],
            },
        )

    def inject_concurrent_update(self, fixture_id: str, op: str) -> None:
        spec = self._specs[fixture_id]
        t = self.world.tickets[f"tk-{spec.seed}-{spec.index}"]
        if op == "close":
            t.state = "CLOSED"
        elif op == "unassign":
            t.assignee = None
        elif op == "deny":
            t.write_denied = True
        else:
            raise ValueError(f"unknown concurrent op {op}")


def _build_mcp(sb: TicketSandbox) -> MCPServer:
    mcp = MCPServer(SERVER_ID, version=sb.version)
    rules = sb.rules

    def guard(fn: Any) -> Any:
        def inner(*a: Any, **kw: Any) -> Any:
            try:
                return fn(sb.world, *a, **kw)
            except RuleError as e:
                raise ToolError(f"{e.code}: {e.message}") from None

        return inner

    @mcp.tool()
    def create_ticket(project_id: str, title: str) -> CreateResult:
        """Create a ticket in a project. Returns its id."""
        return CreateResult(
            **counted(sb.counter, "create_ticket", guard(rules.create_ticket))(project_id, title)
        )

    @mcp.tool()
    def get_ticket(ticket_id: str) -> TicketView:
        """Fetch a ticket: state, assignee, project and title."""
        return TicketView(**counted(sb.counter, "get_ticket", guard(rules.get_ticket))(ticket_id))

    if sb.version == "v2":

        @mcp.tool()
        def list_users(project_id: str, include_inactive: bool = False) -> UsersResult:
            """List the users of a project."""
            return UsersResult(
                **counted(sb.counter, "list_users", guard(rules.list_users))(
                    project_id, include_inactive
                )
            )
    else:

        @mcp.tool()
        def list_users(project_id: str) -> UsersResult:  # type: ignore[misc]
            """List the users of a project."""
            return UsersResult(
                **counted(sb.counter, "list_users", guard(rules.list_users))(project_id)
            )

    @mcp.tool()
    def assign_ticket(ticket_id: str, user_id: str) -> OkResult:
        """Assign a ticket to a user."""
        return OkResult(
            **counted(sb.counter, "assign_ticket", guard(rules.assign_ticket))(ticket_id, user_id)
        )

    @mcp.tool()
    def move_ticket(ticket_id: str, state: str) -> OkResult:
        """Move a ticket to another state."""
        return OkResult(
            **counted(sb.counter, "move_ticket", guard(rules.move_ticket))(ticket_id, state)
        )

    @mcp.tool()
    def close_ticket(ticket_id: str) -> OkResult:
        """Close a ticket."""
        return OkResult(**counted(sb.counter, "close_ticket", guard(rules.close_ticket))(ticket_id))

    return mcp
