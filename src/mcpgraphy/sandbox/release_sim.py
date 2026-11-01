"""`release_sim`: a release lifecycle service with visible and hidden state.

Visible (via `get_release`): status, tests_passed, version, title.
Hidden (never exposed by any tool): `locked`, `tested_version` (a version mismatch when it differs
from `version`). Hidden-state failures deliberately reuse STATE_CONFLICT so the hidden cause does
not leak through the error code.

Version "v2" differs from "v1" in exactly two ways (drift tests, spec §10.1 and §15 cats 6-7):
  * `archive_release` from DRAFT succeeds (behavior change, no schema change);
  * `run_tests` gains an optional `suite` argument (schema change, no behavior change).
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

SERVER_ID = "release_sim"
ACTING_TENANT = "t1"
STATUSES = ("DRAFT", "PUBLISHED", "ARCHIVED")


class RuleError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass
class Release:
    id: str
    tenant: str
    title: str
    status: str = "DRAFT"
    tests_passed: bool = False
    version: int = 1
    # hidden
    locked: bool = False
    tested_version: int = 0

    def visible(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "tests_passed": self.tests_passed,
            "version": self.version,
            "title": self.title,
        }


@dataclass
class ReleaseWorld:
    releases: dict[str, Release] = field(default_factory=dict)
    deleted: set[str] = field(default_factory=set)
    next_id: int = 1


class ReleaseView(BaseModel):
    id: str
    status: str
    tests_passed: bool
    version: int
    title: str


class CreateResult(BaseModel):
    id: str


class OkResult(BaseModel):
    ok: bool
    id: str


class ReleaseRules:
    """Pure transition rules. Raise RuleError; never touch anything but `world`."""

    def __init__(self, version: Literal["v1", "v2"] = "v1") -> None:
        self.version = version

    def _get(self, world: ReleaseWorld, release_id: str) -> Release:
        r = world.releases.get(release_id)
        if r is None or r.tenant != ACTING_TENANT:
            raise RuleError("NOT_FOUND", f"release {release_id!r} not found")
        return r

    def create_release(self, world: ReleaseWorld, name: str) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip():
            raise RuleError("INVALID_ARGUMENT", "name must be a non-empty string")
        rid = f"rel-new-{world.next_id}"
        world.next_id += 1
        world.releases[rid] = Release(id=rid, tenant=ACTING_TENANT, title=name)
        return {"id": rid}

    def get_release(self, world: ReleaseWorld, release_id: str) -> dict[str, Any]:
        return self._get(world, release_id).visible()

    def edit_release(self, world: ReleaseWorld, release_id: str, title: str) -> dict[str, Any]:
        r = self._get(world, release_id)
        if not isinstance(title, str) or not title.strip():
            raise RuleError("INVALID_ARGUMENT", "title must be a non-empty string")
        if r.status != "DRAFT" or r.locked:
            raise RuleError("STATE_CONFLICT", "release is not editable")
        r.title = title
        r.version += 1
        r.tests_passed = False
        return {"ok": True, "id": r.id}

    def run_tests(
        self, world: ReleaseWorld, release_id: str, suite: str = "default"
    ) -> dict[str, Any]:
        r = self._get(world, release_id)
        if r.status == "ARCHIVED":
            raise RuleError("STATE_CONFLICT", "cannot run tests on an archived release")
        r.tests_passed = True
        r.tested_version = r.version
        return {"ok": True, "id": r.id}

    def publish_release(self, world: ReleaseWorld, release_id: str) -> dict[str, Any]:
        r = self._get(world, release_id)
        if r.status != "DRAFT" or not r.tests_passed or r.locked or r.tested_version != r.version:
            raise RuleError("STATE_CONFLICT", "release cannot be published in its current state")
        r.status = "PUBLISHED"
        return {"ok": True, "id": r.id}

    def archive_release(self, world: ReleaseWorld, release_id: str) -> dict[str, Any]:
        r = self._get(world, release_id)
        allowed = {"PUBLISHED"} | ({"DRAFT"} if self.version == "v2" else set())
        if r.status not in allowed or r.locked:
            raise RuleError("STATE_CONFLICT", "release cannot be archived in its current state")
        r.status = "ARCHIVED"
        return {"ok": True, "id": r.id}


@dataclass(frozen=True)
class _FixtureSpec:
    fixture_id: str
    seed: int
    index: int
    status: str
    tests_passed: bool
    locked: bool
    mismatch: bool
    title: str


def _grid() -> list[tuple[str, bool, bool, bool]]:
    """(status, tests_passed, locked, mismatch). Hidden conditions are the exception, not the
    rule: a version mismatch only exists for tested drafts, a lock only for published/archived
    releases. 11 cells."""
    cells: list[tuple[str, bool, bool, bool]] = [("DRAFT", False, False, False)]
    cells += [("DRAFT", True, False, False), ("DRAFT", True, False, True)]
    for status in ("PUBLISHED", "ARCHIVED"):
        for tests_passed in (False, True):
            cells.append((status, tests_passed, False, False))
            cells.append((status, tests_passed, True, False))
    return cells


class ReleaseSandbox:
    server_id = SERVER_ID

    def __init__(self, version: Literal["v1", "v2"] = "v1") -> None:
        self.version = version
        self.rules = ReleaseRules(version)
        self.world = ReleaseWorld()
        self.counter = CallCounter()
        self._specs: dict[str, _FixtureSpec] = {}
        self.mcp = _build_mcp(self)

    @property
    def reset_supported(self) -> bool:
        return True

    # ---- control API (never MCP tools) -------------------------------------------------
    def fixtures(self, seed: int, n: int) -> list[str]:
        rng = random.Random(f"{SERVER_ID}|{self.version}|{seed}")
        grid = _grid()
        order = list(range(len(grid)))
        rng.shuffle(order)
        ids = []
        for i in range(n):
            status, tp, locked, mismatch = grid[order[i % len(grid)]]
            fid = fixture_id_for(SERVER_ID, self.version, seed, i)
            title = (
                f"Release {rng.choice(['Aurora', 'Basalt', 'Cinder', 'Dune'])} {rng.randint(1, 99)}"
            )
            self._specs[fid] = _FixtureSpec(fid, seed, i, status, tp, locked, mismatch, title)
            ids.append(fid)
        return ids

    def reset_fixture(self, fixture_id: str) -> FixtureManifest:
        spec = self._specs.get(fixture_id)
        if spec is None:
            raise KeyError(f"unknown fixture {fixture_id}")
        w = ReleaseWorld()
        focal_id = f"demo-{spec.seed}-{spec.index}"
        version = 2 if spec.mismatch else 1
        w.releases[focal_id] = Release(
            id=focal_id,
            tenant=ACTING_TENANT,
            title=spec.title,
            status=spec.status,
            tests_passed=spec.tests_passed,
            version=version,
            locked=spec.locked,
            tested_version=(1 if spec.tests_passed else 0),
        )
        extra = f"rel-{spec.seed}-{spec.index}-x"
        w.releases[extra] = Release(id=extra, tenant=ACTING_TENANT, title="Extra draft")
        other = f"rel-{spec.seed}-{spec.index}-t2"
        w.releases[other] = Release(id=other, tenant="t2", title="Other tenant's release")
        deleted = f"rel-{spec.seed}-{spec.index}-del"
        w.deleted.add(deleted)
        self.world = w
        self.counter.resets += 1
        focal = w.releases[focal_id]
        return FixtureManifest(
            fixture_id=fixture_id,
            server_id=SERVER_ID,
            version=self.version,
            focal_ref=ResourceRef.build(SERVER_ID, ACTING_TENANT, "release", focal_id),
            expected_visible=focal.visible(),
            id_pools={
                "EXISTING": [focal_id],
                "EXISTING_OTHER": [extra],
                "OTHER_TENANT": [other],
                "STALE_DELETED": [deleted],
                "NONEXISTENT": [f"rel-{spec.seed}-{spec.index}-nope"],
                "MALFORMED": ["", "not an id"],
            },
        )

    def inject_concurrent_update(self, fixture_id: str, op: str) -> None:
        spec = self._specs[fixture_id]
        r = self.world.releases[f"demo-{spec.seed}-{spec.index}"]
        if op == "publish":
            r.status = "PUBLISHED"
        elif op == "archive":
            r.status = "ARCHIVED"
        elif op == "lock":
            r.locked = True
        elif op == "invalidate_tests":
            r.tests_passed = False
        else:
            raise ValueError(f"unknown concurrent op {op}")


def _build_mcp(sb: ReleaseSandbox) -> MCPServer:
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
    def create_release(name: str) -> CreateResult:
        """Create a new release with the given name. Returns its id."""
        return CreateResult(
            **counted(sb.counter, "create_release", guard(rules.create_release))(name)
        )

    @mcp.tool()
    def get_release(release_id: str) -> ReleaseView:
        """Fetch a release: status, whether tests passed, version and title."""
        return ReleaseView(
            **counted(sb.counter, "get_release", guard(rules.get_release))(release_id)
        )

    @mcp.tool()
    def edit_release(release_id: str, title: str) -> OkResult:
        """Change a release's title."""
        return OkResult(
            **counted(sb.counter, "edit_release", guard(rules.edit_release))(release_id, title)
        )

    if sb.version == "v2":

        @mcp.tool()
        def run_tests(release_id: str, suite: str = "default") -> OkResult:
            """Run the test suite for a release."""
            return OkResult(
                **counted(sb.counter, "run_tests", guard(rules.run_tests))(release_id, suite)
            )
    else:

        @mcp.tool()
        def run_tests(release_id: str) -> OkResult:  # type: ignore[misc]
            """Run the test suite for a release."""
            return OkResult(**counted(sb.counter, "run_tests", guard(rules.run_tests))(release_id))

    @mcp.tool()
    def publish_release(release_id: str) -> OkResult:
        """Publish a release."""
        return OkResult(
            **counted(sb.counter, "publish_release", guard(rules.publish_release))(release_id)
        )

    @mcp.tool()
    def archive_release(release_id: str) -> OkResult:
        """Archive a release."""
        return OkResult(
            **counted(sb.counter, "archive_release", guard(rules.archive_release))(release_id)
        )

    return mcp
