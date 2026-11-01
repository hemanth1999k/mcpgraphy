"""ticket_sim: one success and one failure per tool, parameter namespaces, hidden permission."""

import pytest

from mcpgraphy.client_adapter import InMemoryToolClient
from mcpgraphy.models import OutcomeKind
from mcpgraphy.sandbox import TicketSandbox

pytestmark = pytest.mark.anyio


def _fixture_with(sb: TicketSandbox, **want):
    for fid in sb.fixtures(seed=0, n=10):
        spec = sb._specs[fid]
        if all(getattr(spec, k) == v for k, v in want.items()):
            return fid
    raise AssertionError(f"no fixture with {want}")


async def _call(sb, tool, **args):
    async with InMemoryToolClient(sb.mcp) as c:
        return await c.call(tool, args)


async def test_tools_list_excludes_control_api(ticket_sandbox):
    async with InMemoryToolClient(ticket_sandbox.mcp) as c:
        names = {t.name for t in await c.list_tools()}
    assert names == {
        "create_ticket",
        "get_ticket",
        "assign_ticket",
        "move_ticket",
        "close_ticket",
        "list_users",
    }


async def test_create_ticket_success_and_failures(ticket_sandbox):
    sb = ticket_sandbox
    sb.reset_fixture(sb.fixtures(0, 1)[0])
    ok = await _call(sb, "create_ticket", project_id="p-alpha", title="Do thing")
    assert ok.outcome.kind is OutcomeKind.OK
    assert (
        await _call(sb, "create_ticket", project_id="p-gamma", title="x")
    ).outcome.kind is OutcomeKind.ERR_NOT_FOUND
    assert (
        await _call(sb, "create_ticket", project_id="p-alpha", title="")
    ).outcome.kind is OutcomeKind.ERR_VALIDATION


async def test_get_ticket_and_list_users(ticket_sandbox):
    sb = ticket_sandbox
    m = sb.reset_fixture(sb.fixtures(0, 1)[0])
    ok = await _call(sb, "get_ticket", ticket_id=m.id_pools["EXISTING"][0])
    assert set(ok.raw) == {"id", "state", "assignee", "project_id", "title"}
    assert (
        await _call(sb, "get_ticket", ticket_id=m.id_pools["STALE_DELETED"][0])
    ).outcome.kind is OutcomeKind.ERR_NOT_FOUND
    users = await _call(sb, "list_users", project_id="p-alpha")
    assert users.raw["users"] == ["u-alpha-1", "u-alpha-2"]
    assert (
        await _call(sb, "list_users", project_id="p-gamma")
    ).outcome.kind is OutcomeKind.ERR_NOT_FOUND


async def test_assign_user_from_similar_named_project_rejected(ticket_sandbox):
    sb = ticket_sandbox
    m = sb.reset_fixture(_fixture_with(sb, state="OPEN", write_denied=False))
    tid = m.id_pools["EXISTING"][0]
    ok = await _call(sb, "assign_ticket", ticket_id=tid, user_id="u-alpha-2")
    assert ok.outcome.kind is OutcomeKind.OK
    bad = await _call(sb, "assign_ticket", ticket_id=tid, user_id="u-ops-1")
    assert bad.outcome.kind is OutcomeKind.ERR_VALIDATION
    m = sb.reset_fixture(_fixture_with(sb, state="CLOSED", write_denied=False))
    closed = await _call(
        sb, "assign_ticket", ticket_id=m.id_pools["EXISTING"][0], user_id="u-alpha-2"
    )
    assert closed.outcome.kind is OutcomeKind.ERR_STATE


async def test_hidden_write_denied_yields_permission_error(ticket_sandbox):
    sb = ticket_sandbox
    m = sb.reset_fixture(_fixture_with(sb, state="OPEN", write_denied=True))
    r = await _call(sb, "assign_ticket", ticket_id=m.id_pools["EXISTING"][0], user_id="u-alpha-2")
    assert r.outcome.kind is OutcomeKind.ERR_PERMISSION


async def test_move_requires_assignee_and_legal_transition(ticket_sandbox):
    sb = ticket_sandbox
    m = sb.reset_fixture(_fixture_with(sb, state="OPEN", has_assignee=False, write_denied=False))
    tid = m.id_pools["EXISTING"][0]
    r = await _call(sb, "move_ticket", ticket_id=tid, state="IN_PROGRESS")
    assert r.outcome.kind is OutcomeKind.ERR_STATE
    m = sb.reset_fixture(_fixture_with(sb, state="OPEN", has_assignee=True, write_denied=False))
    tid = m.id_pools["EXISTING"][0]
    assert (
        await _call(sb, "move_ticket", ticket_id=tid, state="IN_PROGRESS")
    ).outcome.kind is OutcomeKind.OK
    assert (
        await _call(sb, "move_ticket", ticket_id=tid, state="OPEN")
    ).outcome.kind is OutcomeKind.ERR_STATE
    assert (
        await _call(sb, "move_ticket", ticket_id=tid, state="LIMBO")
    ).outcome.kind is OutcomeKind.ERR_VALIDATION


async def test_close_only_from_done(ticket_sandbox):
    sb = ticket_sandbox
    m = sb.reset_fixture(_fixture_with(sb, state="DONE", write_denied=False))
    assert (
        await _call(sb, "close_ticket", ticket_id=m.id_pools["EXISTING"][0])
    ).outcome.kind is OutcomeKind.OK
    m = sb.reset_fixture(_fixture_with(sb, state="OPEN", write_denied=False))
    assert (
        await _call(sb, "close_ticket", ticket_id=m.id_pools["EXISTING"][0])
    ).outcome.kind is OutcomeKind.ERR_STATE


def test_fixture_grid_covers_10_cells_and_is_deterministic():
    a, b = TicketSandbox(), TicketSandbox()
    ids = a.fixtures(5, 10)
    assert ids == b.fixtures(5, 10)
    cells = {(s.state, s.has_assignee, s.write_denied) for s in (a._specs[i] for i in ids)}
    assert len(cells) == 10


def test_manifest_never_contains_hidden_fields():
    sb = TicketSandbox()
    m = sb.reset_fixture(sb.fixtures(0, 1)[0])
    assert "write_denied" not in m.model_dump_json()
