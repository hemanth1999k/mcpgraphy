"""release_sim: one success and one failure per tool, through the real MCP client."""

import pytest

from mcpgraphy.client_adapter import InMemoryToolClient
from mcpgraphy.models import OutcomeKind
from mcpgraphy.sandbox import ReleaseSandbox

pytestmark = pytest.mark.anyio


def _fixture_with(sb: ReleaseSandbox, **want):
    """Find a fixture whose focal release matches the wanted visible/hidden cell."""
    for fid in sb.fixtures(seed=0, n=11):
        spec = sb._specs[fid]
        if all(getattr(spec, k) == v for k, v in want.items()):
            return fid
    raise AssertionError(f"no fixture with {want}")


async def _call(sb, tool, **args):
    async with InMemoryToolClient(sb.mcp) as c:
        return await c.call(tool, args)


async def test_tools_list_excludes_control_api(release_sandbox):
    async with InMemoryToolClient(release_sandbox.mcp) as c:
        names = {t.name for t in await c.list_tools()}
    assert names == {
        "create_release",
        "get_release",
        "edit_release",
        "run_tests",
        "publish_release",
        "archive_release",
    }
    assert not names & {"reset_fixture", "fixtures", "inject_concurrent_update"}


async def test_create_release_success_and_failure(release_sandbox):
    sb = release_sandbox
    sb.reset_fixture(sb.fixtures(0, 1)[0])
    ok = await _call(sb, "create_release", name="Hotfix")
    assert ok.outcome.kind is OutcomeKind.OK and ok.raw["id"].startswith("rel-new-")
    bad = await _call(sb, "create_release", name="")
    assert bad.outcome.kind is OutcomeKind.ERR_VALIDATION and bad.outcome.code == "INVALID_ARGUMENT"


async def test_get_release_success_and_not_found(release_sandbox):
    sb = release_sandbox
    m = sb.reset_fixture(sb.fixtures(0, 1)[0])
    ok = await _call(sb, "get_release", release_id=m.id_pools["EXISTING"][0])
    assert ok.outcome.kind is OutcomeKind.OK
    assert set(ok.raw) == {"id", "status", "tests_passed", "version", "title"}  # no hidden fields
    missing = await _call(sb, "get_release", release_id=m.id_pools["NONEXISTENT"][0])
    assert missing.outcome.kind is OutcomeKind.ERR_NOT_FOUND


async def test_other_tenant_id_returns_not_found_not_permission(release_sandbox):
    sb = release_sandbox
    m = sb.reset_fixture(sb.fixtures(0, 1)[0])
    r = await _call(sb, "get_release", release_id=m.id_pools["OTHER_TENANT"][0])
    assert r.outcome.kind is OutcomeKind.ERR_NOT_FOUND


async def test_stale_deleted_id_is_not_found(release_sandbox):
    sb = release_sandbox
    m = sb.reset_fixture(sb.fixtures(0, 1)[0])
    r = await _call(sb, "publish_release", release_id=m.id_pools["STALE_DELETED"][0])
    assert r.outcome.kind is OutcomeKind.ERR_NOT_FOUND


async def test_edit_release_success_and_failure(release_sandbox):
    sb = release_sandbox
    fid = _fixture_with(sb, status="DRAFT", locked=False, tests_passed=True)
    m = sb.reset_fixture(fid)
    rid = m.id_pools["EXISTING"][0]
    ok = await _call(sb, "edit_release", release_id=rid, title="New title")
    assert ok.outcome.kind is OutcomeKind.OK
    after = await _call(sb, "get_release", release_id=rid)
    assert after.raw["title"] == "New title" and after.raw["tests_passed"] is False
    assert after.raw["version"] == m.expected_visible["version"] + 1
    sb.reset_fixture(_fixture_with(sb, status="PUBLISHED"))
    bad = await _call(
        sb,
        "edit_release",
        release_id=sb.reset_fixture(_fixture_with(sb, status="PUBLISHED")).id_pools["EXISTING"][0],
        title="x",
    )
    assert bad.outcome.kind is OutcomeKind.ERR_STATE


async def test_locked_published_archive_is_state_conflict_without_leaking_cause(release_sandbox):
    sb = release_sandbox
    m = sb.reset_fixture(_fixture_with(sb, status="PUBLISHED", locked=True))
    r = await _call(sb, "archive_release", release_id=m.id_pools["EXISTING"][0])
    assert r.outcome.kind is OutcomeKind.ERR_STATE
    assert "lock" not in r.outcome.message.lower()


async def test_run_tests_success_and_failure(release_sandbox):
    sb = release_sandbox
    m = sb.reset_fixture(_fixture_with(sb, status="DRAFT", tests_passed=False))
    rid = m.id_pools["EXISTING"][0]
    assert (await _call(sb, "run_tests", release_id=rid)).outcome.kind is OutcomeKind.OK
    assert (await _call(sb, "get_release", release_id=rid)).raw["tests_passed"] is True
    m = sb.reset_fixture(_fixture_with(sb, status="ARCHIVED"))
    r = await _call(sb, "run_tests", release_id=m.id_pools["EXISTING"][0])
    assert r.outcome.kind is OutcomeKind.ERR_STATE


async def test_publish_success_requires_tests_and_no_mismatch(release_sandbox):
    sb = release_sandbox
    m = sb.reset_fixture(
        _fixture_with(sb, status="DRAFT", tests_passed=True, locked=False, mismatch=False)
    )
    rid = m.id_pools["EXISTING"][0]
    assert (await _call(sb, "publish_release", release_id=rid)).outcome.kind is OutcomeKind.OK
    assert (await _call(sb, "get_release", release_id=rid)).raw["status"] == "PUBLISHED"
    for cell in (
        {"status": "DRAFT", "tests_passed": False},
        {"status": "DRAFT", "tests_passed": True, "locked": False, "mismatch": True},
        {"status": "PUBLISHED"},
    ):
        m = sb.reset_fixture(_fixture_with(sb, **cell))
        r = await _call(sb, "publish_release", release_id=m.id_pools["EXISTING"][0])
        assert r.outcome.kind is OutcomeKind.ERR_STATE, cell


async def test_mismatch_is_repaired_by_running_tests(release_sandbox):
    sb = release_sandbox
    m = sb.reset_fixture(
        _fixture_with(sb, status="DRAFT", tests_passed=True, locked=False, mismatch=True)
    )
    rid = m.id_pools["EXISTING"][0]
    assert (await _call(sb, "run_tests", release_id=rid)).outcome.kind is OutcomeKind.OK
    assert (await _call(sb, "publish_release", release_id=rid)).outcome.kind is OutcomeKind.OK


async def test_archive_v1_from_draft_conflicts_and_from_published_succeeds(release_sandbox):
    sb = release_sandbox
    m = sb.reset_fixture(_fixture_with(sb, status="DRAFT", locked=False))
    r = await _call(sb, "archive_release", release_id=m.id_pools["EXISTING"][0])
    assert r.outcome.kind is OutcomeKind.ERR_STATE
    m = sb.reset_fixture(_fixture_with(sb, status="PUBLISHED", locked=False))
    r = await _call(sb, "archive_release", release_id=m.id_pools["EXISTING"][0])
    assert r.outcome.kind is OutcomeKind.OK


async def test_v2_archive_from_draft_succeeds_and_run_tests_has_suite(release_sandbox_v2):
    sb = release_sandbox_v2
    m = sb.reset_fixture(_fixture_with(sb, status="DRAFT", locked=False))
    r = await _call(sb, "archive_release", release_id=m.id_pools["EXISTING"][0])
    assert r.outcome.kind is OutcomeKind.OK
    async with InMemoryToolClient(sb.mcp) as c:
        tools = {t.name: t for t in await c.list_tools()}
    assert "suite" in tools["run_tests"].input_schema["properties"]


async def test_schema_validation_error_is_err_validation(release_sandbox):
    sb = release_sandbox
    sb.reset_fixture(sb.fixtures(0, 1)[0])
    r = await _call(sb, "get_release", release_id=5)
    assert r.outcome.kind is OutcomeKind.ERR_VALIDATION


def test_fixture_generation_deterministic_for_seed():
    a = ReleaseSandbox("v1")
    b = ReleaseSandbox("v1")
    ids_a, ids_b = a.fixtures(7, 20), b.fixtures(7, 20)
    assert ids_a == ids_b
    assert [a._specs[i] for i in ids_a] == [b._specs[i] for i in ids_b]
    assert a.fixtures(8, 20) != ids_a
    assert len(set(ids_a)) == 20


def test_fixture_grid_covers_all_11_cells():
    sb = ReleaseSandbox("v1")
    ids = sb.fixtures(3, 11)
    cells = {(s.status, s.tests_passed, s.locked, s.mismatch) for s in (sb._specs[i] for i in ids)}
    assert len(cells) == 11


def test_manifest_never_contains_hidden_fields():
    sb = ReleaseSandbox("v1")
    m = sb.reset_fixture(sb.fixtures(0, 1)[0])
    text = m.model_dump_json()
    assert "locked" not in text and "tested_version" not in text


def test_reset_restores_world_and_counts(release_sandbox):
    sb = release_sandbox
    fid = sb.fixtures(0, 1)[0]
    m = sb.reset_fixture(fid)
    sb.world.releases[m.id_pools["EXISTING"][0]].status = "ARCHIVED"
    m2 = sb.reset_fixture(fid)
    assert sb.world.releases[m2.id_pools["EXISTING"][0]].status == m.expected_visible["status"]
    assert sb.counter.resets == 2
