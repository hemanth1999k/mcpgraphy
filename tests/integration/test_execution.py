"""Delegated execution gate: no bypass of host approval, unknown offers never write, retries never
duplicate effects, stale snapshots refuse, mismatches become counterexamples (never trust)."""

from __future__ import annotations

import pytest
from mcp import Client

from mcpgraphy.graph.store import GraphStore
from mcpgraphy.learn.pipeline import run_learn
from mcpgraphy.mcp_server import build_offer_server
from mcpgraphy.models import Availability, BehaviorStatus
from mcpgraphy.offers import load_demo_fixture
from mcpgraphy.offers.executor import ExecutionPolicy
from mcpgraphy.sandbox import ReleaseSandbox

pytestmark = pytest.mark.anyio

TOKEN = "host-configured-secret"


@pytest.fixture
async def learned(tmp_path):
    db = tmp_path / "exec.sqlite"
    await run_learn("release_sim", "v1", 0, 12, 120, db, tmp_path / "j.jsonl")
    return str(db)


def _draft_index(sb: ReleaseSandbox, tested: bool) -> int:
    ids = sb.fixtures(0, 12)
    return next(
        i
        for i, f in enumerate(ids)
        if sb._specs[f].status == "DRAFT"
        and sb._specs[f].tests_passed is tested
        and not sb._specs[f].mismatch
    )


async def _setup(db, policy, index=None):
    sb = ReleaseSandbox("v1")
    idx = _draft_index(sb, tested=False) if index is None else index
    resource = f"release:demo-0-{idx}"
    load_demo_fixture(sb, resource)
    server = build_offer_server(db, {"release_sim": sb}, policy=policy)
    return sb, server, resource


async def _offers(client, resource):
    res = await client.call_tool(
        "list_affordances", {"server_id": "release_sim", "resource_ref": resource}
    )
    return {o["tool"]: o for o in res.structured_content["offers"]}


async def _exec(client, offer, args=None, **extra):
    res = await client.call_tool(
        "execute_offered_action",
        {"offer_id": offer["offer_id"], "args": args or offer["argument_template"], **extra},
    )
    assert not res.is_error, res.content
    return res.structured_content


async def test_tools_now_include_execute_and_report(learned):
    _, server, _ = await _setup(learned, ExecutionPolicy())
    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert {"execute_offered_action", "report_counterexample"} <= names


async def test_default_policy_denies_every_effectful_call_but_allows_reads(learned):
    sb, server, resource = await _setup(learned, ExecutionPolicy())
    async with Client(server) as client:
        offers = await _offers(client, resource)
        before = sb.counter.snapshot()
        rec = await _exec(client, offers["run_tests"])
        assert rec["decision"] == "REFUSED" and "POLICY_DENY" in rec["reason"]
        assert sb.counter.snapshot() == before  # not even a read-back happened
        rec = await _exec(client, offers["get_release"])
        assert rec["decision"] == "EXECUTED" and rec["outcome_kind"] == "OK"


async def test_unknown_and_unsupported_offers_never_write(learned):
    sb, server, resource = await _setup(
        learned, ExecutionPolicy(mode="approved", approved_tools=frozenset({"publish_release"}))
    )
    async with Client(server) as client:
        offers = await _offers(client, resource)
        publish = offers["publish_release"]  # DRAFT/untested: observed to fail
        assert publish["availability"] == Availability.NOT_SUPPORTED_FROM_THIS_STATE
        before = sb.counter.snapshot()
        rec = await _exec(client, publish)
        assert rec["decision"] == "REFUSED" and "NOT_SUPPORTED" in rec["reason"]
        assert sb.counter.snapshot() == before
        # a resource that does not exist -> NEEDS_OBSERVATION -> never executes
        missing = await _offers(client, "release:nope")
        before = sb.counter.snapshot()  # listing offers performs a read; executing must not
        rec = await _exec(client, missing["publish_release"])
        assert rec["decision"] == "REFUSED" and "UNKNOWN_OFFER" in rec["reason"]
        assert sb.counter.snapshot() == before


async def test_approval_required_needs_host_approval_not_a_self_issued_token(learned):
    # allow_uncertain so the approval gate (not the uncertainty gate) is what decides here
    sb, server, resource = await _setup(
        learned, ExecutionPolicy(mode="approved", approval_token=TOKEN, allow_uncertain=True)
    )
    rid = resource.split(":")[1]
    async with Client(server) as client:
        offers = await _offers(client, resource)
        rec = await _exec(client, offers["run_tests"])  # no approval needed
        assert rec["decision"] == "EXECUTED" and rec["prediction_matched"] is True
        assert rec["observed_effects"] == ["tests_passed becomes true"]
        offers = await _offers(client, resource)
        publish = offers["publish_release"]
        assert publish["authorization"]["required"] is True
        writes_before = sb.counter.calls.get("publish_release", 0)
        rec = await _exec(client, publish)
        assert rec["decision"] == "REFUSED" and "APPROVAL_REQUIRED" in rec["reason"]
        rec = await _exec(client, publish, approval_token="guess")
        assert rec["decision"] == "REFUSED" and "APPROVAL_REQUIRED" in rec["reason"]
        assert sb.counter.calls.get("publish_release", 0) == writes_before
        rec = await _exec(client, publish, approval_token=TOKEN)
        assert rec["decision"] == "EXECUTED" and rec["outcome_kind"] == "OK"
        assert sb.world.releases[rid].status == "PUBLISHED"
        assert rec["observed_effects"] == ["status becomes PUBLISHED"]


async def test_pre_approved_tool_list_is_the_other_host_approval_path(learned):
    sb, server, resource = await _setup(
        learned,
        ExecutionPolicy(
            mode="approved", approved_tools=frozenset({"publish_release"}), allow_uncertain=True
        ),
    )
    async with Client(server) as client:
        offers = await _offers(client, resource)
        await _exec(client, offers["run_tests"])
        offers = await _offers(client, resource)
        rec = await _exec(client, offers["publish_release"])  # no token, but pre-approved
        assert rec["decision"] == "EXECUTED"
        rec = await _exec(client, offers["archive_release"])  # not pre-approved, no token
        assert rec["decision"] == "REFUSED"


async def test_retry_with_same_idempotency_key_does_not_duplicate_effects(learned):
    sb, server, resource = await _setup(learned, ExecutionPolicy(mode="approved"))
    async with Client(server) as client:
        offers = await _offers(client, resource)
        edit = offers["edit_release"]
        rec1 = await _exec(client, edit, idempotency_key="req-1")
        assert rec1["decision"] == "EXECUTED"
        version_after_first = sb.world.releases[resource.split(":")[1]].version
        rec2 = await _exec(client, edit, idempotency_key="req-1")
        assert rec2["decision"] == "REPLAYED" and rec2["executed"] is False
        assert sb.world.releases[resource.split(":")[1]].version == version_after_first
        # a refusal does not consume a key: the same key can be retried after the refusal
        rid = resource.split(":")[1]
        sb.world.releases[rid].status = "PUBLISHED"
        refused = await _exec(client, edit, idempotency_key="req-2")
        assert refused["decision"] == "REFUSED"
        sb.world.releases[rid].status = "DRAFT"
        retried = await _exec(client, edit, idempotency_key="req-2")
        assert retried["decision"] != "REPLAYED"
        assert rec2["execution_id"] == rec1["execution_id"]
        # the audit survives a server restart (it is in SQLite, not memory)
        with GraphStore(learned) as store:
            assert store.execution(rec1["execution_id"])["decision"] == "EXECUTED"


async def test_stale_snapshot_refuses_without_executing(learned):
    sb, server, resource = await _setup(learned, ExecutionPolicy(mode="approved"))
    async with Client(server) as client:
        offers = await _offers(client, resource)
        edit = offers["edit_release"]
        # the world changes behind the offer's back
        sb.world.releases[resource.split(":")[1]].status = "PUBLISHED"
        before = sb.counter.calls.get("edit_release", 0)
        rec = await _exec(client, edit)
        assert rec["decision"] == "REFUSED" and "STALE_SNAPSHOT" in rec["reason"]
        assert sb.counter.calls.get("edit_release", 0) == before


async def test_mismatch_becomes_counterexample_and_match_never_adds_trust(learned):
    sb, server, resource = await _setup(learned, ExecutionPolicy(mode="approved"))
    rid = resource.split(":")[1]
    async with Client(server) as client:
        offers = await _offers(client, resource)
        edit = offers["edit_release"]
        assert edit["behavior"]["status"] == BehaviorStatus.OBSERVED_SUPPORTED
        # hidden lock appears in production: the supported edit will now fail
        sb.world.releases[rid].locked = True
        rec = await _exec(client, edit)
        assert rec["decision"] == "EXECUTED" and rec["outcome_kind"] == "ERR_STATE"
        assert rec["prediction_matched"] is False
        rep = await client.call_tool("report_counterexample", {"execution_id": rec["execution_id"]})
        body = rep.structured_content
        assert body["recorded"] is True and body["contradicted_transition_ids"]
        again = await client.call_tool(
            "report_counterexample", {"execution_id": rec["execution_id"]}
        )
        assert again.structured_content["recorded"] is False
        # the offer is now honest about it
        offers = await _offers(client, resource)
        assert offers["edit_release"]["behavior"]["status"] == BehaviorStatus.CONTRADICTED
        # a matching execution never raises trust
        sb.world.releases[rid].locked = False
        rec_ok = await _exec(client, offers["run_tests"])
        assert rec_ok["prediction_matched"] is True
        rep_ok = await client.call_tool(
            "report_counterexample", {"execution_id": rec_ok["execution_id"]}
        )
        assert rep_ok.structured_content["recorded"] is False
        assert "never raise trust" in rep_ok.structured_content["reason"]
