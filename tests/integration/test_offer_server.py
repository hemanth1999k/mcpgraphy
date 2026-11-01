"""The action-offer MCP server, called through the official in-memory client."""

from __future__ import annotations

import json

import pytest
from mcp import Client

from mcpgraphy.learn.pipeline import run_learn
from mcpgraphy.mcp_server import build_offer_server
from mcpgraphy.models import Availability, BehaviorStatus
from mcpgraphy.offers import load_demo_fixture
from mcpgraphy.sandbox import ReleaseSandbox

pytestmark = pytest.mark.anyio


@pytest.fixture
async def learned(tmp_path):
    db = tmp_path / "offers.sqlite"
    r = await run_learn("release_sim", "v1", 0, 12, 120, db, tmp_path / "j.jsonl")
    return r, db


async def _affordances(client, resource):
    res = await client.call_tool(
        "list_affordances", {"server_id": "release_sim", "resource_ref": resource}
    )
    assert not res.is_error, res.content
    return res.structured_content


def _offer(data, tool):
    return next(o for o in data["offers"] if o["tool"] == tool)


async def test_tools_exposed_including_controlled_execution(learned):
    _, db = learned
    sb = ReleaseSandbox("v1")
    server = build_offer_server(str(db), {"release_sim": sb})
    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == {
        "inspect_server",
        "inspect_resource",
        "list_affordances",
        "explain_affordance",
        "simulate_action",
        "get_behavior_graph",
        "execute_offered_action",
        "report_counterexample",
    }


async def test_offers_differ_between_draft_and_published(learned):
    _, db = learned
    sb = ReleaseSandbox("v1")
    server = build_offer_server(str(db), {"release_sim": sb})
    # find a DRAFT/untested and a PUBLISHED demo fixture under seed 0
    ids = sb.fixtures(0, 12)
    draft = next(
        i
        for i, f in enumerate(ids)
        if sb._specs[f].status == "DRAFT" and not sb._specs[f].tests_passed
    )
    published = next(
        i
        for i, f in enumerate(ids)
        if sb._specs[f].status == "PUBLISHED" and not sb._specs[f].locked
    )
    async with Client(server) as client:
        load_demo_fixture(sb, f"release:demo-0-{draft}")
        d = await _affordances(client, f"release:demo-0-{draft}")
        load_demo_fixture(sb, f"release:demo-0-{published}")
        p = await _affordances(client, f"release:demo-0-{published}")
    assert d["observed_state"] == {"status": "DRAFT", "tests_passed": False}
    assert p["observed_state"]["status"] == "PUBLISHED"
    # spec §5.4 shape
    for o in d["offers"]:
        assert {
            "offer_id",
            "server_id",
            "resource_ref",
            "tool",
            "argument_template",
            "behavior",
            "authorization",
            "availability",
            "uncertainty",
        } <= set(o)
        assert o["uncertainty"]["possible_unobserved_effects"] is True
        assert o["authorization"]["granted"] is False
    # DRAFT/untested: edit is a plain candidate, publish is observed-unsupported
    assert _offer(d, "edit_release")["availability"] == Availability.CANDIDATE
    assert _offer(d, "edit_release")["behavior"]["status"] == BehaviorStatus.OBSERVED_SUPPORTED
    assert (
        _offer(d, "publish_release")["availability"] == Availability.NOT_SUPPORTED_FROM_THIS_STATE
    )
    assert _offer(d, "run_tests")["behavior"]["predicted_effects"] == ["tests_passed becomes true"]
    # PUBLISHED: edit is unsupported, archive is uncertain (hidden lock) and requires approval
    assert _offer(p, "edit_release")["availability"] == Availability.NOT_SUPPORTED_FROM_THIS_STATE
    archive = _offer(p, "archive_release")
    assert archive["authorization"]["required"] is True
    assert archive["availability"] in (
        Availability.UNCERTAIN_CANDIDATE,
        Availability.CANDIDATE_REQUIRES_APPROVAL,
    )
    # ranking puts candidates before unsupported
    ranks = [o["availability"] for o in d["offers"]]
    assert ranks.index(Availability.CANDIDATE) < ranks.index(
        Availability.NOT_SUPPORTED_FROM_THIS_STATE
    )


async def test_draft_tested_publish_is_uncertain_candidate_with_alternatives(learned):
    _, db = learned
    sb = ReleaseSandbox("v1")
    server = build_offer_server(str(db), {"release_sim": sb})
    ids = sb.fixtures(0, 12)
    idx = next(
        i
        for i, f in enumerate(ids)
        if sb._specs[f].status == "DRAFT"
        and sb._specs[f].tests_passed
        and not sb._specs[f].mismatch
    )
    async with Client(server) as client:
        load_demo_fixture(sb, f"release:demo-0-{idx}")
        d = await _affordances(client, f"release:demo-0-{idx}")
        publish = _offer(d, "publish_release")
        assert publish["behavior"]["status"] == BehaviorStatus.CONTRADICTED
        assert publish["availability"] == Availability.UNCERTAIN_CANDIDATE
        kinds = {a["outcome_kind"] for a in publish["uncertainty"]["alternative_outcomes"]}
        assert kinds == {"OK", "ERR_STATE"}
        assert publish["authorization"]["required"] is True
        # explain links to precise experiments and a counterexample
        ex = await client.call_tool("explain_affordance", {"offer_id": publish["offer_id"]})
        assert not ex.is_error
        e = ex.structured_content
        assert e["supporting_experiments"] and e["counterexamples"]
        assert any("hidden condition" in c for c in e["caveats"])


async def test_simulate_action_makes_no_downstream_calls(learned):
    _, db = learned
    sb = ReleaseSandbox("v1")
    server = build_offer_server(str(db), {"release_sim": sb})
    async with Client(server) as client:
        load_demo_fixture(sb, "release:demo-0-0")
        d = await _affordances(client, "release:demo-0-0")
        offer = d["offers"][0]
        before = sb.counter.snapshot()
        sim = await client.call_tool(
            "simulate_action", {"offer_id": offer["offer_id"], "args": offer["argument_template"]}
        )
        assert not sim.is_error
        assert sim.structured_content["executed"] is False
        assert sb.counter.snapshot() == before
        bad = await client.call_tool("simulate_action", {"offer_id": offer["offer_id"], "args": {}})
        assert bad.is_error


async def test_unknown_offer_id_is_tool_error(learned):
    _, db = learned
    server = build_offer_server(str(db), {"release_sim": ReleaseSandbox("v1")})
    async with Client(server) as client:
        res = await client.call_tool("explain_affordance", {"offer_id": "offer_nope"})
        assert res.is_error


async def test_missing_resource_is_unknown_not_a_candidate(learned):
    _, db = learned
    sb = ReleaseSandbox("v1")
    server = build_offer_server(str(db), {"release_sim": sb})
    async with Client(server) as client:
        load_demo_fixture(sb, "release:demo-0-0")
        d = await _affordances(client, "release:nope")
        assert d["exists"] is False
        # a NOT_FOUND read-back is a real observed state (exists=False); the sandbox learner never
        # probed from it, so everything is UNKNOWN_UNTESTED and nothing is a candidate
        assert {o["availability"] for o in d["offers"]} == {Availability.UNKNOWN_UNTESTED}
        assert {o["behavior"]["status"] for o in d["offers"]} == {BehaviorStatus.UNKNOWN}


async def test_get_behavior_graph_json_round_trips_and_inspect_server_reports_drift(learned):
    _, db = learned
    server = build_offer_server(str(db), {"release_sim": ReleaseSandbox("v2")})
    async with Client(server) as client:
        g = await client.call_tool("get_behavior_graph", {"server_id": "release_sim"})
        data = json.loads(g.structured_content["content"])
        assert data["format_version"] == 1 and data["transitions"]
        m = await client.call_tool(
            "get_behavior_graph", {"server_id": "release_sim", "format": "mermaid"}
        )
        assert m.structured_content["content"].startswith("flowchart")
        insp = await client.call_tool("inspect_server", {"server_id": "release_sim"})
        assert insp.structured_content["fingerprint_drift"] is True  # model is v1, server v2
