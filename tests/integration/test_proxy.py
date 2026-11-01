"""Proxy mode against a real downstream process over stdio (the simulator run as a subprocess).

An agent uses the proxied tools exactly as it would use the server directly. The proxy learns
from those calls alone (no probing, no resets), then serves offers, warns, or blocks.
"""

from __future__ import annotations

import sys

import pytest
from mcp import Client, StdioServerParameters

from mcpgraphy.proxy import Proxy, ProxyPolicy
from mcpgraphy.proxy.adapter import EXAMPLE_ADAPTER, AdapterSpec

pytestmark = pytest.mark.anyio


def _downstream() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcpgraphy.cli", "sandbox", "--server", "release_sim", "--fixture-seed", "0"],
    )


def _proxy(tmp_path, policy: ProxyPolicy, refresh_every: int = 4) -> Proxy:
    return Proxy(
        AdapterSpec(**EXAMPLE_ADAPTER),
        _downstream(),
        db_path=tmp_path / "proxy.sqlite",
        journal_path=tmp_path / "proxy.jsonl",
        policy=policy,
        refresh_every=refresh_every,
    )


async def _agent_workflow(client: Client, rounds: int = 3) -> list[str]:
    """What a naive agent does: create a release, try to publish, fail, run tests, publish."""
    ids = []
    for _ in range(rounds):
        created = await client.call_tool("create_release", {"name": "Release"})
        rid = created.structured_content["id"]
        ids.append(rid)
        first = await client.call_tool("publish_release", {"release_id": rid})
        assert first.is_error  # untested draft
        assert not (await client.call_tool("run_tests", {"release_id": rid})).is_error
        assert not (await client.call_tool("publish_release", {"release_id": rid})).is_error
    return ids


async def test_tools_pass_through_and_ours_are_added(tmp_path):
    async with _proxy(tmp_path, ProxyPolicy(mode="observe")) as proxy:
        async with Client(proxy.server) as client:
            names = {t.name for t in (await client.list_tools()).tools}
    assert {"create_release", "get_release", "publish_release", "run_tests"} <= names
    assert {
        "mcpgraphy_affordances",
        "mcpgraphy_explain",
        "mcpgraphy_graph",
        "mcpgraphy_status",
    } <= names


async def test_proxy_learns_from_passive_calls_and_serves_offers(tmp_path):
    async with _proxy(tmp_path, ProxyPolicy(mode="observe")) as proxy:
        async with Client(proxy.server) as client:
            await _agent_workflow(client, rounds=3)
            status = (await client.call_tool("mcpgraphy_status", {})).structured_content
            assert status["calls_recorded"] == 9 and status["model_version"]
            created = await client.call_tool("create_release", {"name": "Fresh"})
            rid = created.structured_content["id"]
            aff = await client.call_tool(
                "mcpgraphy_affordances", {"resource_ref": f"release:{rid}"}
            )
            body = aff.structured_content
            assert body["observed_state"] == {"status": "DRAFT", "tests_passed": False}
            offers = {o["tool"]: o for o in body["offers"]}
            assert offers["publish_release"]["availability"] == "NOT_SUPPORTED_FROM_THIS_STATE"
            assert offers["run_tests"]["availability"] == "CANDIDATE"
            assert offers["run_tests"]["behavior"]["predicted_effects"] == [
                "tests_passed becomes true"
            ]
            ex = await client.call_tool(
                "mcpgraphy_explain", {"offer_id": offers["publish_release"]["offer_id"]}
            )
            assert ex.structured_content["exclusions"]
            g = await client.call_tool("mcpgraphy_graph", {"format": "mermaid"})
            assert g.structured_content["content"].startswith("flowchart")


async def test_warn_mode_passes_through_with_a_note(tmp_path):
    async with _proxy(tmp_path, ProxyPolicy(mode="warn")) as proxy:
        async with Client(proxy.server) as client:
            await _agent_workflow(client, rounds=2)
            created = await client.call_tool("create_release", {"name": "Fresh"})
            rid = created.structured_content["id"]
            res = await client.call_tool("publish_release", {"release_id": rid})
            assert res.is_error  # the real server still answered
            texts = [c.text for c in res.content if getattr(c, "type", "") == "text"]
            assert any("mcpgraphy:" in t and "observed to fail" in t for t in texts)


async def test_block_mode_refuses_known_failures_and_unapproved_destructive_calls(tmp_path):
    async with _proxy(tmp_path, ProxyPolicy(mode="observe")) as proxy:
        async with Client(proxy.server) as client:
            await _agent_workflow(client, rounds=2)
    # a new proxy session on the same journal/db keeps what was learned
    policy = ProxyPolicy(mode="block", approved_tools=frozenset({"publish_release"}))
    async with _proxy(tmp_path, policy) as proxy:
        assert proxy.model_version
        async with Client(proxy.server) as client:
            created = await client.call_tool("create_release", {"name": "Fresh"})
            rid = created.structured_content["id"]
            blocked = await client.call_tool("publish_release", {"release_id": rid})
            assert blocked.is_error and "mcpgraphy blocked" in blocked.content[0].text
            assert "observed to fail" in blocked.content[0].text
            # the downstream never saw the call: the release is still an untested draft
            info = await client.call_tool("get_release", {"release_id": rid})
            assert info.structured_content["tests_passed"] is False
            assert not (await client.call_tool("run_tests", {"release_id": rid})).is_error
            ok = await client.call_tool("publish_release", {"release_id": rid})
            assert not ok.is_error  # pre-approved and supported from DRAFT/tested
            arch = await client.call_tool("archive_release", {"release_id": rid})
            assert arch.is_error and "not pre-approved" in arch.content[0].text
            status = (await client.call_tool("mcpgraphy_status", {})).structured_content
            assert status["blocked"] == 2


async def test_unknown_tools_are_forwarded_untouched(tmp_path):
    async with _proxy(tmp_path, ProxyPolicy(mode="block")) as proxy:
        async with Client(proxy.server) as client:
            res = await client.call_tool(
                "create_release", {"name": "X"}
            )  # no id param: unknown state
            assert not res.is_error and res.structured_content["id"].startswith("rel-new-")
