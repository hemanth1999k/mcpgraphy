"""The proxy server (low-level SDK API so downstream tool schemas pass through untouched)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mcp import Client, types
from mcp.server.lowlevel import Server

from mcpgraphy.client_adapter import RemoteToolClient, record_from_result
from mcpgraphy.graph.builder import materialize
from mcpgraphy.graph.export import dumps, export_json, export_mermaid
from mcpgraphy.graph.store import GraphStore
from mcpgraphy.learn.abstract import features_of
from mcpgraphy.models import (
    AbstractState,
    Availability,
    EffectClass,
    Experiment,
    Features,
    Offer,
    ServerVersion,
    Snapshot,
    canonical_json,
    make_model_version,
    make_run_id,
)
from mcpgraphy.observe.journal import Journal
from mcpgraphy.offers.explain import explain
from mcpgraphy.offers.selector import classify_rows, resolve_state, select_offers
from mcpgraphy.proxy.adapter import AdapterSpec, action_class_for_call, id_value

OUR_PREFIX = "mcpgraphy_"


@dataclass(frozen=True)
class ProxyPolicy:
    """observe: log only. warn: pass everything through, annotate risky calls.
    block: refuse calls the evidence says fail here, and destructive / approval-required tools
    that the host did not pre-approve."""

    mode: Literal["observe", "warn", "block"] = "warn"
    approved_tools: frozenset[str] = field(default_factory=frozenset)
    allow_uncertain: bool = True


@dataclass
class Gate:
    label: str  # SUPPORTED | UNCERTAIN | UNKNOWN | NOT_SUPPORTED
    note: str | None
    blocked: bool
    reason: str | None = None


class Proxy:
    def __init__(
        self,
        adapter: AdapterSpec,
        downstream: Any,
        db_path: str | Path,
        journal_path: str | Path,
        policy: ProxyPolicy | None = None,
        refresh_every: int = 5,
    ) -> None:
        self.adapter = adapter
        self.downstream = downstream
        self.db_path = str(db_path)
        self.journal_path = Path(journal_path)
        self.policy = policy or ProxyPolicy()
        self.refresh_every = max(refresh_every, 1)
        self._client: Client | None = None
        self.remote: RemoteToolClient | None = None
        self.tools: list[types.Tool] = []
        self.spec = None
        self.sv: ServerVersion | None = None
        self.journal: Journal | None = None
        self.run_id = make_run_id(
            adapter.server_id,
            adapter.version,
            0,
            0,
            0,
            {"proxy": True, "journal": str(self.journal_path)},
        )
        self.model_version: str | None = None
        self.calls = 0
        self.seq = 0
        self.blocked = 0
        self.warned = 0
        self._offers: dict[str, tuple[Offer, str]] = {}
        self.server = Server(
            f"mcpgraphy-proxy-{adapter.server_id}",
            instructions=(
                "Proxy in front of an MCP server. Tools are passed through unchanged; mcpgraphy_* "
                "tools expose learned, uncertainty-aware offers. Call mcpgraphy_affordances before "
                "acting on a resource."
            ),
            on_list_tools=self._on_list_tools,
            on_call_tool=self._on_call_tool,
        )

    # ---------------------------------------------------------------- lifecycle
    async def __aenter__(self) -> Proxy:
        self._client = Client(self.downstream)
        await self._client.__aenter__()
        self.remote = RemoteToolClient(self._client)
        listed = await self._client.list_tools()
        self.tools = list(listed.tools)
        specs = await self.remote.list_tools()
        self.spec = self.adapter.build_spec(specs)
        self.adapter.register_extractor()
        self.sv = ServerVersion.build(self.adapter.server_id, self.adapter.version, specs)
        self.journal = Journal(self.journal_path)
        if self.journal.seq == 0:
            self.journal.append(
                "run_start",
                {
                    "run_id": self.run_id,
                    "model_version": make_model_version(
                        self.adapter.server_id, self.adapter.version, self.run_id
                    ),
                    "server_id": self.adapter.server_id,
                    "version": self.adapter.version,
                    "server_version_id": self.sv.server_version_id,
                    "fingerprint": self.sv.fingerprint,
                    "tools": [t.to_dict() for t in self.sv.tools],
                    "seed": 0,
                    "fixtures": 0,
                    "budget": 0,
                    "ablation": "proxy",
                    "base_model_version": None,
                    "reset_supported": False,
                },
            )
        else:
            self.seq = _max_seq(self.journal_path)
            self._materialize()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self.calls:
            self._materialize()
        if self.journal:
            self.journal.close()
        if self._client:
            await self._client.__aexit__(*exc)

    # ---------------------------------------------------------------- MCP handlers
    async def _on_list_tools(self, ctx: Any, params: Any) -> types.ListToolsResult:
        ours = [
            types.Tool(
                name=OUR_PREFIX + "affordances",
                description="Learned action offers for a resource's current observed state, "
                "with evidence, uncertainty and approval requirements. Call before acting.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "resource_ref": {
                            "type": "string",
                            "description": f"'{self.adapter.resource_class}:<id>' or the id",
                        },
                        "goal": {
                            "type": "string",
                            "description": "optional feature=value to promote",
                        },
                    },
                    "required": ["resource_ref"],
                },
            ),
            types.Tool(
                name=OUR_PREFIX + "explain",
                description="Guard, evidence, counterexamples and caveats for an offer id.",
                input_schema={
                    "type": "object",
                    "properties": {"offer_id": {"type": "string"}},
                    "required": ["offer_id"],
                },
            ),
            types.Tool(
                name=OUR_PREFIX + "graph",
                description="Export the learned behavior graph (json or mermaid).",
                input_schema={
                    "type": "object",
                    "properties": {"format": {"type": "string", "enum": ["json", "mermaid"]}},
                },
            ),
            types.Tool(
                name=OUR_PREFIX + "status",
                description="What the proxy has learned so far and the active policy.",
                input_schema={"type": "object", "properties": {}},
            ),
        ]
        return types.ListToolsResult(tools=self.tools + ours)

    async def _on_call_tool(
        self, ctx: Any, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        name, args = params.name, dict(params.arguments or {})
        try:
            if name.startswith(OUR_PREFIX):
                return await self._ours(name[len(OUR_PREFIX) :], args)
            return await self._proxied(name, args)
        except Exception as e:  # never take the downstream server down with us
            return _error(f"mcpgraphy proxy error: {type(e).__name__}: {e}")

    # ---------------------------------------------------------------- proxied calls
    async def _proxied(self, name: str, args: dict[str, Any]) -> types.CallToolResult:
        assert self._client is not None and self.spec is not None
        known = any(t.name == name for t in self.spec.tools)
        rid = id_value(self.spec, name, args) if known else None
        before: Snapshot | None = None
        if rid is not None and name != self.spec.observation_tool:
            before = await self._observe(rid)
        gate = self._gate(name, args, before) if known else Gate("UNKNOWN", None, False)
        if gate.blocked:
            self.blocked += 1
            return _error(f"mcpgraphy blocked this call: {gate.reason}")
        result = await self._client.call_tool(name, args)
        after = await self._observe(rid) if before is not None else None
        if known and name != self.spec.observation_tool:
            self._record(name, args, before, after, result)
        if self.policy.mode == "warn" and gate.note:
            self.warned += 1
            result = result.model_copy(
                update={
                    "content": list(result.content)
                    + [types.TextContent(type="text", text=gate.note)]
                }
            )
        return result

    def _gate(self, name: str, args: dict[str, Any], before: Snapshot | None) -> Gate:
        assert self.spec is not None
        policy_tool = self.spec.tool(name)
        label, note = "UNKNOWN", None
        if self.model_version and before is not None:
            exists = before.raw is not None and before.outcome.kind.is_ok
            feats: Features = features_of(self.spec.resource_class, before)
            state = AbstractState.build(self.spec.server_id, self.spec.resource_class, feats)
            ac = action_class_for_call(self.spec, name, args, exists)
            with GraphStore(self.db_path) as store:
                rows = store.transitions_for(self.model_version, state.state_id, ac.action_class_id)
                status, _obs, availability, _req, effects, evidence, _alts = classify_rows(
                    rows, store, policy_tool.approval_required
                )
            label = {
                Availability.CANDIDATE: "SUPPORTED",
                Availability.CANDIDATE_REQUIRES_APPROVAL: "SUPPORTED",
                Availability.UNCERTAIN_CANDIDATE: "UNCERTAIN",
                Availability.UNKNOWN_UNTESTED: "UNKNOWN",
                Availability.NOT_SUPPORTED_FROM_THIS_STATE: "NOT_SUPPORTED",
                Availability.NEEDS_OBSERVATION: "UNKNOWN",
            }[availability]
            pretty = ", ".join(f"{k}={v}" for k, v in feats.items())
            if label == "NOT_SUPPORTED":
                note = (
                    f"mcpgraphy: {name} was observed to fail from state [{pretty}] in "
                    f"{len(evidence)} prior call(s) ({', '.join(effects)}). "
                    f"Call mcpgraphy_affordances('{self.spec.resource_class}:"
                    f"{args.get(self.spec.id_param)}') for alternatives."
                )
            elif label == "UNCERTAIN":
                note = (
                    f"mcpgraphy: {name} from state [{pretty}] had mixed outcomes in prior "
                    f"calls ({len(evidence)} evidence); treat the result as uncertain."
                )
        if self.policy.mode != "block":
            return Gate(label, note, False)
        # block mode. Evidence first (the more useful reason), then the approval gates, which
        # are independent of what has been learned.
        if label == "NOT_SUPPORTED":
            return Gate(label, note, True, note)
        if label == "UNCERTAIN" and not self.policy.allow_uncertain:
            return Gate(label, note, True, note)
        needs_approval = (
            policy_tool.approval_required or policy_tool.effect is EffectClass.DESTRUCTIVE
        )
        if needs_approval and name not in self.policy.approved_tools:
            return Gate(
                label,
                note,
                True,
                f"{name} is destructive or requires approval and is not pre-approved "
                f"(start the proxy with --approve {name} to allow it)",
            )
        return Gate(label, note, False)

    async def _observe(self, rid: Any) -> Snapshot:
        assert self.remote is not None and self.spec is not None
        rec = await self.remote.call(self.spec.observation_tool, {self.spec.id_param: rid})
        return Snapshot(tool=self.spec.observation_tool, raw=rec.raw, outcome=rec.outcome)

    def _record(
        self,
        name: str,
        args: dict[str, Any],
        before: Snapshot | None,
        after: Snapshot | None,
        result: types.CallToolResult,
    ) -> None:
        assert self.journal is not None and self.spec is not None and self.sv is not None
        if before is None:
            return
        rec = record_from_result(name, args, result)
        exists = before.raw is not None and before.outcome.kind.is_ok
        ac = action_class_for_call(self.spec, name, args, exists)
        bstate = AbstractState.build(
            self.spec.server_id,
            self.spec.resource_class,
            features_of(self.spec.resource_class, before),
        )
        astate = (
            AbstractState.build(
                self.spec.server_id,
                self.spec.resource_class,
                features_of(self.spec.resource_class, after),
            )
            if after is not None
            else None
        )
        self.seq += 1
        exp = Experiment(
            experiment_id=Experiment.make_id(
                self.run_id, self.sv.server_version_id, "live", ac.action_class_id, self.seq
            ),
            run_id=self.run_id,
            seq=self.seq,
            server_version_id=self.sv.server_version_id,
            fixture_id="live",
            action_class_id=ac.action_class_id,
            resource_ref=f"{self.spec.resource_class}:{id_value(self.spec, name, args)}",
            args=args,
            before=before,
            before_state_id=bstate.state_id,
            result=rec.outcome,
            raw_result=rec.raw,
            after=after,
            after_state_id=astate.state_id if astate else None,
            reset_confirmed=False,
            phase="proxy",
            created_at=self.seq,
        )
        self.journal.append(
            "experiment",
            {
                **exp.to_dict(),
                "before_features": bstate.features,
                "after_features": astate.features if astate else None,
                "action_class": ac.to_dict(),
            },
        )
        self.calls += 1
        if self.calls % self.refresh_every == 0:
            self._materialize()

    def _materialize(self) -> None:
        with GraphStore(self.db_path) as store:
            try:
                self.model_version = materialize(self.journal_path, store, run_id=self.run_id)
            except Exception:
                self.model_version = None

    # ---------------------------------------------------------------- our tools
    async def _ours(self, name: str, args: dict[str, Any]) -> types.CallToolResult:
        assert self.remote is not None and self.spec is not None
        if name == "status":
            return _ok(
                {
                    "server_id": self.spec.server_id,
                    "policy": self.policy.mode,
                    "approved_tools": sorted(self.policy.approved_tools),
                    "calls_recorded": self.calls,
                    "blocked": self.blocked,
                    "warned": self.warned,
                    "model_version": self.model_version,
                    "journal": str(self.journal_path),
                    "db": self.db_path,
                }
            )
        if name == "graph":
            if not self.model_version:
                return _error("nothing learned yet: no proxied calls have been recorded")
            fmt = args.get("format", "json")
            with GraphStore(self.db_path) as store:
                data = export_json(store, self.model_version)
            return _ok(
                {
                    "format": fmt,
                    "model_version": self.model_version,
                    "content": dumps(data) if fmt == "json" else export_mermaid(data),
                }
            )
        if name == "affordances":
            ref = str(args.get("resource_ref", ""))
            local_id = ref.split(":", 1)[1] if ":" in ref else ref
            resolved = await resolve_state(self.remote, self.spec, local_id)  # type: ignore[arg-type]
            if not self.model_version:
                return _ok(
                    {
                        "resource_ref": resolved.resource_ref.render(),
                        "exists": resolved.exists,
                        "observed_state": dict(resolved.state.features),
                        "model_version": None,
                        "offers": [],
                        "note": "nothing learned yet; every action is UNKNOWN",
                    }
                )
            with GraphStore(self.db_path) as store:
                offers = select_offers(
                    store, self.spec, self.model_version, resolved, args.get("goal")
                )
            for o in offers:
                self._offers[o.offer_id] = (o, resolved.state.state_id)
            return _ok(
                {
                    "resource_ref": resolved.resource_ref.render(),
                    "exists": resolved.exists,
                    "observed_state": dict(resolved.state.features),
                    "model_version": self.model_version,
                    "offers": [o.to_dict() for o in offers],
                }
            )
        if name == "explain":
            hit = self._offers.get(str(args.get("offer_id")))
            if hit is None:
                return _error("unknown offer id; call mcpgraphy_affordances first")
            offer, state_id = hit
            with GraphStore(self.db_path) as store:
                return _ok(
                    explain(store, self.spec, offer, state_id, False).model_dump(mode="json")
                )
        return _error(f"unknown mcpgraphy tool {name!r}")


async def serve_stdio(proxy: Proxy) -> None:
    from mcp.server.stdio import stdio_server

    async with proxy:
        async with stdio_server() as (read, write):
            await proxy.server.run(read, write, proxy.server.create_initialization_options())


def _ok(data: dict[str, Any]) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(data, indent=2, sort_keys=True))],
        structured_content=data,
    )


def _error(text: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=True)


def _max_seq(path: Path) -> int:
    events, _ = Journal.read(path)
    return max((e.get("seq", 0) for e in events if e.get("kind") == "experiment"), default=0)


__all__ = ["Proxy", "ProxyPolicy", "canonical_json", "serve_stdio"]
