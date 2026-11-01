"""The action-offer MCP server.

Tools: inspect_server, inspect_resource, list_affordances, explain_affordance, simulate_action,
execute_offered_action, report_counterexample, get_behavior_graph. `simulate_action` only reads
the model; the only downstream calls are the declared read-only observation tool and, when the
host policy enables it, `execute_offered_action` through the audited executor.
"""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

from mcpgraphy.client_adapter import InMemoryToolClient
from mcpgraphy.graph.export import dumps, export_json, export_mermaid
from mcpgraphy.graph.store import GraphStore
from mcpgraphy.models import Offer, ServerVersion
from mcpgraphy.offers.executor import (
    CounterexampleReport,
    ExecutionPolicy,
    ExecutionRecord,
    Executor,
)
from mcpgraphy.offers.explain import Explanation, explain
from mcpgraphy.offers.selector import ResolvedState, resolve_state, select_offers
from mcpgraphy.registry import get_spec
from mcpgraphy.sandbox.base import SandboxControl


class ServerInspection(BaseModel):
    server_id: str
    version: str
    fingerprint: str
    model_version: str | None
    model_fingerprint: str | None
    fingerprint_drift: bool
    tools: list[dict[str, Any]]


class ResourceInspection(BaseModel):
    server_id: str
    resource_ref: str
    exists: bool
    snapshot: dict[str, Any] | None
    observed_state: dict[str, Any]
    state_id: str
    read_outcome: str


class AffordanceList(BaseModel):
    server_id: str
    resource_ref: str
    exists: bool
    observed_state: dict[str, Any]
    model_version: str
    fingerprint_drift: bool
    offers: list[Offer]


class PredictedOutcome(BaseModel):
    outcome_kind: str
    target_state: dict[str, Any] | None
    evidence_count: int
    status: str


class Simulation(BaseModel):
    offer_id: str
    tool: str
    args: dict[str, Any]
    behavior_status: str
    availability: str
    predicted_outcomes: list[PredictedOutcome]
    executed: bool
    caveats: list[str]


class GraphExport(BaseModel):
    server_id: str
    model_version: str
    format: str
    content: str


class OfferServer:
    """State held by the MCP server: a store, sandboxes to observe, and an offer cache."""

    def __init__(
        self,
        db_path: str,
        sandboxes: dict[str, SandboxControl],
        model: str = "latest",
        policy: ExecutionPolicy | None = None,
    ) -> None:
        self.db_path = db_path
        self.sandboxes = sandboxes
        self.model = model
        self.policy = policy or ExecutionPolicy()
        self.executor = Executor(db_path, self.policy)
        self._offers: dict[
            str, tuple[Offer, str, bool]
        ] = {}  # offer_id -> (offer, state_id, drift)

    def _sandbox(self, server_id: str) -> SandboxControl:
        sb = self.sandboxes.get(server_id)
        if sb is None:
            raise ToolError(f"NOT_FOUND: unknown server {server_id!r}")
        return sb

    async def _fingerprint(self, server_id: str) -> str:
        sb = self._sandbox(server_id)
        async with InMemoryToolClient(sb.mcp) as client:
            tools = await client.list_tools()
        return ServerVersion.fingerprint_of(tools)

    def _model_version(self, store: GraphStore, server_id: str) -> str:
        try:
            return store.resolve_model_version(server_id, self.model)
        except LookupError as e:
            raise ToolError(f"NOT_FOUND: {e}") from None

    async def _resolve(self, server_id: str, resource_ref: str) -> ResolvedState:
        sb = self._sandbox(server_id)
        spec = get_spec(server_id, sb.version)
        local_id = resource_ref.split(":", 1)[1] if ":" in resource_ref else resource_ref
        async with InMemoryToolClient(sb.mcp) as client:
            return await resolve_state(client, spec, local_id)

    # ---- tool implementations ---------------------------------------------------------
    async def inspect_server(self, server_id: str) -> ServerInspection:
        sb = self._sandbox(server_id)
        spec = get_spec(server_id, sb.version)
        async with InMemoryToolClient(sb.mcp) as client:
            tools = await client.list_tools()
        fp = ServerVersion.fingerprint_of(tools)
        with GraphStore(self.db_path) as store:
            m = store.latest_model(server_id) if self.model == "latest" else store.model(self.model)
            msv = store.server_version(m.server_version_id) if m else None
        return ServerInspection(
            server_id=server_id,
            version=sb.version,
            fingerprint=fp,
            model_version=m.model_version if m else None,
            model_fingerprint=msv.fingerprint if msv else None,
            fingerprint_drift=bool(msv and msv.fingerprint != fp),
            tools=[
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                    "declared_effect_class": str(spec.tool(t.name).effect),
                    "declared_approval_required": spec.tool(t.name).approval_required,
                }
                for t in tools
            ],
        )

    async def inspect_resource(self, server_id: str, resource_ref: str) -> ResourceInspection:
        r = await self._resolve(server_id, resource_ref)
        return ResourceInspection(
            server_id=server_id,
            resource_ref=r.resource_ref.render(),
            exists=r.exists,
            snapshot=r.snapshot.raw,
            observed_state=dict(r.state.features),
            state_id=r.state.state_id,
            read_outcome=str(r.snapshot.outcome.kind),
        )

    async def list_affordances(
        self, server_id: str, resource_ref: str, goal: str | None = None
    ) -> AffordanceList:
        sb = self._sandbox(server_id)
        spec = get_spec(server_id, sb.version)
        r = await self._resolve(server_id, resource_ref)
        fp = await self._fingerprint(server_id)
        with GraphStore(self.db_path) as store:
            mv = self._model_version(store, server_id)
            msv = store.server_version(store.model(mv).server_version_id)
            drift = bool(msv and msv.fingerprint != fp)
            offers = select_offers(store, spec, mv, r, goal)
        for o in offers:
            self._offers[o.offer_id] = (o, r.state.state_id, drift)
        return AffordanceList(
            server_id=server_id,
            resource_ref=r.resource_ref.render(),
            exists=r.exists,
            observed_state=dict(r.state.features),
            model_version=mv,
            fingerprint_drift=drift,
            offers=offers,
        )

    def _cached(self, offer_id: str) -> tuple[Offer, str, bool]:
        hit = self._offers.get(offer_id)
        if hit is None:
            raise ToolError(f"NOT_FOUND: unknown offer {offer_id!r}; call list_affordances first")
        return hit

    async def explain_affordance(self, offer_id: str) -> Explanation:
        offer, state_id, drift = self._cached(offer_id)
        sb = self._sandbox(offer.server_id)
        spec = get_spec(offer.server_id, sb.version)
        with GraphStore(self.db_path) as store:
            return explain(store, spec, offer, state_id, drift)

    async def simulate_action(self, offer_id: str, args: dict[str, Any]) -> Simulation:
        offer, _state_id, drift = self._cached(offer_id)
        for k, v in offer.argument_template.items():
            if k not in args:
                raise ToolError(f"INVALID_ARGUMENT: missing argument {k!r}")
            if not (isinstance(v, str) and v.startswith("<")) and args[k] != v:
                raise ToolError(
                    f"INVALID_ARGUMENT: argument {k!r} must equal the offer template value {v!r}"
                )
        caveats = [
            "simulation only: no downstream tool was called and no state changed",
            "predictions describe sandbox-tested applicability, not guaranteed runtime success",
        ]
        if drift:
            caveats.append("server schema drifted since this model was learned")
        return Simulation(
            offer_id=offer_id,
            tool=offer.tool,
            args=args,
            behavior_status=str(offer.behavior.status),
            availability=str(offer.availability),
            predicted_outcomes=[
                PredictedOutcome(
                    outcome_kind=str(a.outcome_kind),
                    target_state=a.target_state,
                    evidence_count=a.evidence_count,
                    status=str(a.status),
                )
                for a in offer.uncertainty.alternative_outcomes
            ],
            executed=False,
            caveats=caveats,
        )

    async def execute_offered_action(
        self,
        offer_id: str,
        args: dict[str, Any],
        approval_token: str | None = None,
        idempotency_key: str | None = None,
    ) -> ExecutionRecord:
        offer, state_id, _drift = self._cached(offer_id)
        sb = self._sandbox(offer.server_id)
        spec = get_spec(offer.server_id, sb.version)
        async with InMemoryToolClient(sb.mcp) as client:
            return await self.executor.execute(
                client, spec, offer, state_id, args, approval_token, idempotency_key
            )

    async def report_counterexample(self, execution_id: str) -> CounterexampleReport:
        return self.executor.report_counterexample(execution_id)

    async def get_behavior_graph(self, server_id: str, format: str = "json") -> GraphExport:
        if format not in ("json", "mermaid"):
            raise ToolError("INVALID_ARGUMENT: format must be 'json' or 'mermaid'")
        with GraphStore(self.db_path) as store:
            mv = self._model_version(store, server_id)
            data = export_json(store, mv)
        content = dumps(data) if format == "json" else export_mermaid(data)
        return GraphExport(server_id=server_id, model_version=mv, format=format, content=content)


def build_offer_server(
    db_path: str,
    sandboxes: dict[str, SandboxControl],
    model: str = "latest",
    policy: ExecutionPolicy | None = None,
) -> MCPServer:
    state = OfferServer(db_path, sandboxes, model, policy)
    mcp = MCPServer(
        "mcpgraphy",
        instructions=(
            "Uncertainty-aware action offers learned from sandbox experiments. Offers describe "
            "tested applicability, never guaranteed success; UNKNOWN is neither valid nor "
            "invalid. Effectful execution is disabled unless the host operator enabled a policy "
            "at launch; unknown offers never execute."
        ),
    )

    @mcp.tool()
    async def inspect_server(server_id: str) -> ServerInspection:
        """Tool schemas, declared risk hints and whether the learned model's fingerprint matches."""
        return await state.inspect_server(server_id)

    @mcp.tool()
    async def inspect_resource(server_id: str, resource_ref: str) -> ResourceInspection:
        """Observe a resource (e.g. 'release:demo-0-0') with the read-only observation tool."""
        return await state.inspect_resource(server_id, resource_ref)

    @mcp.tool()
    async def list_affordances(
        server_id: str, resource_ref: str, goal: str | None = None
    ) -> AffordanceList:
        """Typed action offers for the resource's current observed state. Optional goal like
        'status=PUBLISHED' promotes offers predicted to reach it."""
        return await state.list_affordances(server_id, resource_ref, goal)

    @mcp.tool()
    async def explain_affordance(offer_id: str) -> Explanation:
        """Guard, evidence, counterexamples, untested regions and exclusions for an offer."""
        return await state.explain_affordance(offer_id)

    @mcp.tool()
    async def simulate_action(offer_id: str, args: dict[str, Any]) -> Simulation:
        """Predicted outcomes for an offer with concrete arguments. Changes no remote state."""
        return await state.simulate_action(offer_id, args)

    @mcp.tool()
    async def execute_offered_action(
        offer_id: str,
        args: dict[str, Any],
        approval_token: str | None = None,
        idempotency_key: str | None = None,
    ) -> ExecutionRecord:
        """Delegated execution of an offer behind the host policy: unknown offers never run,
        effectful tools need an enabling policy, approval-required tools need host approval,
        the resource is re-observed first, retries with the same idempotency key never repeat
        effects, and the result is read back and audited."""
        return await state.execute_offered_action(offer_id, args, approval_token, idempotency_key)

    @mcp.tool()
    async def report_counterexample(execution_id: str) -> CounterexampleReport:
        """Turn an execution whose read-back disagreed with the prediction into a counterexample
        (downgrades the transition). Matches never raise trust."""
        return await state.report_counterexample(execution_id)

    @mcp.tool()
    async def get_behavior_graph(server_id: str, format: str = "json") -> GraphExport:
        """Export the learned graph: 'json' (authoritative) or 'mermaid' (presentation)."""
        return await state.get_behavior_graph(server_id, format)

    return mcp
