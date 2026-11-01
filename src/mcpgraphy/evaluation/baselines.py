"""Guidance sources B0, B2, B3, B4, B5 and ablations.

B1 (progressive tool discovery) is not implemented: with 6 tools per server it would be
identical to B0 on this suite, so claiming a comparison would be hollow.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mcpgraphy.client_adapter import InMemoryToolClient
from mcpgraphy.evaluation.agents import Action
from mcpgraphy.graph.store import GraphStore
from mcpgraphy.learn.abstract import features_of_raw
from mcpgraphy.models import AbstractState, ActionClass, Availability, BehaviorStatus, Features
from mcpgraphy.offers.selector import classify_rows
from mcpgraphy.registry import ServerSpec, classify_value
from mcpgraphy.sandbox.base import FixtureManifest

AVAIL_TO_LABEL = {
    Availability.CANDIDATE: "SUPPORTED",
    Availability.CANDIDATE_REQUIRES_APPROVAL: "SUPPORTED",
    Availability.UNCERTAIN_CANDIDATE: "UNCERTAIN",
    Availability.UNKNOWN_UNTESTED: "UNKNOWN",
    Availability.NOT_SUPPORTED_FROM_THIS_STATE: "NOT_SUPPORTED",
    Availability.NEEDS_OBSERVATION: "UNKNOWN",
}


class RawToolsGuidance:
    """B0: ordinary MCP. Tool descriptions only; every action is UNKNOWN; nothing is suppressed."""

    name = "B0"

    async def rank(self, client, manifest, features, prior: list[Action]) -> list[str]:
        return ["UNKNOWN" for _ in prior]


class OracleGuidance:
    """B2: handwritten ground truth with full hidden state. An upper-bound reference."""

    name = "B2"

    def __init__(
        self,
        truth: Callable[[str, dict[str, Any]], bool],
        changes_state: Callable[[str, dict[str, Any]], bool] | None = None,
    ) -> None:
        self._truth = truth
        self._changes = changes_state

    async def rank(self, client, manifest, features, prior: list[Action]) -> list[str]:
        labels = []
        for a in prior:
            if not self._truth(a.tool, a.args):
                labels.append("NOT_SUPPORTED")
            elif self._changes is not None and not self._changes(a.tool, a.args):
                labels.append("NOOP")
            else:
                labels.append("SUPPORTED")
        return labels


class LearnedGuidance:
    """B5: offers from the learned graph (and B4 when the model came from passive traces).

    Ablations: `unknown_as_valid` labels UNKNOWN as SUPPORTED; `no_guards` ignores hypothesized
    (CONDITIONAL) rows so they count as UNKNOWN."""

    def __init__(
        self,
        name: str,
        spec: ServerSpec,
        store: GraphStore,
        model_version: str,
        unknown_as_valid: bool = False,
        no_guards: bool = False,
    ) -> None:
        self.name = name
        self.spec = spec
        self.store = store
        self.model_version = model_version
        self.unknown_as_valid = unknown_as_valid
        self.no_guards = no_guards

    async def rank(
        self,
        client: InMemoryToolClient,
        manifest: FixtureManifest,
        features: Features,
        prior: list[Action],
    ) -> list[str]:
        labels = []
        cache: dict[str, Features | None] = {}
        for a in prior:
            target_id = a.args.get(self.spec.id_param)
            if target_id is None or target_id == manifest.focal_ref.local_id:
                feats: Features | None = features
            else:
                if target_id not in cache:
                    rec = await client.call(
                        self.spec.observation_tool, {self.spec.id_param: target_id}
                    )
                    cache[target_id] = (
                        features_of_raw(self.spec.resource_class, rec.raw) if rec.raw else None
                    )
                feats = cache[target_id]
            if feats is None:
                labels.append("NOT_SUPPORTED")  # the resource does not exist: nothing to act on
                continue
            ac = self._action_class(manifest, a)
            state = AbstractState.build(self.spec.server_id, self.spec.resource_class, feats)
            rows = self.store.transitions_for(
                self.model_version, state.state_id, ac.action_class_id
            )
            if self.no_guards:
                rows = [
                    r
                    for r in rows
                    if r.status is not BehaviorStatus.CONDITIONAL
                    or r.provenance != "HYPOTHESIZED_BY_GUARD"
                ]
            approval = self.spec.tool(a.tool).approval_required
            _status, _obs, availability, _req, effects, *_ = classify_rows(
                rows, self.store, approval
            )
            label = AVAIL_TO_LABEL[availability]
            if label == "SUPPORTED" and tuple(effects) == ("no state change",):
                label = "NOOP"
            if self.unknown_as_valid and label == "UNKNOWN":
                label = "SUPPORTED"
            labels.append(label)
        return labels

    def _action_class(self, manifest: FixtureManifest, a: Action) -> ActionClass:
        tool = self.spec.tool(a.tool)
        classes = {
            p.name: classify_value(manifest, self.spec, a.tool, p.name, a.args.get(p.name))
            for p in tool.params
        }
        return ActionClass.build(self.spec.server_id, a.tool, classes, tool.effect)


class TraceGuidance:
    """B3 fallback and the Mealy-state view: label from (observed features, action class) ->
    set of outcomes seen in traces. SUPPORTED if only OK was seen, NOT_SUPPORTED if only errors,
    UNCERTAIN if both, UNKNOWN if never seen."""

    def __init__(self, name: str, spec: ServerSpec, table: dict[tuple[str, str], set[str]]) -> None:
        self.name = name
        self.spec = spec
        self.table = table

    async def rank(self, client, manifest, features, prior: list[Action]) -> list[str]:
        labels = []
        fkey = _fkey(features)
        for a in prior:
            tool = self.spec.tool(a.tool)
            classes = {
                p.name: classify_value(manifest, self.spec, a.tool, p.name, a.args.get(p.name))
                for p in tool.params
            }
            ac = ActionClass.build(self.spec.server_id, a.tool, classes, tool.effect)
            seen = self.table.get((fkey, ac.action_class_id))
            if not seen:
                labels.append("UNKNOWN")
            elif seen == {"OK"}:
                labels.append("SUPPORTED")
            elif "OK" in seen:
                labels.append("UNCERTAIN")
            else:
                labels.append("NOT_SUPPORTED")
        return labels


def _fkey(features: Features) -> str:
    return str(sorted(features.items()))
