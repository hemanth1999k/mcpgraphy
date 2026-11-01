"""Why is this offer shown, and why not more confidently?"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from mcpgraphy.graph.store import GraphStore
from mcpgraphy.models import ActionClass, Offer
from mcpgraphy.registry import ServerSpec


class ExperimentSummary(BaseModel):
    experiment_id: str
    fixture_id: str
    phase: str
    args: dict[str, Any]
    before_state: dict[str, Any]
    outcome: str
    after_state: dict[str, Any] | None
    reset_confirmed: bool


class CounterexampleSummary(BaseModel):
    counterexample_id: str
    transition_id: str
    experiment_id: str
    predicted: dict[str, Any]
    observed: dict[str, Any]


class Explanation(BaseModel):
    offer_id: str
    tool: str
    behavior_status: str
    availability: str
    observed_state: dict[str, Any]
    guard: dict[str, Any] | None
    supporting_experiments: list[ExperimentSummary]
    counterexamples: list[CounterexampleSummary]
    untested_parameter_regions: list[str]
    exclusions: list[str]
    authorization: dict[str, Any]
    model_version: str
    fingerprint_drift: bool
    caveats: list[str]


def explain(
    store: GraphStore,
    spec: ServerSpec,
    offer: Offer,
    state_id: str,
    fingerprint_drift: bool,
) -> Explanation:
    ac = action_class_for(store, spec, offer)
    rows = (
        store.transitions_for(offer.behavior.model_version, state_id, ac.action_class_id)
        if ac
        else []
    )
    guard = None
    for r in rows:
        if r.guard is not None:
            g = r.guard
            guard = {
                "predicates": g.render(),
                "coverage_tested": g.coverage_tested,
                "coverage_total": g.coverage_total,
                "note": g.note,
            }
            break
    supporting = []
    for eid in offer.behavior.evidence_ids:
        e = store.experiment(eid)
        if e is None:
            continue
        before = store.state(e.before_state_id)
        after = store.state(e.after_state_id) if e.after_state_id else None
        supporting.append(
            ExperimentSummary(
                experiment_id=e.experiment_id,
                fixture_id=e.fixture_id,
                phase=e.phase,
                args=e.args,
                before_state=before.features if before else {},
                outcome=str(e.result.kind),
                after_state=after.features if after else None,
                reset_confirmed=e.reset_confirmed,
            )
        )
    cx_ids = {c for r in rows for c in r.counterexample_ids}
    cxs = [
        CounterexampleSummary(
            counterexample_id=c.counterexample_id,
            transition_id=c.transition_id,
            experiment_id=c.experiment_id,
            predicted=c.predicted,
            observed=c.observed,
        )
        for c in store.counterexamples(offer.behavior.model_version)
        if c.counterexample_id in cx_ids
    ]
    exclusions = []
    for r in rows:
        if r.outcome_kind.is_ok:
            continue
        exclusions.append(
            f"{r.outcome_kind} observed {len(r.evidence_ids)}x from this state ({r.status})"
        )
    caveats = [
        "read-back verification cannot prove the absence of other side effects",
        "tested applicability in a sandbox is not a guarantee of runtime success",
    ]
    if fingerprint_drift:
        caveats.append("server tool schema differs from the one this model was learned on")
    if offer.behavior.status == "CONTRADICTED":
        caveats.append(
            "identical observed features produced different outcomes: a hidden "
            "condition is likely; treat as uncertain"
        )
    return Explanation(
        offer_id=offer.offer_id,
        tool=offer.tool,
        behavior_status=str(offer.behavior.status),
        availability=str(offer.availability),
        observed_state=dict(offer.behavior.observed_state),
        guard=guard,
        supporting_experiments=supporting,
        counterexamples=cxs,
        untested_parameter_regions=list(offer.uncertainty.untested_parameter_regions),
        exclusions=exclusions,
        authorization=offer.authorization.to_dict(),
        model_version=offer.behavior.model_version,
        fingerprint_drift=fingerprint_drift,
        caveats=caveats,
    )


def action_class_for(store: GraphStore, spec: ServerSpec, offer: Offer) -> ActionClass | None:
    classes = {}
    tool = spec.tool(offer.tool)
    for p in tool.params:
        if p.name == spec.id_param:
            classes[p.name] = "EXISTING"
        else:
            v = offer.argument_template.get(p.name)
            cls = next((c for c in p.classes if _matches(c, v)), p.core[0])
            classes[p.name] = cls
    return ActionClass.build(spec.server_id, offer.tool, classes, tool.effect)


def _matches(cls: str, value: object) -> bool:
    from mcpgraphy.registry import CONSTANT_VALUES

    if isinstance(value, str) and value == f"<{cls}>":
        return True
    return cls in CONSTANT_VALUES and CONSTANT_VALUES[cls] == value
