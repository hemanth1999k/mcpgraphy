"""Affordance selection: live snapshot + learned model -> uncertainty-aware offers.

Behavior and authorization are kept apart on purpose. `authorization` comes only from the
registry's declared `approval_required` flag; `behavior` comes only from evidence rows. Nothing
here executes a tool: the only call made is the declared read-only observation tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mcpgraphy.client_adapter import InMemoryToolClient
from mcpgraphy.graph.store import GraphStore
from mcpgraphy.learn.abstract import features_of
from mcpgraphy.models import (
    AbstractState,
    ActionClass,
    AlternativeOutcome,
    AuthStatus,
    Availability,
    BehaviorStatus,
    EffectClass,
    Features,
    ObservedOutcome,
    Offer,
    OfferAuthorization,
    OfferBehavior,
    OfferUncertainty,
    OutcomeKind,
    Provenance,
    ResourceRef,
    Snapshot,
    Transition,
)
from mcpgraphy.registry import CONSTANT_VALUES, ServerSpec

AVAILABILITY_RANK = {
    Availability.CANDIDATE: 0,
    Availability.CANDIDATE_REQUIRES_APPROVAL: 1,
    Availability.UNCERTAIN_CANDIDATE: 2,
    Availability.UNKNOWN_UNTESTED: 3,
    Availability.NOT_SUPPORTED_FROM_THIS_STATE: 4,
    Availability.NEEDS_OBSERVATION: 5,
}


@dataclass
class ResolvedState:
    resource_ref: ResourceRef
    snapshot: Snapshot
    state: AbstractState
    exists: bool


async def resolve_state(
    client: InMemoryToolClient, spec: ServerSpec, local_id: str
) -> ResolvedState:
    rec = await client.call(spec.observation_tool, {spec.id_param: local_id})
    snap = Snapshot(tool=spec.observation_tool, raw=rec.raw, outcome=rec.outcome)
    feats = features_of(spec.resource_class, snap)
    state = AbstractState.build(spec.server_id, spec.resource_class, feats)
    ref = ResourceRef.build(spec.server_id, spec.tenant, spec.resource_class, local_id)
    return ResolvedState(ref, snap, state, exists=rec.outcome.kind.is_ok)


def template_action_classes(spec: ServerSpec, exists: bool = True) -> list[ActionClass]:
    """One offer template per tool and per state-dependent (core) argument combination. The
    identifier parameter is EXISTING for a live resource and NONEXISTENT when the read-back said
    the resource is missing (so evidence about acting on missing resources is used); other
    parameters take their core classes."""
    wanted = "EXISTING" if exists else "NONEXISTENT"
    out: list[ActionClass] = []
    for ac in spec.action_classes():
        id_classes = [p.name for p in spec.tool(ac.tool).params if "EXISTING" in p.classes]
        if any(ac.arg_classes.get(name) != wanted for name in id_classes):
            continue
        others_core = all(
            ac.arg_classes[p.name] in p.core
            for p in spec.tool(ac.tool).params
            if p.name not in id_classes
        )
        if others_core:
            out.append(ac)
    return out


def select_offers(
    store: GraphStore,
    spec: ServerSpec,
    model_version: str,
    resolved: ResolvedState,
    goal: str | None = None,
) -> list[Offer]:
    offers: list[Offer] = []
    acs_known = store.action_classes(spec.server_id)
    for ac in template_action_classes(spec, exists=resolved.exists):
        policy = spec.tool(ac.tool)
        template = _argument_template(spec, ac, resolved.resource_ref.local_id)
        if not resolved.exists and resolved.snapshot.outcome.kind is not OutcomeKind.ERR_NOT_FOUND:
            offers.append(
                _offer(
                    spec,
                    model_version,
                    resolved,
                    ac,
                    template,
                    status=BehaviorStatus.UNKNOWN,
                    observed=None,
                    required=(),
                    effects=(),
                    evidence=(),
                    availability=Availability.NEEDS_OBSERVATION,
                    untested=_untested_regions(store, spec, model_version, resolved, ac, acs_known),
                    alternatives=(),
                    approval=policy.approval_required,
                )
            )
            continue
        rows = store.transitions_for(model_version, resolved.state.state_id, ac.action_class_id)
        status, observed, availability, required, effects, evidence, alternatives = classify_rows(
            rows, store, policy.approval_required
        )
        offers.append(
            _offer(
                spec,
                model_version,
                resolved,
                ac,
                template,
                status=status,
                observed=observed,
                required=required,
                effects=effects,
                evidence=evidence,
                availability=availability,
                untested=_untested_regions(store, spec, model_version, resolved, ac, acs_known),
                alternatives=alternatives,
                approval=policy.approval_required,
            )
        )
    goal_feats = _parse_goal(goal)
    offers.sort(key=lambda o: _rank_key(o, goal_feats))
    return offers


# ---------------------------------------------------------------------------------------------


def classify_rows(rows: list[Transition], store: GraphStore, approval: bool):
    ok_rows = [r for r in rows if r.outcome_kind.is_ok]
    err_rows = [r for r in rows if not r.outcome_kind.is_ok]
    evidence = tuple(sorted({e for r in rows for e in r.evidence_ids}))
    alternatives = tuple(
        AlternativeOutcome(
            outcome_kind=r.outcome_kind,
            target_state=(
                store.state(r.target_state_id).features
                if r.target_state_id and store.state(r.target_state_id)
                else None
            ),
            evidence_count=len(r.evidence_ids),
            status=r.status,
        )
        for r in sorted(rows, key=lambda r: (-len(r.evidence_ids), str(r.outcome_kind)))
    )
    if not rows:
        return (BehaviorStatus.UNKNOWN, None, Availability.UNKNOWN_UNTESTED, (), (), (), ())
    if all(r.status is BehaviorStatus.UNKNOWN for r in rows):
        # inherited from an older server version and not yet retested
        return (
            BehaviorStatus.UNKNOWN,
            None,
            Availability.UNKNOWN_UNTESTED,
            ("inherited from an older model: needs retest",),
            (),
            evidence,
            alternatives,
        )
    mixed = (ok_rows and err_rows) or any(r.status is BehaviorStatus.CONTRADICTED for r in rows)
    if mixed:
        guard = next((r.guard for r in ok_rows if r.guard), None)
        required = tuple(guard.render()) if guard else ()
        return (
            BehaviorStatus.CONTRADICTED,
            ObservedOutcome.MIXED,
            Availability.UNCERTAIN_CANDIDATE,
            required,
            tuple(ok_rows[0].predicted_effects) if ok_rows else (),
            evidence,
            alternatives,
        )
    if ok_rows:
        row = ok_rows[0]
        required = tuple(row.guard.render()) if row.guard else ()
        if row.status is BehaviorStatus.OBSERVED_SUPPORTED:
            availability = (
                Availability.CANDIDATE_REQUIRES_APPROVAL if approval else Availability.CANDIDATE
            )
            return (
                BehaviorStatus.OBSERVED_SUPPORTED,
                ObservedOutcome.SUCCESS,
                availability,
                required,
                tuple(row.predicted_effects),
                evidence,
                alternatives,
            )
        return (
            BehaviorStatus.CONDITIONAL,
            None,
            Availability.UNCERTAIN_CANDIDATE,
            required or ("hypothesized from guard; not probed at this state",),
            tuple(row.predicted_effects),
            evidence,
            alternatives,
        )
    # failures only
    kinds = sorted({str(r.outcome_kind) for r in err_rows})
    return (
        BehaviorStatus.OBSERVED_SUPPORTED,
        ObservedOutcome.FAILURE,
        Availability.NOT_SUPPORTED_FROM_THIS_STATE,
        (),
        tuple(f"call fails with {k}" for k in kinds),
        evidence,
        alternatives,
    )


def _offer(
    spec,
    model_version,
    resolved,
    ac,
    template,
    *,
    status,
    observed,
    required,
    effects,
    evidence,
    availability,
    untested,
    alternatives,
    approval,
) -> Offer:
    return Offer(
        offer_id=Offer.make_id(
            model_version,
            resolved.resource_ref.resource_ref_id,
            ac.action_class_id,
            resolved.state.state_id,
        ),
        server_id=spec.server_id,
        resource_ref=resolved.resource_ref.render(),
        tool=ac.tool,
        argument_template=template,
        behavior=OfferBehavior(
            status=status,
            observed_state=dict(resolved.state.features),
            required_predicates=required,
            predicted_effects=effects,
            evidence_ids=evidence,
            model_version=model_version,
            observed_outcome=observed,
        ),
        authorization=OfferAuthorization(
            required=approval,
            granted=False,
            status=AuthStatus.REQUIRED if approval else AuthStatus.NOT_REQUIRED,
        ),
        availability=availability,
        uncertainty=OfferUncertainty(
            untested_parameter_regions=untested,
            possible_unobserved_effects=True,
            alternative_outcomes=alternatives,
        ),
    )


def _argument_template(spec: ServerSpec, ac: ActionClass, local_id: str) -> dict:
    args: dict = {}
    for param, cls in ac.arg_classes.items():
        if param == spec.id_param:
            args[param] = local_id
        elif cls in CONSTANT_VALUES:
            args[param] = CONSTANT_VALUES[cls]
        else:
            args[param] = f"<{cls}>"
    return args


def _untested_regions(store, spec, model_version, resolved, ac, acs_known) -> tuple[str, ...]:
    """Argument classes of this tool never probed from the current abstract state."""
    regions = []
    for other_id, other in sorted(acs_known.items()):
        if other.tool != ac.tool or other_id == ac.action_class_id:
            continue
        if not store.transitions_for(model_version, resolved.state.state_id, other_id):
            regions.append(other.label)
    return tuple(regions)


def _parse_goal(goal: str | None) -> Features | None:
    if not goal:
        return None
    m = re.fullmatch(r"\s*([A-Za-z_][\w]*)\s*=\s*(\S+)\s*", goal)
    if not m:
        return None
    key, raw = m.group(1), m.group(2)
    value: object = {"true": True, "false": False}.get(raw.lower(), raw)
    return {key: value}  # type: ignore[dict-item]


def _rank_key(o: Offer, goal: Features | None):
    goal_hit = 0
    if goal:
        wanted = {f"{k} becomes {_fmt(v)}" for k, v in goal.items()}
        goal_hit = 0 if wanted & set(o.behavior.predicted_effects) else 1
    return (
        AVAILABILITY_RANK[o.availability],
        goal_hit,
        -len(o.behavior.evidence_ids),
        o.tool,
        str(sorted(o.argument_template.items())),
    )


def _fmt(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


__all__ = [
    "Provenance",
    "ResolvedState",
    "resolve_state",
    "select_offers",
    "template_action_classes",
]


def inline_hint(offers: list[Offer], spec: ServerSpec, state: Features | None = None) -> str:
    """A few lines for an agent, straight from the graph: what this state supports.

    Appended to read-backs (and refusals) by the proxy so the agent learns the state's
    affordances without spending a turn on `mcpgraphy_affordances`. Each entry names the
    action with its non-id arguments and, when known, what it changes. Read-only tools are
    left out."""
    groups: dict[str, list[str]] = {
        "ok": [],
        "needs approval": [],
        "uncertain": [],
        "not supported": [],
        "untested": [],
    }
    for o in offers:
        if spec.tool(o.tool).effect is EffectClass.READ_ONLY:
            continue
        args = ", ".join(f"{k}={v}" for k, v in o.argument_template.items() if k != spec.id_param)
        label = f"{o.tool}({args})" if args else o.tool
        effects = [e for e in o.behavior.predicted_effects if not e.startswith("call fails")]
        av = o.availability
        if av is Availability.CANDIDATE:
            key = "ok"
        elif av is Availability.CANDIDATE_REQUIRES_APPROVAL:
            key = "needs approval"
        elif av is Availability.UNCERTAIN_CANDIDATE:
            key = "uncertain"
        elif av is Availability.NOT_SUPPORTED_FROM_THIS_STATE:
            key = "not supported"
        else:
            key = "untested"
        if key in ("ok", "needs approval", "uncertain") and effects:
            label += " -> " + "; ".join(effects)
        groups[key].append(label)
    head = "mcpgraphy, from this state"
    if state:
        head += " [" + ", ".join(f"{k}={v}" for k, v in sorted(state.items())) + "]"
    lines = [f"  {k}: {'; '.join(sorted(set(v)))}" for k, v in groups.items() if v]
    return head + ":\n" + "\n".join(lines)
