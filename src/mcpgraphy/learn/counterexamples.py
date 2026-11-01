"""Counterexamples across model versions.

Within one run, counterexamples are produced by `guards.induce` (mixed outcomes at one abstract
state). Across runs, `inherit_for_retest` carries a parent model's transitions into a new model
as UNKNOWN rows flagged `retest_required`; rows whose action class no longer exists (schema
drift) are dropped; rows the new run re-probed are compared and, when the new evidence
disagrees, a counterexample links the old transition to the new experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mcpgraphy.graph.store import GraphStore
from mcpgraphy.models import (
    AbstractState,
    ActionClass,
    BehaviorStatus,
    Counterexample,
    Provenance,
    ServerVersion,
    Transition,
)


@dataclass
class Inherited:
    transitions: list[Transition] = field(default_factory=list)
    counterexamples: list[Counterexample] = field(default_factory=list)
    dropped_for_schema_drift: int = 0


def inherit_for_retest(
    store: GraphStore,
    parent_model_version: str,
    model_version: str,
    sv: ServerVersion,
    states: dict[str, AbstractState],
    action_classes: dict[str, ActionClass],
    new_transitions: list[Transition],
) -> Inherited:
    out = Inherited()
    parent = store.model(parent_model_version)
    if parent is None:
        return out
    parent_sv = store.server_version(parent.server_version_id)
    changed_tools = _changed_tools(parent_sv, sv)
    probed = {(t.source_state_id, t.action_class_id): t for t in new_transitions}
    probed_pairs = {(t.source_state_id, t.action_class_id) for t in new_transitions}
    existing_ids = {t.transition_id for t in new_transitions}
    for old in store.transitions(parent_model_version):
        ac = action_classes.get(old.action_class_id) or store.action_classes(parent.server_id).get(
            old.action_class_id
        )
        if ac is None or ac.tool in changed_tools:
            out.dropped_for_schema_drift += 1
            continue
        key = (old.source_state_id, old.action_class_id)
        if key in probed_pairs:
            # the new run re-tested this pair: disagreement becomes a counterexample on the old row
            new = probed[key]
            agreeing = [
                t
                for t in new_transitions
                if (t.source_state_id, t.action_class_id) == key
                and t.outcome_kind == old.outcome_kind
            ]
            if (
                old.status in (BehaviorStatus.OBSERVED_SUPPORTED, BehaviorStatus.CONDITIONAL)
                and not agreeing
            ):
                tid = Transition.make_id(
                    model_version,
                    old.source_state_id,
                    old.action_class_id,
                    old.outcome_kind,
                    old.target_state_id,
                )
                if tid in existing_ids:
                    continue
                cxs = []
                for eid in new.evidence_ids[:1]:
                    cx = Counterexample(
                        counterexample_id=Counterexample.make_id(model_version, tid, eid),
                        model_version=model_version,
                        transition_id=tid,
                        experiment_id=eid,
                        predicted={
                            "outcome": str(old.outcome_kind),
                            "target_state_id": old.target_state_id,
                            "from_model": parent_model_version,
                        },
                        observed={
                            "outcome": str(new.outcome_kind),
                            "target_state_id": new.target_state_id,
                        },
                    )
                    out.counterexamples.append(cx)
                    cxs.append(cx.counterexample_id)
                out.transitions.append(
                    old.model_copy(
                        update={
                            "transition_id": tid,
                            "model_version": model_version,
                            "status": BehaviorStatus.CONTRADICTED,
                            "provenance": Provenance.INHERITED_NEEDS_RETEST,
                            "counterexample_ids": tuple(cxs),
                            "retest_required": False,
                        }
                    )
                )
            continue
        inherited_id = Transition.make_id(
            model_version,
            old.source_state_id,
            old.action_class_id,
            old.outcome_kind,
            old.target_state_id,
        )
        if inherited_id in existing_ids:
            continue
        existing_ids.add(inherited_id)
        out.transitions.append(
            old.model_copy(
                update={
                    "transition_id": inherited_id,
                    "model_version": model_version,
                    "status": BehaviorStatus.UNKNOWN,
                    "provenance": Provenance.INHERITED_NEEDS_RETEST,
                    "retest_required": True,
                    "counterexample_ids": (),
                }
            )
        )
        if old.source_state_id not in states:
            s = store.state(old.source_state_id)
            if s is not None:
                states[s.state_id] = s
        if old.target_state_id and old.target_state_id not in states:
            s = store.state(old.target_state_id)
            if s is not None:
                states[s.state_id] = s
    return out


def _changed_tools(parent_sv: ServerVersion | None, sv: ServerVersion) -> set[str]:
    if parent_sv is None:
        return set()
    old = {t.name: t.input_schema for t in parent_sv.tools}
    new = {t.name: t.input_schema for t in sv.tools}
    return {n for n in old if n not in new or old[n] != new[n]} | {n for n in new if n not in old}
