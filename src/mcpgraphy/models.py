"""Core data model: typed, frozen, content-addressed.

Every identifier in the system is a content hash of a canonical JSON payload. Nothing here depends
on wall-clock time, memory addresses, or insertion order, so two runs with the same inputs produce
byte-identical ids and exports. `created_at` fields exist for audit only and are never hashed.

Spec references: §3 (graph design), §5.4 (offer shape), §8 (parameter classes), §11 (storage).
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------------
# Canonical encoding and ids
# --------------------------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, ASCII only."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_default
    )


def _default(o: Any) -> Any:
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not canonicalizable: {type(o).__name__}")


def content_id(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()
    return f"{prefix}_{digest[:16]}"


# --------------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------------


class BehaviorStatus(StrEnum):
    OBSERVED_SUPPORTED = "OBSERVED_SUPPORTED"
    CONDITIONAL = "CONDITIONAL"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"


class AuthStatus(StrEnum):
    NOT_REQUIRED = "NOT_REQUIRED"
    REQUIRED = "REQUIRED"
    GRANTED_FOR_THIS_ACTION = "GRANTED_FOR_THIS_ACTION"
    DENIED = "DENIED"
    UNKNOWN = "UNKNOWN"


class Availability(StrEnum):
    CANDIDATE = "CANDIDATE"
    CANDIDATE_REQUIRES_APPROVAL = "CANDIDATE_REQUIRES_APPROVAL"
    UNCERTAIN_CANDIDATE = "UNCERTAIN_CANDIDATE"
    NOT_SUPPORTED_FROM_THIS_STATE = "NOT_SUPPORTED_FROM_THIS_STATE"
    UNKNOWN_UNTESTED = "UNKNOWN_UNTESTED"
    NEEDS_OBSERVATION = "NEEDS_OBSERVATION"


class OutcomeKind(StrEnum):
    OK = "OK"
    ERR_NOT_FOUND = "ERR_NOT_FOUND"
    ERR_STATE = "ERR_STATE"
    ERR_PERMISSION = "ERR_PERMISSION"
    ERR_VALIDATION = "ERR_VALIDATION"
    ERR_OTHER = "ERR_OTHER"

    @property
    def is_ok(self) -> bool:
        return self is OutcomeKind.OK


class EffectClass(StrEnum):
    READ_ONLY = "READ_ONLY"
    MUTATING = "MUTATING"
    DESTRUCTIVE = "DESTRUCTIVE"


class Provenance(StrEnum):
    SANDBOX_PROBE = "SANDBOX_PROBE"
    HYPOTHESIZED_BY_GUARD = "HYPOTHESIZED_BY_GUARD"
    INHERITED_NEEDS_RETEST = "INHERITED_NEEDS_RETEST"
    DECLARED = "DECLARED"


class ObservedOutcome(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    MIXED = "MIXED"


# --------------------------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------------------------

FeatureValue = str | bool | int | None
Features = dict[str, FeatureValue]


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------------------------
# Server, resources, states, actions
# --------------------------------------------------------------------------------------------


class ToolSpec(Frozen):
    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] = Field(default_factory=dict)


class ServerVersion(Frozen):
    server_version_id: str
    server_id: str
    version: str
    fingerprint: str
    tools: tuple[ToolSpec, ...]

    @staticmethod
    def fingerprint_of(tools: list[ToolSpec] | tuple[ToolSpec, ...]) -> str:
        payload = sorted((t.name, canonical_json(t.input_schema)) for t in tools)
        return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()[:16]

    @classmethod
    def build(cls, server_id: str, version: str, tools: list[ToolSpec]) -> ServerVersion:
        fp = cls.fingerprint_of(tools)
        return cls(
            server_version_id=content_id(
                "sv", {"server_id": server_id, "version": version, "fingerprint": fp}
            ),
            server_id=server_id,
            version=version,
            fingerprint=fp,
            tools=tuple(sorted(tools, key=lambda t: t.name)),
        )


class ResourceRef(Frozen):
    """Tenant-scoped resource identity. `t1/demo-7` and `t2/demo-7` are different resources."""

    resource_ref_id: str
    server_id: str
    tenant: str
    resource_class: str
    local_id: str

    @classmethod
    def build(cls, server_id: str, tenant: str, resource_class: str, local_id: str) -> ResourceRef:
        payload = {
            "server_id": server_id,
            "tenant": tenant,
            "resource_class": resource_class,
            "local_id": local_id,
        }
        return cls(resource_ref_id=content_id("res", payload), **payload)

    def render(self) -> str:
        return f"{self.resource_class}:{self.local_id}"


class AbstractState(Frozen):
    state_id: str
    server_id: str
    resource_class: str
    features: Features

    @classmethod
    def build(cls, server_id: str, resource_class: str, features: Features) -> AbstractState:
        feats = dict(sorted(features.items()))
        payload = {"server_id": server_id, "resource_class": resource_class, "features": feats}
        return cls(
            state_id=content_id("st", payload),
            server_id=server_id,
            resource_class=resource_class,
            features=feats,
        )

    def label(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.features.items())


class ActionClass(Frozen):
    """A tool plus an abstraction of its arguments, e.g. publish_release(release_id=EXISTING)."""

    action_class_id: str
    server_id: str
    tool: str
    arg_classes: dict[str, str]
    effect_class: EffectClass

    @classmethod
    def build(
        cls, server_id: str, tool: str, arg_classes: dict[str, str], effect_class: EffectClass
    ) -> ActionClass:
        ac = dict(sorted(arg_classes.items()))
        payload = {"server_id": server_id, "tool": tool, "arg_classes": ac}
        return cls(
            action_class_id=content_id("ac", payload),
            server_id=server_id,
            tool=tool,
            arg_classes=ac,
            effect_class=effect_class,
        )

    @property
    def label(self) -> str:
        inner = ", ".join(f"{k}={v}" for k, v in self.arg_classes.items())
        return f"{self.tool}({inner})"


# --------------------------------------------------------------------------------------------
# Observations and experiments
# --------------------------------------------------------------------------------------------


class Outcome(Frozen):
    kind: OutcomeKind
    code: str | None = None
    message: str = ""


class Snapshot(Frozen):
    """A read-back of a resource via a read-only tool. `raw` is None when the read failed."""

    tool: str
    raw: dict[str, Any] | None
    outcome: Outcome


class Experiment(Frozen):
    experiment_id: str
    run_id: str
    seq: int
    server_version_id: str
    fixture_id: str
    action_class_id: str
    resource_ref: str
    args: dict[str, Any]
    before: Snapshot
    before_state_id: str
    result: Outcome
    raw_result: Any = None
    after: Snapshot | None = None
    after_state_id: str | None = None
    reset_confirmed: bool = False
    phase: str = "probe"
    created_at: int = 0

    @staticmethod
    def make_id(
        run_id: str, server_version_id: str, fixture_id: str, action_class_id: str, seq: int
    ) -> str:
        return content_id(
            "exp",
            {
                "run_id": run_id,
                "server_version_id": server_version_id,
                "fixture_id": fixture_id,
                "action_class_id": action_class_id,
                "seq": seq,
            },
        )


# --------------------------------------------------------------------------------------------
# Guards, transitions, counterexamples, models
# --------------------------------------------------------------------------------------------


class Predicate(Frozen):
    feature: str
    op: str = "=="
    value: FeatureValue = None

    def holds(self, features: Features) -> bool:
        actual = features.get(self.feature)
        return (actual == self.value) if self.op == "==" else (actual != self.value)

    def __str__(self) -> str:
        return f"{self.feature} {self.op} {_fmt(self.value)}"


def _fmt(v: FeatureValue) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    return str(v)


class Guard(Frozen):
    all_of: tuple[Predicate, ...] = ()
    coverage_tested: int = 0
    coverage_total: int = 0
    note: str | None = None

    def holds(self, features: Features) -> bool:
        return all(p.holds(features) for p in self.all_of)

    def render(self) -> list[str]:
        return [str(p) for p in self.all_of]


class Transition(Frozen):
    transition_id: str
    model_version: str
    source_state_id: str
    action_class_id: str
    outcome_kind: OutcomeKind
    target_state_id: str | None
    status: BehaviorStatus
    guard: Guard | None = None
    provenance: Provenance = Provenance.SANDBOX_PROBE
    evidence_ids: tuple[str, ...] = ()
    counterexample_ids: tuple[str, ...] = ()
    predicted_effects: tuple[str, ...] = ()
    retest_required: bool = False
    created_at: int = 0

    @staticmethod
    def make_id(
        model_version: str,
        source_state_id: str,
        action_class_id: str,
        outcome_kind: OutcomeKind,
        target_state_id: str | None,
    ) -> str:
        return content_id(
            "tr",
            {
                "model_version": model_version,
                "source": source_state_id,
                "action": action_class_id,
                "outcome": str(outcome_kind),
                "target": target_state_id,
            },
        )

    @property
    def evidence_count(self) -> int:
        return len(self.evidence_ids)


class Counterexample(Frozen):
    counterexample_id: str
    model_version: str
    transition_id: str
    experiment_id: str
    predicted: dict[str, Any]
    observed: dict[str, Any]
    created_at: int = 0

    @staticmethod
    def make_id(model_version: str, transition_id: str, experiment_id: str) -> str:
        return content_id(
            "cx",
            {
                "model_version": model_version,
                "transition_id": transition_id,
                "experiment_id": experiment_id,
            },
        )


class ModelRecord(Frozen):
    model_version: str
    server_id: str
    server_version_id: str
    run_id: str
    seed: int
    budget: dict[str, Any] = Field(default_factory=dict)
    coverage: dict[str, Any] = Field(default_factory=dict)
    parent_model_version: str | None = None
    created_at: int = 0


def make_run_id(
    server_id: str,
    version: str,
    seed: int,
    fixtures: int,
    budget: int,
    learner_config: dict[str, Any] | None = None,
) -> str:
    return content_id(
        "run",
        {
            "server_id": server_id,
            "version": version,
            "seed": seed,
            "fixtures": fixtures,
            "budget": budget,
            "learner_config": learner_config or {},
        },
    )


def make_model_version(server_id: str, version: str, run_id: str) -> str:
    return f"graph_{server_id}_{version}_{run_id}"


# --------------------------------------------------------------------------------------------
# Offers
# --------------------------------------------------------------------------------------------


class OfferBehavior(Frozen):
    status: BehaviorStatus
    observed_state: Features
    required_predicates: tuple[str, ...]
    predicted_effects: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    model_version: str
    observed_outcome: ObservedOutcome | None = None  # additive


class OfferAuthorization(Frozen):
    required: bool
    granted: bool
    status: AuthStatus  # additive


class AlternativeOutcome(Frozen):
    outcome_kind: OutcomeKind
    target_state: Features | None
    evidence_count: int
    status: BehaviorStatus


class OfferUncertainty(Frozen):
    untested_parameter_regions: tuple[str, ...]
    possible_unobserved_effects: bool = True
    alternative_outcomes: tuple[AlternativeOutcome, ...] = ()  # additive


class Offer(Frozen):
    offer_id: str
    server_id: str
    resource_ref: str
    tool: str
    argument_template: dict[str, Any]
    behavior: OfferBehavior
    authorization: OfferAuthorization
    availability: Availability
    uncertainty: OfferUncertainty

    @staticmethod
    def make_id(
        model_version: str, resource_ref_id: str, action_class_id: str, observed_state_id: str
    ) -> str:
        return content_id(
            "offer",
            {
                "model_version": model_version,
                "resource": resource_ref_id,
                "action": action_class_id,
                "state": observed_state_id,
            },
        )
