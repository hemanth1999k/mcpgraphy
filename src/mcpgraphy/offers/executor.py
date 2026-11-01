"""Controlled delegated execution.

Every gate below runs *before* any downstream call, in this order:

  1. the offer must come from this server's last `list_affordances` and the arguments must
     match its template;
  2. a repeated request with the same idempotency key is answered from the audit record and is
     never re-executed (retries do not duplicate effects);
  3. UNKNOWN / untested / needs-observation offers never execute a write;
  4. offers observed to fail from this state never execute;
  5. uncertain (mixed / hypothesized) offers execute only if the host policy allows it;
  6. effectful tools execute only under a host policy that allows effectful execution;
  7. approval-required offers execute only with host approval: the tool is on the host's
     approved list, or the caller presents the host-configured approval token. The token is a
     confirmation boundary configured by the operator (environment variable), not an
     authorization system; it is never issued by this server;
  8. the resource is re-observed; if its abstract state no longer matches the offer, nothing runs.

After execution the resource is read back, the observed effects are compared with the offer's
prediction, and everything is written to the `executions` audit table. A mismatch is only ever
turned into a counterexample (`report_counterexample`), never into new trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from mcpgraphy.client_adapter import InMemoryToolClient
from mcpgraphy.graph.store import GraphStore
from mcpgraphy.learn.abstract import diff_effects, features_of
from mcpgraphy.models import (
    AbstractState,
    Availability,
    BehaviorStatus,
    Counterexample,
    EffectClass,
    Experiment,
    Offer,
    OutcomeKind,
    Snapshot,
    content_id,
)
from mcpgraphy.offers.explain import action_class_for
from mcpgraphy.registry import ServerSpec


@dataclass(frozen=True)
class ExecutionPolicy:
    """Host-controlled. `deny` (default) refuses every effectful call; `read_only` allows only
    READ_ONLY tools; `approved` allows effectful tools subject to the approval gates."""

    mode: Literal["deny", "read_only", "approved"] = "deny"
    approved_tools: frozenset[str] = field(default_factory=frozenset)
    approval_token: str | None = None
    allow_uncertain: bool = False


class ExecutionRecord(BaseModel):
    execution_id: str
    idempotency_key: str
    offer_id: str
    server_id: str
    tool: str
    args: dict[str, Any]
    decision: Literal["EXECUTED", "REFUSED", "REPLAYED"]
    reason: str
    executed: bool
    before_state: dict[str, Any] | None = None
    after_state: dict[str, Any] | None = None
    outcome_kind: str | None = None
    predicted_effects: list[str]
    observed_effects: list[str] | None = None
    prediction_matched: bool | None = None
    model_version: str
    transition_ids: list[str]
    counterexample_id: str | None = None
    created_at: int = 0
    caveats: list[str] = []


class CounterexampleReport(BaseModel):
    execution_id: str
    recorded: bool
    reason: str
    counterexample_id: str | None = None
    contradicted_transition_ids: list[str] = []


class Executor:
    def __init__(self, db_path: str, policy: ExecutionPolicy) -> None:
        self.db_path = db_path
        self.policy = policy

    async def execute(
        self,
        client: InMemoryToolClient,
        spec: ServerSpec,
        offer: Offer,
        offer_state_id: str,
        args: dict[str, Any],
        approval_token: str | None,
        idempotency_key: str | None,
    ) -> ExecutionRecord:
        key = idempotency_key or content_id("ikey", {"offer_id": offer.offer_id, "args": args})
        with GraphStore(self.db_path) as store:
            prior = store.execution_by_key(key)
            if prior is not None:
                return ExecutionRecord(
                    **{k: v for k, v in prior.items() if k != "decision"},
                    decision="REPLAYED",
                    executed=False,
                    caveats=[
                        "idempotent replay: this key was already processed; no call was made",
                        f"original decision: {prior['decision']}",
                    ],
                )
            seq = store.executions_count() + 1
        ac = action_class_for(None, spec, offer)
        with GraphStore(self.db_path) as store:
            rows = store.transitions_for(
                offer.behavior.model_version, offer_state_id, ac.action_class_id
            )
        transition_ids = [t.transition_id for t in rows]
        base = {
            "execution_id": content_id("exec", {"key": key, "seq": seq}),
            "idempotency_key": key,
            "offer_id": offer.offer_id,
            "server_id": offer.server_id,
            "tool": offer.tool,
            "args": dict(args),
            "predicted_effects": list(offer.behavior.predicted_effects),
            "model_version": offer.behavior.model_version,
            "transition_ids": transition_ids,
            "created_at": seq,
        }

        def refuse(reason: str) -> ExecutionRecord:
            rec = ExecutionRecord(
                **base,
                decision="REFUSED",
                reason=reason,
                executed=False,
                caveats=["nothing was executed"],
            )
            self._persist(rec)
            return rec

        # 1. argument template
        for k, v in offer.argument_template.items():
            if k not in args:
                return refuse(f"INVALID_ARGUMENT: missing argument {k!r}")
            if not (isinstance(v, str) and v.startswith("<")) and args[k] != v:
                return refuse(f"INVALID_ARGUMENT: argument {k!r} must equal the offer template")
        # 3-5. behavioral gates
        if (
            offer.availability in (Availability.NEEDS_OBSERVATION, Availability.UNKNOWN_UNTESTED)
            or offer.behavior.status is BehaviorStatus.UNKNOWN
        ):
            return refuse("UNKNOWN_OFFER: unknown or untested offers never execute")
        if offer.availability is Availability.NOT_SUPPORTED_FROM_THIS_STATE:
            return refuse("NOT_SUPPORTED: observed to fail from this state")
        if (
            offer.availability is Availability.UNCERTAIN_CANDIDATE
            and not self.policy.allow_uncertain
        ):
            return refuse("UNCERTAIN: policy does not allow uncertain offers")
        # 6. effect policy
        effect = spec.tool(offer.tool).effect
        if effect is not EffectClass.READ_ONLY:
            if self.policy.mode == "deny":
                return refuse("POLICY_DENY: effectful execution is disabled by the host policy")
            if self.policy.mode == "read_only":
                return refuse("POLICY_READ_ONLY: only read-only tools may execute")
        # 7. approval
        if offer.authorization.required:
            by_list = offer.tool in self.policy.approved_tools
            by_token = (
                self.policy.approval_token is not None
                and approval_token is not None
                and approval_token == self.policy.approval_token
            )
            if not (by_list or by_token):
                return refuse("APPROVAL_REQUIRED: host approval not granted for this tool")
        # 8. freshness
        local_id = offer.argument_template.get(spec.id_param)
        before = await self._observe(client, spec, str(local_id)) if local_id is not None else None
        if before is not None:
            before_state = AbstractState.build(
                spec.server_id, spec.resource_class, features_of(spec.resource_class, before)
            )
            if before_state.state_id != offer_state_id:
                return refuse("STALE_SNAPSHOT: resource state changed since the offer was made")
            base["before_state"] = dict(before_state.features)
        # execute
        call = await client.call(offer.tool, args)
        after = await self._observe(client, spec, str(local_id)) if local_id is not None else None
        after_feats = features_of(spec.resource_class, after) if after is not None else None
        observed = (
            list(diff_effects(base.get("before_state") or {}, after_feats))
            if after_feats is not None and base.get("before_state") is not None
            else None
        )
        matched: bool | None = None
        if observed is not None:
            if call.outcome.kind.is_ok:
                matched = observed == list(offer.behavior.predicted_effects)
            else:
                matched = False
        rec = ExecutionRecord(
            **base,
            decision="EXECUTED",
            reason="all gates passed",
            executed=True,
            after_state=after_feats,
            outcome_kind=str(call.outcome.kind),
            observed_effects=observed,
            prediction_matched=matched,
            caveats=[
                "read-back verification cannot prove the absence of other side effects",
                "a mismatch here is a drift signal; it never adds trust to the model",
            ],
        )
        self._persist(
            rec,
            before=before,
            after=after,
            outcome=call.outcome.kind,
            ac_id=ac.action_class_id,
            spec=spec,
            local_id=str(local_id),
        )
        return rec

    async def _observe(
        self, client: InMemoryToolClient, spec: ServerSpec, local_id: str
    ) -> Snapshot:
        rec = await client.call(spec.observation_tool, {spec.id_param: local_id})
        return Snapshot(tool=spec.observation_tool, raw=rec.raw, outcome=rec.outcome)

    def _persist(
        self,
        rec: ExecutionRecord,
        *,
        before: Snapshot | None = None,
        after: Snapshot | None = None,
        outcome: OutcomeKind | None = None,
        ac_id: str | None = None,
        spec: ServerSpec | None = None,
        local_id: str | None = None,
    ) -> None:
        with GraphStore(self.db_path) as store, store.conn:
            row = rec.model_dump(mode="json")
            if not rec.executed:
                # a refusal is audited but never claims the idempotency key: only an executed
                # call must be protected from repetition; a refused request may be retried once
                # the policy, approval or state has changed
                row["idempotency_key"] = f"{rec.idempotency_key}#refused#{rec.execution_id}"
            store.save_execution(row)
            if rec.executed and before is not None and after is not None and spec is not None:
                # audit the execution as an experiment row so counterexamples can link to it
                run_id = f"exec_{rec.model_version}"
                exp = Experiment(
                    experiment_id=rec.execution_id,
                    run_id=run_id,
                    seq=rec.created_at,
                    server_version_id="live",
                    fixture_id="live",
                    action_class_id=ac_id or "",
                    resource_ref=f"{spec.resource_class}:{local_id}",
                    args=rec.args,
                    before=before,
                    before_state_id=AbstractState.build(
                        spec.server_id,
                        spec.resource_class,
                        features_of(spec.resource_class, before),
                    ).state_id,
                    result=__import__("mcpgraphy.models", fromlist=["Outcome"]).Outcome(
                        kind=outcome
                    ),
                    after=after,
                    after_state_id=AbstractState.build(
                        spec.server_id, spec.resource_class, features_of(spec.resource_class, after)
                    ).state_id,
                    reset_confirmed=False,
                    phase="execution",
                    created_at=rec.created_at,
                )
                store.save_experiments([exp])

    def report_counterexample(self, execution_id: str) -> CounterexampleReport:
        with GraphStore(self.db_path) as store:
            ex = store.execution(execution_id)
            if ex is None:
                return CounterexampleReport(
                    execution_id=execution_id, recorded=False, reason="NOT_FOUND: unknown execution"
                )
            if not ex["decision"] == "EXECUTED":
                return CounterexampleReport(
                    execution_id=execution_id,
                    recorded=False,
                    reason="nothing executed; no behavior observed",
                )
            if ex["prediction_matched"] is None:
                return CounterexampleReport(
                    execution_id=execution_id,
                    recorded=False,
                    reason="no read-back available; cannot judge",
                )
            if ex["prediction_matched"]:
                return CounterexampleReport(
                    execution_id=execution_id,
                    recorded=False,
                    reason="observed behavior matched the prediction; matches never raise trust",
                )
            if ex["counterexample_id"]:
                return CounterexampleReport(
                    execution_id=execution_id,
                    recorded=False,
                    reason="already recorded",
                    counterexample_id=ex["counterexample_id"],
                )
            contradicted: list[str] = []
            cx_id: str | None = None
            with store.conn:
                for tid in ex["transition_ids"]:
                    rows = [
                        t for t in store.transitions(ex["model_version"]) if t.transition_id == tid
                    ]
                    if not rows:
                        continue
                    t = rows[0]
                    if not t.outcome_kind.is_ok or t.status is BehaviorStatus.CONTRADICTED:
                        continue
                    cx = Counterexample(
                        counterexample_id=Counterexample.make_id(
                            ex["model_version"], tid, execution_id
                        ),
                        model_version=ex["model_version"],
                        transition_id=tid,
                        experiment_id=execution_id,
                        predicted={"outcome": "OK", "effects": ex["predicted_effects"]},
                        observed={"outcome": ex["outcome_kind"], "effects": ex["observed_effects"]},
                        created_at=ex["created_at"],
                    )
                    store.contradict_transition(tid, cx)
                    contradicted.append(tid)
                    cx_id = cx_id or cx.counterexample_id
                if cx_id:
                    ex["counterexample_id"] = cx_id
                    store.save_execution(ex)
            if not contradicted:
                return CounterexampleReport(
                    execution_id=execution_id,
                    recorded=False,
                    reason="no supported/conditional success edge to contradict",
                )
            return CounterexampleReport(
                execution_id=execution_id,
                recorded=True,
                reason="transition(s) downgraded to CONTRADICTED",
                counterexample_id=cx_id,
                contradicted_transition_ids=contradicted,
            )
