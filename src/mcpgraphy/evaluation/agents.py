"""One deterministic simulated agent, many guidance sources.

The agent is intentionally simple and identical for every baseline: observe, check the goal,
take the handwritten prior list of plausible actions for the goal, ask the guidance to label
each action, and take the best-labelled untried action. The only thing that differs between
baselines is `Guidance.rank`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from mcpgraphy.client_adapter import InMemoryToolClient
from mcpgraphy.learn.abstract import features_of_raw
from mcpgraphy.models import Features
from mcpgraphy.registry import ServerSpec
from mcpgraphy.sandbox.base import FixtureManifest

# Policy: follow the agent's own goal-ordered prior, but (1) never take NOT_SUPPORTED actions,
# (2) prefer evidence-backed actions (SUPPORTED or UNCERTAIN) over UNKNOWN ones, and (3) take
# actions predicted to change nothing (NOOP) only when nothing else is left.
LABEL_RANK = {"SUPPORTED": 0, "UNCERTAIN": 0, "UNKNOWN": 1, "NOOP": 2, "NOT_SUPPORTED": 9}


@dataclass(frozen=True)
class Action:
    tool: str
    args: dict[str, Any]

    @property
    def key(self) -> str:
        return f"{self.tool}({', '.join(f'{k}={v}' for k, v in sorted(self.args.items()))})"


class Guidance(Protocol):
    name: str

    async def rank(
        self,
        client: InMemoryToolClient,
        manifest: FixtureManifest,
        features: Features,
        prior: list[Action],
    ) -> list[str]:
        """One label per prior action: SUPPORTED | UNCERTAIN | UNKNOWN | NOOP | NOT_SUPPORTED.
        NOOP means "supported, but predicted to leave the observed state unchanged"."""
        ...


@dataclass
class StepRecord:
    step: int
    tool: str
    args: dict[str, Any]
    label: str
    valid: bool
    outcome: str
    state_before: Features


@dataclass
class Episode:
    scenario_id: str
    baseline: str
    result: str  # COMPLETED | ABSTAINED | EXHAUSTED
    steps: list[StepRecord] = field(default_factory=list)
    invalid_calls: int = 0
    suppressed_valid: int = 0
    suppression_opportunities: int = 0
    abstain_was_correct: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "baseline": self.baseline,
            "result": self.result,
            "steps": [s.__dict__ for s in self.steps],
            "calls": len(self.steps),
            "invalid_calls": self.invalid_calls,
            "suppressed_valid": self.suppressed_valid,
            "suppression_opportunities": self.suppression_opportunities,
            "abstain_was_correct": self.abstain_was_correct,
        }


ValidityFn = Callable[[str, dict[str, Any]], bool]
HookFn = Callable[[], Awaitable[None] | None]


async def run_episode(
    scenario_id: str,
    baseline: str,
    spec: ServerSpec,
    client: InMemoryToolClient,
    manifest: FixtureManifest,
    goal: dict[str, Any],
    prior: list[Action],
    guidance: Guidance,
    is_valid: ValidityFn,
    max_steps: int,
    achievable: bool,
    after_first_observe: HookFn | None = None,
) -> Episode:
    ep = Episode(scenario_id=scenario_id, baseline=baseline, result="EXHAUSTED")
    local_id = manifest.focal_ref.local_id
    # attempts per (observed state, action); an action may be retried once in the same visible
    # state after some other call succeeded (hidden state may have been repaired), never more.
    attempts: dict[str, dict[str, int]] = {}
    MAX_ATTEMPTS = 2
    blocked: set[str] = set()  # actions attempted since the last successful call
    for step in range(1, max_steps + 1):
        rec = await client.call(spec.observation_tool, {spec.id_param: local_id})
        features = features_of_raw(spec.resource_class, rec.raw) if rec.raw else {"exists": False}
        if rec.raw and all(rec.raw.get(k) == v for k, v in goal.items()):
            ep.result = "COMPLETED"
            return ep
        if step == 1 and after_first_observe is not None:
            maybe = after_first_observe()
            if maybe is not None:
                await maybe
        labels = await guidance.rank(client, manifest, features, prior)
        validity = [is_valid(a.tool, a.args) for a in prior]
        if any(validity):
            ep.suppression_opportunities += 1
            if any(v and lab == "NOT_SUPPORTED" for v, lab in zip(validity, labels, strict=True)):
                ep.suppressed_valid += 1
        state_key = str(sorted(features.items()))
        done = attempts.setdefault(state_key, {})
        candidates = [
            (LABEL_RANK[lab], i, a)
            for i, (a, lab) in enumerate(zip(prior, labels, strict=True))
            if lab != "NOT_SUPPORTED" and done.get(a.key, 0) < MAX_ATTEMPTS and a.key not in blocked
        ]
        if not candidates:
            ep.result = "ABSTAINED"
            ep.abstain_was_correct = not achievable
            return ep
        _, idx, action = min(candidates)
        done[action.key] = done.get(action.key, 0) + 1
        blocked.add(action.key)
        valid = validity[idx]
        call = await client.call(action.tool, action.args)
        ep.steps.append(
            StepRecord(
                step=step,
                tool=action.tool,
                args=action.args,
                label=labels[idx],
                valid=valid,
                outcome=str(call.outcome.kind),
                state_before=features,
            )
        )
        if not valid:
            ep.invalid_calls += 1
        if call.outcome.kind.is_ok:
            # a successful call may have changed hidden state: earlier failures may be stale,
            # so everything becomes retryable again (bounded by MAX_ATTEMPTS per state)
            blocked.clear()
    # one last goal check after the final step
    rec = await client.call(spec.observation_tool, {spec.id_param: local_id})
    if rec.raw and all(rec.raw.get(k) == v for k, v in goal.items()):
        ep.result = "COMPLETED"
    return ep


# ---------------------------------------------------------------------------------------------
# Handwritten intent tables: what an agent reading the tool descriptions would try, in order.
# Identical for every baseline.
# ---------------------------------------------------------------------------------------------


def prior_actions(
    spec: ServerSpec, manifest: FixtureManifest, goal: dict[str, Any], distractors: dict[str, str]
) -> list[Action]:
    rid = manifest.focal_ref.local_id
    pools = manifest.id_pools
    if spec.server_id == "release_sim":
        table = {
            "PUBLISHED": [
                ("publish_release", {}),
                ("run_tests", {}),
                ("edit_release", {"title": "Edited by agent"}),
            ],
            "ARCHIVED": [("archive_release", {}), ("publish_release", {}), ("run_tests", {})],
            "DRAFT": [("edit_release", {"title": "Edited by agent"}), ("run_tests", {})],
        }
        base = [Action(t, {"release_id": rid, **a}) for t, a in table[goal["status"]]]
        if "release_id" in distractors:
            wrong = pools[distractors["release_id"]][0]
            base = [
                Action(t, {"release_id": wrong, **a}) for t, a in table[goal["status"]][:2]
            ] + base
        return base
    table = {
        "DONE": [
            ("move_ticket", {"state": "DONE"}),
            ("move_ticket", {"state": "IN_PROGRESS"}),
            ("assign_ticket", {"user_id": "u-alpha-2"}),
            ("close_ticket", {}),
        ],
        "IN_PROGRESS": [
            ("move_ticket", {"state": "IN_PROGRESS"}),
            ("assign_ticket", {"user_id": "u-alpha-2"}),
            ("move_ticket", {"state": "OPEN"}),
        ],
        "CLOSED": [
            ("close_ticket", {}),
            ("move_ticket", {"state": "DONE"}),
            ("move_ticket", {"state": "IN_PROGRESS"}),
            ("assign_ticket", {"user_id": "u-alpha-2"}),
        ],
    }
    base = [Action(t, {"ticket_id": rid, **a}) for t, a in table[goal["state"]]]
    extra: list[Action] = []
    if "user_id" in distractors:
        extra.append(
            Action("assign_ticket", {"ticket_id": rid, "user_id": pools[distractors["user_id"]][0]})
        )
    if "ticket_id" in distractors:
        extra.append(
            Action(
                "move_ticket",
                {"ticket_id": pools[distractors["ticket_id"]][0], "state": goal["state"]},
            )
        )
    return extra + base
