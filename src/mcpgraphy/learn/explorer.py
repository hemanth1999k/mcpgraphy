"""Safe experiment scheduler.

Every probe is: reset fixture -> observe -> safety check -> call one tool once -> observe ->
journal. The explorer has no random number generator; probe order is a sorted tuple. It only
ever talks to a `RestrictedSandboxView`, so it cannot see hidden state or the oracle.

Phases:
  0  retest   - with a base model: re-probe transitions flagged `retest_required` first
  1  tier 0/1 - every fixture x core action classes, then parameter regions on two fixtures
  2  replicate - re-probe (state, action) pairs seen once on a *different* fixture with the same
                 visible features; this is where hidden conditions surface as contradictions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mcpgraphy.client_adapter import InMemoryToolClient
from mcpgraphy.learn.abstract import features_of, features_of_raw
from mcpgraphy.models import (
    AbstractState,
    ActionClass,
    EffectClass,
    Experiment,
    Features,
    ServerVersion,
    Snapshot,
    make_model_version,
    make_run_id,
)
from mcpgraphy.observe.journal import Journal
from mcpgraphy.registry import ServerSpec, concretize
from mcpgraphy.safety import Budget, ResetError, SafetyPolicy
from mcpgraphy.sandbox.base import FixtureManifest, RestrictedSandboxView


class ExplorationAborted(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ExplorerConfig:
    seed: int = 0
    fixtures: int = 20
    budget: int = 150
    phase2_fraction: float = 0.3
    tier1_fixtures: int = 2
    ablation: str | None = None  # "no_counterexamples" skips phase 2; "no_arg_classes" tier 1
    base_model_version: str | None = None


@dataclass
class RetestTarget:
    """A transition inherited from an older model that must be re-probed."""

    source_features: Features
    action_class_id: str


@dataclass
class RunSummary:
    run_id: str
    model_version: str
    server_version: ServerVersion
    experiments: int
    budget: dict[str, Any]
    aborted: str | None = None
    phases: dict[str, int] = field(default_factory=dict)


class Explorer:
    def __init__(
        self,
        sandbox: RestrictedSandboxView,
        spec: ServerSpec,
        journal: Journal,
        config: ExplorerConfig,
        retest_targets: list[RetestTarget] | None = None,
    ) -> None:
        self.sb = sandbox
        self.spec = spec
        self.journal = journal
        self.cfg = config
        self.retest_targets = retest_targets or []
        self.budget = Budget(total=config.budget)
        self.policy = SafetyPolicy(allowed_tools=spec.allowed_tools)
        self.run_id = make_run_id(
            spec.server_id,
            spec.version,
            config.seed,
            config.fixtures,
            config.budget,
            {
                "ablation": config.ablation,
                "base": config.base_model_version,
                "reset_supported": bool(sandbox.reset_supported),
            },
        )
        self.model_version = make_model_version(spec.server_id, spec.version, self.run_id)
        self._seq = 0
        self._experiments: list[Experiment] = []
        self._manifests: dict[str, FixtureManifest] = {}
        self._fixture_features: dict[str, Features] = {}
        self._fixture_order: list[str] = []
        self._action_classes: dict[str, ActionClass] = {
            ac.action_class_id: ac for ac in spec.action_classes()
        }
        self._ac_order = {ac_id: i for i, ac_id in enumerate(self._action_classes)}
        self.phases: dict[str, int] = {"retest": 0, "tier0": 0, "tier1": 0, "replicate": 0}

    # ------------------------------------------------------------------ public entry point
    async def run(self) -> RunSummary:
        async with InMemoryToolClient(self.sb.mcp) as client:
            tools = await client.list_tools()
            sv = ServerVersion.build(self.spec.server_id, self.spec.version, tools)
            self.journal.append(
                "run_start",
                {
                    "run_id": self.run_id,
                    "model_version": self.model_version,
                    "server_id": self.spec.server_id,
                    "version": self.spec.version,
                    "server_version_id": sv.server_version_id,
                    "fingerprint": sv.fingerprint,
                    "tools": [t.to_dict() for t in sv.tools],
                    "seed": self.cfg.seed,
                    "fixtures": self.cfg.fixtures,
                    "budget": self.cfg.budget,
                    "ablation": self.cfg.ablation,
                    "base_model_version": self.cfg.base_model_version,
                    "reset_supported": bool(self.sb.reset_supported),
                },
            )
            aborted: str | None = None
            try:
                self._prepare_fixtures()
                await self._phase_retest(client, sv)
                await self._phase_sweep(client, sv)
                if self.cfg.ablation != "no_counterexamples":
                    await self._phase_replicate(client, sv)
            except ExplorationAborted as e:
                aborted = e.reason
            self.journal.append(
                "run_end",
                {
                    "run_id": self.run_id,
                    "budget": self.budget.to_dict(),
                    "calls": self.sb.counter.snapshot(),
                    "aborted": aborted,
                    "phases": dict(self.phases),
                    "known_states": sorted(
                        {
                            AbstractState.build(
                                self.spec.server_id, self.spec.resource_class, f
                            ).state_id
                            for f in self._fixture_features.values()
                        }
                    ),
                },
            )
        return RunSummary(
            run_id=self.run_id,
            model_version=self.model_version,
            server_version=sv,
            experiments=len(self._experiments),
            budget=self.budget.to_dict(),
            aborted=aborted,
            phases=dict(self.phases),
        )

    # ------------------------------------------------------------------ phases
    def _prepare_fixtures(self) -> None:
        """Reset every fixture once up front: proves reset works and records each fixture's
        visible features so later phases can pick fixtures by abstract state."""
        for fid in self.sb.fixtures(self.cfg.seed, self.cfg.fixtures):
            manifest = self._reset(fid)
            self._fixture_order.append(fid)
            self._fixture_features[fid] = features_of_raw(
                self.spec.resource_class, manifest.expected_visible
            )

    async def _phase_retest(self, client: InMemoryToolClient, sv: ServerVersion) -> None:
        for target in self.retest_targets:
            ac = self._action_classes.get(target.action_class_id)
            if ac is None:
                continue  # schema drift removed this action class; builder drops those rows
            fid = self._fixture_with(target.source_features, exclude=())
            if fid is None:
                continue
            if await self._probe(client, sv, fid, ac, phase="retest") is None:
                break

    async def _phase_sweep(self, client: InMemoryToolClient, sv: ServerVersion) -> None:
        sweep_cap = int(self.cfg.budget * (1 - self.cfg.phase2_fraction))
        if self.cfg.ablation == "no_counterexamples":
            sweep_cap = self.cfg.budget
        queue: list[tuple[int, int, int, str, str]] = []
        for fi, fid in enumerate(self._fixture_order):
            for ac_id, ac in self._action_classes.items():
                if self.spec.is_core(ac):
                    queue.append((0, fi, self._ac_order[ac_id], fid, ac_id))
                elif fi < self.cfg.tier1_fixtures and self.cfg.ablation != "no_arg_classes":
                    queue.append((1, fi, self._ac_order[ac_id], fid, ac_id))
        queue.sort()
        for tier, _fi, _ai, fid, ac_id in queue:
            if self.budget.used >= sweep_cap:
                break
            phase = "tier0" if tier == 0 else "tier1"
            if await self._probe(client, sv, fid, self._action_classes[ac_id], phase) is None:
                break

    async def _phase_replicate(self, client: InMemoryToolClient, sv: ServerVersion) -> None:
        # (source_state_id, action_class_id) -> fixtures already used
        seen: dict[tuple[str, str], list[str]] = {}
        for exp in self._experiments:
            seen.setdefault((exp.before_state_id, exp.action_class_id), []).append(exp.fixture_id)
        targets = sorted(
            (ac_id, sid, fids) for (sid, ac_id), fids in seen.items() if len(set(fids)) == 1
        )
        for ac_id, sid, fids in targets:
            ac = self._action_classes[ac_id]
            if ac.effect_class is EffectClass.READ_ONLY:
                continue  # read-only probes cannot reveal hidden write conditions
            feats = self._features_for_state_id(sid)
            if feats is None:
                continue
            fid = self._fixture_with(feats, exclude=tuple(fids))
            if fid is None:
                continue
            if await self._probe(client, sv, fid, ac, phase="replicate") is None:
                break

    # ------------------------------------------------------------------ one probe
    async def _probe(
        self, client: InMemoryToolClient, sv: ServerVersion, fid: str, ac: ActionClass, phase: str
    ) -> Experiment | None:
        """Run one experiment. Returns None when the budget is exhausted (caller stops)."""
        reason = self.policy.check(
            ac.tool, ac.effect_class, self.budget, bool(self.sb.reset_supported)
        )
        if reason == "budget exhausted":
            self.journal.append("budget_exhausted", {"run_id": self.run_id, "phase": phase})
            return None
        if reason is not None:
            self.budget.skipped += 1
            self.journal.append(
                "skipped",
                {
                    "run_id": self.run_id,
                    "fixture_id": fid,
                    "action_class_id": ac.action_class_id,
                    "phase": phase,
                    "reason": reason,
                },
            )
            return Experiment.model_construct()  # sentinel: not None, keeps the loop going
        manifest = self._reset(fid)
        args = concretize(manifest, ac)
        if args is None:
            self.budget.skipped += 1
            self.journal.append(
                "skipped",
                {
                    "run_id": self.run_id,
                    "fixture_id": fid,
                    "action_class_id": ac.action_class_id,
                    "phase": phase,
                    "reason": "no concrete value for argument class",
                },
            )
            return Experiment.model_construct()
        focal = manifest.focal_ref.local_id
        before = await self._observe(client, focal)
        reset_confirmed = before.raw is not None and _visible_matches(
            before.raw, manifest.expected_visible
        )
        before_state = AbstractState.build(
            self.spec.server_id,
            self.spec.resource_class,
            features_of(self.spec.resource_class, before),
        )
        record = await client.call(ac.tool, args)
        self.budget.consume(ac.effect_class)
        after = await self._observe(client, focal)
        after_state = AbstractState.build(
            self.spec.server_id,
            self.spec.resource_class,
            features_of(self.spec.resource_class, after),
        )
        self._seq += 1
        exp = Experiment(
            experiment_id=Experiment.make_id(
                self.run_id, sv.server_version_id, fid, ac.action_class_id, self._seq
            ),
            run_id=self.run_id,
            seq=self._seq,
            server_version_id=sv.server_version_id,
            fixture_id=fid,
            action_class_id=ac.action_class_id,
            resource_ref=manifest.focal_ref.render(),
            args=args,
            before=before,
            before_state_id=before_state.state_id,
            result=record.outcome,
            raw_result=record.raw,
            after=after,
            after_state_id=after_state.state_id,
            reset_confirmed=reset_confirmed,
            phase=phase,
            created_at=self._seq,
        )
        self._experiments.append(exp)
        self.phases[phase] = self.phases.get(phase, 0) + 1
        self.journal.append(
            "experiment",
            {
                **exp.to_dict(),
                "before_features": before_state.features,
                "after_features": after_state.features,
                "action_class": ac.to_dict(),
            },
        )
        return exp

    async def _observe(self, client: InMemoryToolClient, local_id: str) -> Snapshot:
        rec = await client.call(self.spec.observation_tool, {self.spec.id_param: local_id})
        return Snapshot(tool=self.spec.observation_tool, raw=rec.raw, outcome=rec.outcome)

    # ------------------------------------------------------------------ helpers
    def _reset(self, fid: str) -> FixtureManifest:
        try:
            manifest = self.sb.reset_fixture(fid)
        except ResetError as e:
            self.journal.append(
                "abort", {"run_id": self.run_id, "fixture_id": fid, "reason": f"reset failed: {e}"}
            )
            raise ExplorationAborted(f"reset failed: {e}") from e
        self.budget.count_reset()
        self._manifests[fid] = manifest
        return manifest

    def _fixture_with(self, features: Features, exclude: tuple[str, ...]) -> str | None:
        for fid in self._fixture_order:
            if fid in exclude:
                continue
            if self._fixture_features[fid] == features:
                return fid
        return None

    def _features_for_state_id(self, state_id: str) -> Features | None:
        for feats in self._fixture_features.values():
            s = AbstractState.build(self.spec.server_id, self.spec.resource_class, feats)
            if s.state_id == state_id:
                return feats
        return None


def _visible_matches(raw: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(raw.get(k) == v for k, v in expected.items())
