"""B4: trace-only passive corpus.

Runs the raw-tools agent (B0) on the learner's fixtures toward a fixed goal list and records
every call as a journal experiment in the same format the explorer writes: before snapshot,
call, after snapshot. No resets between steps, no contrast probes, no replication. The journal
is then materialized with the same builder, so B4 and B5 differ only in how evidence was
gathered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcpgraphy.client_adapter import InMemoryToolClient
from mcpgraphy.evaluation.agents import Action, prior_actions
from mcpgraphy.graph.builder import materialize
from mcpgraphy.graph.store import GraphStore
from mcpgraphy.learn.abstract import features_of
from mcpgraphy.models import (
    AbstractState,
    ActionClass,
    Experiment,
    ServerVersion,
    Snapshot,
    make_model_version,
    make_run_id,
)
from mcpgraphy.observe.journal import Journal
from mcpgraphy.registry import ServerSpec, classify_value
from mcpgraphy.sandbox.base import SandboxControl

GOALS = {
    "release_sim": [{"status": "PUBLISHED"}, {"status": "ARCHIVED"}],
    "ticket_sim": [{"state": "DONE"}, {"state": "CLOSED"}],
}


async def record_passive_corpus(
    sandbox: SandboxControl,
    spec: ServerSpec,
    db_path: str | Path,
    journal_path: str | Path,
    seed: int = 0,
    fixtures: int = 20,
    max_steps: int = 6,
) -> str:
    run_id = make_run_id(spec.server_id, spec.version, seed, fixtures, 0, {"passive": True})
    model_version = make_model_version(spec.server_id, spec.version, run_id)
    seq = 0
    async with InMemoryToolClient(sandbox.mcp) as client:
        journal = Journal(journal_path)
        tools = await client.list_tools()
        sv = ServerVersion.build(spec.server_id, spec.version, tools)
        journal.append(
            "run_start",
            {
                "run_id": run_id,
                "model_version": model_version,
                "server_id": spec.server_id,
                "version": spec.version,
                "server_version_id": sv.server_version_id,
                "fingerprint": sv.fingerprint,
                "tools": [t.to_dict() for t in sv.tools],
                "seed": seed,
                "fixtures": fixtures,
                "budget": 0,
                "ablation": "passive",
                "base_model_version": None,
                "reset_supported": True,
            },
        )
        known: set[str] = set()
        for fid in sandbox.fixtures(seed, fixtures):
            for goal in GOALS[spec.server_id]:
                manifest = sandbox.reset_fixture(fid)
                local_id = manifest.focal_ref.local_id
                prior = prior_actions(spec, manifest, goal, {})
                tried: set[str] = set()
                for _ in range(max_steps):
                    before = await _snap(client, spec, local_id)
                    bstate = AbstractState.build(
                        spec.server_id,
                        spec.resource_class,
                        features_of(spec.resource_class, before),
                    )
                    known.add(bstate.state_id)
                    if before.raw and all(before.raw.get(k) == v for k, v in goal.items()):
                        break
                    action = next((a for a in prior if a.key not in tried), None)
                    if action is None:
                        break
                    tried.add(action.key)
                    rec = await client.call(action.tool, action.args)
                    after = await _snap(client, spec, local_id)
                    astate = AbstractState.build(
                        spec.server_id, spec.resource_class, features_of(spec.resource_class, after)
                    )
                    ac = _action_class(spec, manifest, action)
                    seq += 1
                    exp = Experiment(
                        experiment_id=Experiment.make_id(
                            run_id, sv.server_version_id, fid, ac.action_class_id, seq
                        ),
                        run_id=run_id,
                        seq=seq,
                        server_version_id=sv.server_version_id,
                        fixture_id=fid,
                        action_class_id=ac.action_class_id,
                        resource_ref=manifest.focal_ref.render(),
                        args=action.args,
                        before=before,
                        before_state_id=bstate.state_id,
                        result=rec.outcome,
                        raw_result=rec.raw,
                        after=after,
                        after_state_id=astate.state_id,
                        reset_confirmed=False,
                        phase="passive",
                        created_at=seq,
                    )
                    journal.append(
                        "experiment",
                        {
                            **exp.to_dict(),
                            "before_features": bstate.features,
                            "after_features": astate.features,
                            "action_class": ac.to_dict(),
                        },
                    )
        journal.append(
            "run_end",
            {
                "run_id": run_id,
                "budget": {
                    "total": 0,
                    "used": 0,
                    "resets": 0,
                    "skipped": 0,
                    "destructive_cap": 0,
                    "by_effect": {},
                },
                "calls": sandbox.counter.snapshot(),
                "aborted": None,
                "phases": {"passive": seq},
                "known_states": sorted(known),
            },
        )
    with GraphStore(db_path) as store:
        return materialize(journal_path, store, run_id=run_id)


async def _snap(client: InMemoryToolClient, spec: ServerSpec, local_id: str) -> Snapshot:
    rec = await client.call(spec.observation_tool, {spec.id_param: local_id})
    return Snapshot(tool=spec.observation_tool, raw=rec.raw, outcome=rec.outcome)


def _action_class(spec: ServerSpec, manifest: Any, a: Action) -> ActionClass:
    tool = spec.tool(a.tool)
    classes = {
        p.name: classify_value(manifest, spec, a.tool, p.name, a.args.get(p.name))
        for p in tool.params
    }
    return ActionClass.build(spec.server_id, a.tool, classes, tool.effect)


__all__ = ["record_passive_corpus"]
