"""`learn` = explore (journal) -> materialize (graph). One call, deterministic for a seed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mcpgraphy.graph.builder import materialize
from mcpgraphy.graph.store import GraphStore
from mcpgraphy.learn.explorer import Explorer, ExplorerConfig, RetestTarget, RunSummary
from mcpgraphy.models import BehaviorStatus
from mcpgraphy.observe.journal import Journal
from mcpgraphy.registry import get_spec
from mcpgraphy.sandbox import RestrictedSandboxView, SandboxControl, make_sandbox


@dataclass
class LearnResult:
    summary: RunSummary
    model_version: str
    journal_path: Path
    db_path: str


async def run_learn(
    server_id: str,
    version: str,
    seed: int,
    fixtures: int,
    budget: int,
    db_path: str | Path,
    journal_path: str | Path,
    base_model: str | None = None,
    ablation: str | None = None,
    sandbox: SandboxControl | None = None,
) -> LearnResult:
    spec = get_spec(server_id, version)
    sb = sandbox or make_sandbox(server_id, version)
    view = RestrictedSandboxView(sb)
    store = GraphStore(db_path)
    try:
        retest: list[RetestTarget] = []
        base_version: str | None = None
        if base_model:
            base_version = store.resolve_model_version(server_id, base_model)
            for t in store.transitions(base_version):
                if t.status is BehaviorStatus.UNKNOWN and not t.retest_required:
                    continue
                s = store.state(t.source_state_id)
                if s is not None:
                    retest.append(RetestTarget(s.features, t.action_class_id))
            # deterministic order, one retest per (state, action) pair
            seen: set[tuple[str, str]] = set()
            deduped = []
            for r in sorted(
                retest, key=lambda r: (r.action_class_id, sorted(r.source_features.items()))
            ):
                key = (str(sorted(r.source_features.items())), r.action_class_id)
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)
            retest = deduped
        cfg = ExplorerConfig(
            seed=seed,
            fixtures=fixtures,
            budget=budget,
            ablation=ablation,
            base_model_version=base_version,
        )
        with Journal(journal_path) as journal:
            explorer = Explorer(view, spec, journal, cfg, retest_targets=retest)
            summary = await explorer.run()
        model_version = materialize(journal_path, store, run_id=summary.run_id)
    finally:
        store.close()
    return LearnResult(summary, model_version, Path(journal_path), str(db_path))
