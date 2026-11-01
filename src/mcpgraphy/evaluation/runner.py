"""Benchmark runner: build worlds deterministically, learn models, run paired baselines, write
results.

Synchronous on purpose: every episode runs under its own `asyncio.run`, and the AALpy baseline
drives a private event loop, so nothing here may be called from inside a running loop.

Outputs under `<out>/<run_id>/`:
  scenarios.json   the 200 scenario configurations
  results.jsonl    one line per scenario x baseline (deterministic for a seed)
  models/*.json    graph exports used by B4/B5 and model-level metrics
  summary.json     per-category and overall metrics, paired bootstrap CIs vs B0
  summary.csv      long format
  report.md        human-readable report
  timing.json      wall-clock numbers (kept out of results.jsonl so it stays byte-identical)
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcpgraphy import __version__
from mcpgraphy.client_adapter import InMemoryToolClient
from mcpgraphy.evaluation.agents import Episode, prior_actions, run_episode
from mcpgraphy.evaluation.baselines import (
    LearnedGuidance,
    OracleGuidance,
    RawToolsGuidance,
    TraceGuidance,
)
from mcpgraphy.evaluation.metrics import summarize, to_csv_rows
from mcpgraphy.evaluation.tasks import CATEGORY_NAMES, RECIPES, Scenario, scenario_seed
from mcpgraphy.graph.export import export_json
from mcpgraphy.graph.store import GraphStore
from mcpgraphy.learn.pipeline import run_learn
from mcpgraphy.models import (
    AbstractState,
    BehaviorStatus,
    OutcomeKind,
    canonical_json,
    content_id,
)
from mcpgraphy.registry import concretize, get_spec
from mcpgraphy.sandbox import NoResetSandbox, make_sandbox


def _suites_dir() -> Path:
    import mcpgraphy.bench as benchmarks

    return Path(benchmarks.__file__).resolve().parent / "suites"


@dataclass
class SuiteConfig:
    name: str
    servers: list[str]
    categories: list[int]
    seeds_per_category: int
    seed_base: int
    max_steps: int
    learner: dict[str, int]
    retest_budget: int
    held_out_seed: int
    baselines: list[str]
    ablations: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, name_or_path: str) -> SuiteConfig:
        p = Path(name_or_path)
        if not p.exists():
            p = _suites_dir() / f"{name_or_path}.json"
        data = json.loads(p.read_text())
        return cls(**data)


# ---------------------------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------------------------


def build_scenarios(cfg: SuiteConfig, limit: int | None = None) -> list[Scenario]:
    from mcpgraphy.bench.ground_truth import reachable

    scenarios: list[Scenario] = []
    for server in cfg.servers:
        for cat in cfg.categories:
            for k in range(cfg.seeds_per_category):
                recipe = RECIPES[server](cat, k)
                version = recipe.get("version", "v1")
                seed = scenario_seed(cfg.seed_base, cat, k)
                sandbox = make_sandbox(server, version)
                ids = sandbox.fixtures(seed, 64)
                index = next(
                    i
                    for i, fid in enumerate(ids)
                    if all(
                        getattr(sandbox._specs[fid], key) == val  # benchmark-side only
                        for key, val in recipe["req"].items()
                    )
                )
                manifest = sandbox.reset_fixture(ids[index])
                if recipe.get("concurrent_op"):
                    sandbox.inject_concurrent_update(ids[index], recipe["concurrent_op"])
                spec = get_spec(server, version)
                actions = [
                    (ac.tool, args)
                    for ac in spec.action_classes()
                    if spec.is_core(ac) and (args := concretize(manifest, ac)) is not None
                ]
                achievable = reachable(
                    sandbox, manifest.focal_ref.local_id, recipe["goal"], actions, cfg.max_steps
                )
                scenarios.append(
                    Scenario(
                        scenario_id=f"{server}-c{cat:02d}-k{k:02d}",
                        server_id=server,
                        server_version=version,
                        learner_version="v1",
                        category=cat,
                        seed=seed,
                        fixture_index=index,
                        local_id=manifest.focal_ref.local_id,
                        goal=recipe["goal"],
                        achievable=achievable,
                        max_steps=cfg.max_steps,
                        no_reset=bool(recipe.get("no_reset")),
                        concurrent_op=recipe.get("concurrent_op"),
                        distractors=recipe.get("distractors", {}),
                        requirement=recipe["req"],
                    )
                )
    scenarios.sort(key=lambda s: s.scenario_id)
    return scenarios[:limit] if limit else scenarios


# ---------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------


@dataclass
class ModelSet:
    db_path: Path
    versions: dict[tuple[str, str], str] = field(default_factory=dict)  # (kind, server) -> mv
    mealy: dict[str, Any] = field(default_factory=dict)  # server -> MealyResult
    metrics: dict[str, Any] = field(default_factory=dict)


def prepare_models(
    cfg: SuiteConfig, run_dir: Path, baselines: list[str], ablations: list[str]
) -> ModelSet:
    from mcpgraphy.evaluation.aalpy_sul import learn_mealy
    from mcpgraphy.evaluation.passive import record_passive_corpus

    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ms = ModelSet(db_path=run_dir / "models.sqlite")
    L = cfg.learner
    for server in cfg.servers:
        if "B5" in baselines or any(b.startswith("B5") for b in baselines):
            r = asyncio.run(
                run_learn(
                    server,
                    "v1",
                    L["seed"],
                    L["fixtures"],
                    L["budget"],
                    ms.db_path,
                    run_dir / f"journal_{server}_B5.jsonl",
                )
            )
            ms.versions[("B5", server)] = r.model_version
            rr = asyncio.run(
                run_learn(
                    server,
                    "v2",
                    L["seed"],
                    L["fixtures"],
                    cfg.retest_budget,
                    ms.db_path,
                    run_dir / f"journal_{server}_B5_RETEST.jsonl",
                    base_model=r.model_version,
                )
            )
            ms.versions[("B5_RETEST", server)] = rr.model_version
            nr = asyncio.run(
                run_learn(
                    server,
                    "v1",
                    L["seed"],
                    L["fixtures"],
                    L["budget"],
                    ms.db_path,
                    run_dir / f"journal_{server}_B5_NORESET.jsonl",
                    sandbox=NoResetSandbox(make_sandbox(server, "v1")),
                )
            )
            ms.versions[("B5_NORESET", server)] = nr.model_version
            for ab in ablations:
                if ab in ("no_counterexamples", "no_arg_classes"):
                    ra = asyncio.run(
                        run_learn(
                            server,
                            "v1",
                            L["seed"],
                            L["fixtures"],
                            L["budget"],
                            ms.db_path,
                            run_dir / f"journal_{server}_B5_{ab}.jsonl",
                            ablation=ab,
                        )
                    )
                    ms.versions[(f"B5-{ab}", server)] = ra.model_version
        if "B4" in baselines:
            spec = get_spec(server, "v1")
            mv = asyncio.run(
                record_passive_corpus(
                    make_sandbox(server, "v1"),
                    spec,
                    ms.db_path,
                    run_dir / f"journal_{server}_B4.jsonl",
                    seed=L["seed"],
                    fixtures=L["fixtures"],
                )
            )
            ms.versions[("B4", server)] = mv
        if "B3" in baselines:
            sandbox = make_sandbox(server, "v1")
            spec = get_spec(server, "v1")
            ids = sandbox.fixtures(L["seed"], L["fixtures"])
            req = RECIPES[server](1, 0)["req"]
            canonical = next(
                f for f in ids if all(getattr(sandbox._specs[f], k) == v for k, v in req.items())
            )
            ms.mealy[server] = learn_mealy(sandbox, spec, canonical, L["budget"], seed=L["seed"])
    # exports + model-level metrics
    with GraphStore(ms.db_path) as store:
        for (kind, server), mv in sorted(ms.versions.items()):
            data = export_json(store, mv)
            (models_dir / f"{kind}_{server}.json").write_text(
                json.dumps(data, indent=2, sort_keys=True)
            )
            ms.metrics[f"{kind}/{server}"] = model_metrics(store, server, mv, cfg, data)
    for server, res in ms.mealy.items():
        ms.metrics[f"B3/{server}"] = {
            "converged": res.converged,
            "probes": res.probes,
            "resets": res.resets,
            "mealy_states": res.states,
            "error": res.error,
            "table_entries": len(res.table),
        }
    return ms


def model_metrics(
    store: GraphStore, server: str, mv: str, cfg: SuiteConfig, data: dict[str, Any]
) -> dict[str, Any]:
    """Behavioral precision on held-out fixtures, coverage of oracle-enumerated pairs,
    exploration cost from the run's own accounting."""
    from mcpgraphy.bench.ground_truth import focal_visible, oracle_for, true_outcome

    oracle = oracle_for(server)
    m = store.model(mv)
    sv = store.server_version(m.server_version_id)
    version = sv.version if sv else "v1"
    spec = get_spec(server, version)
    acs = store.action_classes(server)
    rows = store.transitions(mv)
    # held-out fixtures grouped by visible features
    sandbox = make_sandbox(server, version)
    held = sandbox.fixtures(cfg.held_out_seed, 60)
    by_feats: dict[str, list[str]] = {}
    for fid in held:
        man = sandbox.reset_fixture(fid)
        feats = focal_visible(sandbox, man.focal_ref.local_id)
        by_feats.setdefault(canonical_json(feats), []).append(fid)
    checks = ok = 0
    for t in rows:
        if t.status is not BehaviorStatus.OBSERVED_SUPPORTED or not t.outcome_kind.is_ok:
            continue
        ac = acs.get(t.action_class_id)
        st = store.state(t.source_state_id)
        if ac is None or st is None or not spec.is_core(ac) or "exists" in st.features:
            continue
        for fid in by_feats.get(canonical_json(st.features), [])[:5]:
            man = sandbox.reset_fixture(fid)
            args = concretize(man, ac)
            if args is None:
                continue
            checks += 1
            ok += true_outcome(sandbox, ac.tool, args) is OutcomeKind.OK
    # coverage over visible cells x core action classes
    cells = oracle.visible_cells()
    core = [ac for ac in spec.action_classes() if spec.is_core(ac)]
    covered = {
        (t.source_state_id, t.action_class_id)
        for t in rows
        if t.status is not BehaviorStatus.UNKNOWN
    }
    total = 0
    hit = 0
    for cell in cells:
        sid = AbstractState.build(server, spec.resource_class, cell).state_id
        for ac in core:
            total += 1
            hit += (sid, ac.action_class_id) in covered
    events = [e for e in store.events(m.run_id) if e["kind"] == "run_end"]
    calls = events[0].get("calls", {}) if events else {}
    return {
        "model_version": mv,
        "behavioral_precision": (ok / checks) if checks else None,
        "precision_checks": checks,
        "coverage": hit / total if total else None,
        "coverage_pairs": f"{hit}/{total}",
        "by_status": data["model"]["coverage"].get("by_status", {}),
        "counterexamples": data["model"]["coverage"].get("counterexamples", 0),
        "unknown_pairs": data["model"]["coverage"].get("unknown_pairs"),
        "exploration_cost": {
            "probes": m.budget.get("used"),
            "resets": m.budget.get("resets"),
            "effectful_calls": m.budget.get("by_effect", {}),
            "tool_calls": calls.get("total"),
        },
        "retest": {
            "probes": data["model"]["coverage"].get("phases", {}).get("retest", 0),
            "still_required": sum(1 for t in rows if t.retest_required),
        },
        "aborted": data["model"]["coverage"].get("aborted"),
    }


# ---------------------------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------------------------


def _guidances(
    ms: ModelSet,
    store: GraphStore,
    scenario: Scenario,
    sandbox: Any,
    baselines: list[str],
    ablations: list[str],
) -> dict[str, Any]:
    from mcpgraphy.bench.ground_truth import focal_visible, oracle_for, true_outcome

    def changes_state(tool: str, args: dict[str, Any]) -> bool:
        kind, after = oracle_for(scenario.server_id).true_outcome(sandbox, tool, args)
        rid = args.get(get_spec(scenario.server_id, scenario.server_version).id_param)
        before = focal_visible(sandbox, rid) if rid else None
        if after is None or before is None:
            return True
        after_vis = {k: after.get(k) for k in before}
        return after_vis != before

    spec_live = get_spec(scenario.server_id, scenario.server_version)
    spec_v1 = get_spec(scenario.server_id, "v1")
    server = scenario.server_id
    out: dict[str, Any] = {}
    if "B0" in baselines:
        out["B0"] = RawToolsGuidance()
    if "B2" in baselines:
        out["B2"] = OracleGuidance(
            lambda t, a: true_outcome(sandbox, t, a) is OutcomeKind.OK, changes_state
        )
    if "B3" in baselines and server in ms.mealy:
        out["B3"] = TraceGuidance("B3", spec_v1, ms.mealy[server].table)
    if "B4" in baselines:
        out["B4"] = LearnedGuidance("B4", spec_v1, store, ms.versions[("B4", server)])
    if "B5" in baselines:
        kind = "B5_NORESET" if scenario.no_reset else "B5"
        out["B5"] = LearnedGuidance("B5", spec_v1, store, ms.versions[(kind, server)])
        if scenario.category in (6, 7):
            out["B5_RETEST"] = LearnedGuidance(
                "B5_RETEST", spec_live, store, ms.versions[("B5_RETEST", server)]
            )
        for ab in ablations:
            if ab == "unknown_as_valid":
                out["B5-unknown_as_valid"] = LearnedGuidance(
                    "B5-unknown_as_valid",
                    spec_v1,
                    store,
                    ms.versions[(kind, server)],
                    unknown_as_valid=True,
                )
            elif ab == "no_guards":
                out["B5-no_guards"] = LearnedGuidance(
                    "B5-no_guards", spec_v1, store, ms.versions[(kind, server)], no_guards=True
                )
            elif (f"B5-{ab}", server) in ms.versions:
                out[f"B5-{ab}"] = LearnedGuidance(
                    f"B5-{ab}", spec_v1, store, ms.versions[(f"B5-{ab}", server)]
                )
    return out


def run_one(
    scenario: Scenario,
    baseline: str,
    guidance_factory,
) -> Episode:
    raise NotImplementedError  # pragma: no cover


async def _episode(
    scenario: Scenario, baseline: str, guidance: Any, sandbox: Any, fixture_id: str
) -> Episode:
    from mcpgraphy.bench.ground_truth import true_outcome

    spec = get_spec(scenario.server_id, scenario.server_version)
    manifest = sandbox.reset_fixture(fixture_id)
    prior = prior_actions(spec, manifest, scenario.goal, scenario.distractors)

    def hook() -> None:
        if scenario.concurrent_op:
            sandbox.inject_concurrent_update(fixture_id, scenario.concurrent_op)

    async with InMemoryToolClient(sandbox.mcp) as client:
        return await run_episode(
            scenario.scenario_id,
            baseline,
            spec,
            client,
            manifest,
            scenario.goal,
            prior,
            guidance,
            lambda t, a: true_outcome(sandbox, t, a) is OutcomeKind.OK,
            scenario.max_steps,
            scenario.achievable,
            after_first_observe=hook,
        )


def run_suite(
    cfg: SuiteConfig,
    seed: int,
    out_root: str | Path,
    baselines: list[str] | None,
    ablations: list[str] | None,
    limit: int | None = None,
) -> Path:
    baselines = baselines or cfg.baselines
    ablations = cfg.ablations if ablations is None else ablations
    run_id = content_id(
        "bench",
        {
            "suite": cfg.name,
            "seed": seed,
            "baselines": baselines,
            "ablations": ablations,
            "limit": limit,
            "version": __version__,
        },
    )
    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}
    t0 = time.perf_counter()
    scenarios = build_scenarios(cfg, limit)
    (run_dir / "scenarios.json").write_text(
        json.dumps([s.to_dict() for s in scenarios], indent=2, sort_keys=True)
    )
    timing["scenarios_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    ms = prepare_models(cfg, run_dir, baselines, ablations)
    timing["models_s"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    episodes: list[dict[str, Any]] = []
    with GraphStore(ms.db_path) as store, open(run_dir / "results.jsonl", "w") as fh:
        for sc in scenarios:
            sandbox = make_sandbox(sc.server_id, sc.server_version)
            ids = sandbox.fixtures(sc.seed, sc.fixture_index + 1)
            fid = ids[sc.fixture_index]
            for name, guidance in _guidances(ms, store, sc, sandbox, baselines, ablations).items():
                ep = asyncio.run(_episode(sc, name, guidance, sandbox, fid))
                row = {
                    **ep.to_dict(),
                    "server_id": sc.server_id,
                    "category": sc.category,
                    "achievable": sc.achievable,
                }
                episodes.append(row)
                fh.write(canonical_json(row) + "\n")
    timing["episodes_s"] = time.perf_counter() - t2
    timing["total_s"] = time.perf_counter() - t0
    summary = summarize(episodes, {s.scenario_id: s.to_dict() for s in scenarios}, seed)
    summary["models"] = ms.metrics
    summary["run_id"] = run_id
    summary["suite"] = cfg.name
    summary["seed"] = seed
    summary["baselines"] = sorted({e["baseline"] for e in episodes})
    summary["scenarios"] = len(scenarios)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    with open(run_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["baseline", "server", "category", "metric", "value", "n"]
        )
        w.writeheader()
        w.writerows(to_csv_rows(summary))
    (run_dir / "report.md").write_text(render_report(summary))
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    (Path(out_root) / "latest").write_text(run_id)
    return run_dir


def resolve_run(out_root: str | Path, run: str) -> Path:
    root = Path(out_root)
    if run == "latest":
        run = (root / "latest").read_text().strip()
    p = root / run
    if not p.exists():
        raise LookupError(f"no benchmark run {run} under {root}")
    return p


# ---------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------


def _f(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def render_report(summary: dict[str, Any]) -> str:
    lines = [f"# Benchmark report: {summary['suite']} (seed {summary['seed']})", ""]
    lines.append(
        f"Run `{summary['run_id']}`, {summary['scenarios']} scenarios, baselines: "
        f"{', '.join(summary['baselines'])}."
    )
    lines.append("")
    lines.append(
        "Metrics are per scenario, then averaged. Abstentions and failures are reported "
        "separately and never folded into completion. `vs B0` rows are paired "
        "differences by scenario with a 95% bootstrap CI (2000 resamples)."
    )
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(
        "| baseline | n | completion | completion (achievable only) | abstain | "
        "correct abstain | invalid calls / task | calls / task | false suppression | "
        "invalid vs B0 [CI] | completion vs B0 [CI] |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for b, o in summary["overall"].items():
        iv, cv = o["vs_B0_invalid_calls_diff"], o["vs_B0_completion_diff"]
        lines.append(
            f"| {b} | {o['n']} | {_f(o['completion_rate'])} | "
            f"{_f(o['achievable_completion_rate'])} | {_f(o['abstain_rate'])} | "
            f"{_f(o['correct_abstain_rate'])} | {_f(o['invalid_calls_mean'])} | "
            f"{_f(o['calls_mean'])} | {_f(o['false_suppression_rate'])} | "
            f"{_f(iv['mean'])} [{_f(iv['ci95'][0])}, {_f(iv['ci95'][1])}] | "
            f"{_f(cv['mean'])} [{_f(cv['ci95'][0])}, {_f(cv['ci95'][1])}] |"
        )
    lines.append("")
    lines.append("## By category")
    lines.append("")
    lines.append(
        "| server | category | baseline | n | completion | abstain | invalid/task | "
        "false suppression |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in summary["by_category"]:
        lines.append(
            f"| {r['server']} | {r['category']} {CATEGORY_NAMES[r['category']]} | "
            f"{r['baseline']} | {r['n']} | {_f(r['completion_rate'])} | "
            f"{_f(r['abstain_rate'])} | {_f(r['invalid_calls_mean'])} | "
            f"{_f(r['false_suppression_rate'])} |"
        )
    lines.append("")
    lines.append("## Models")
    lines.append("")
    lines.append(
        "| model | precision (held-out) | checks | coverage | counterexamples | "
        "probes | resets | retest probes / still required |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for k, m in sorted(summary["models"].items()):
        if "behavioral_precision" not in m:
            lines.append(
                f"| {k} | n/a (Mealy: converged={m['converged']}, states="
                f"{m['mealy_states']}, error={m['error']}) | | | | {m['probes']} | "
                f"{m['resets']} | |"
            )
            continue
        ec = m["exploration_cost"]
        lines.append(
            f"| {k} | {_f(m['behavioral_precision'])} | {m['precision_checks']} | "
            f"{m['coverage_pairs']} | {m['counterexamples']} | {ec['probes']} | "
            f"{ec['resets']} | {m['retest']['probes']} / {m['retest']['still_required']} |"
        )
    lines.append("")
    lines.append("## Reading guide")
    lines.append("")
    lines.append(
        "- B0 never suppresses, so its false-suppression rate is 0 by construction and "
        "its invalid-call count is the cost of trial and error."
    )
    lines.append("- B2 sees hidden state; it is an upper bound, not a competitor.")
    lines.append("- B3 (AALpy L*) learns from one reset state and cannot see hidden flags.")
    lines.append(
        "- Categories 4 and 9 contain unachievable or hidden-state tasks: a correct "
        "abstention there is the right answer, and a completion is impossible."
    )
    lines.append(
        "- Categories 6-7 run on v2 servers with v1-learned models; B5_RETEST relearned "
        "on v2 with the retest budget from the v1 model."
    )
    return "\n".join(lines) + "\n"
