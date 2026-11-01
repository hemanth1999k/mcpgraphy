"""Benchmark suite: 200 scenarios, reproducible small slices, negative outcomes present.

These tests are synchronous on purpose: the runner owns its event loops.
"""

from __future__ import annotations

import json
from pathlib import Path

from mcpgraphy.evaluation.aalpy_sul import learn_mealy
from mcpgraphy.evaluation.runner import SuiteConfig, build_scenarios, run_suite
from mcpgraphy.registry import get_spec
from mcpgraphy.sandbox import ReleaseSandbox


def _small_cfg(**over) -> SuiteConfig:
    cfg = SuiteConfig.load("stateful-v1")
    cfg.learner = {"seed": 0, "fixtures": 8, "budget": 60}
    cfg.retest_budget = 12
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def test_suite_generates_200_scenarios_with_mixed_achievability():
    cfg = SuiteConfig.load("stateful-v1")
    scenarios = build_scenarios(cfg)
    assert len(scenarios) == 200
    assert len({s.scenario_id for s in scenarios}) == 200
    by_cat = {}
    for s in scenarios:
        by_cat.setdefault((s.server_id, s.category), []).append(s)
    assert all(len(v) == 10 for v in by_cat.values())
    # category 2 (state guard) and 4 (hidden) are unachievable; 1 and 5 achievable
    for server in ("release_sim", "ticket_sim"):
        assert not any(s.achievable for s in by_cat[(server, 2)])
        assert not any(s.achievable for s in by_cat[(server, 4)])
        assert all(s.achievable for s in by_cat[(server, 1)])
        assert all(s.achievable for s in by_cat[(server, 5)])
        # category 9 alternates with the seed parity
        assert {s.achievable for s in by_cat[(server, 9)]} == {True, False}
        assert all(s.server_version == "v2" for s in by_cat[(server, 7)])
    # learner and benchmark seeds are disjoint
    assert min(s.seed for s in scenarios) >= 1000


def test_runner_small_slice_is_reproducible_and_reports_negatives(tmp_path: Path):
    cfg = _small_cfg(seeds_per_category=1, categories=[1, 2, 4, 5, 7, 9])
    a = run_suite(cfg, 7, tmp_path / "a", ["B0", "B2", "B4", "B5"], ["unknown_as_valid"])
    b = run_suite(cfg, 7, tmp_path / "b", ["B0", "B2", "B4", "B5"], ["unknown_as_valid"])
    assert (a / "results.jsonl").read_text() == (b / "results.jsonl").read_text()
    rows = [json.loads(line) for line in (a / "results.jsonl").read_text().splitlines()]
    baselines = {r["baseline"] for r in rows}
    assert {"B0", "B2", "B4", "B5", "B5-unknown_as_valid", "B5_RETEST"} <= baselines
    # every baseline has at least one non-success somewhere (no cherry-picking possible)
    for bl in ("B0", "B4", "B5"):
        mine = [r for r in rows if r["baseline"] == bl]
        assert any(r["invalid_calls"] > 0 or r["result"] != "COMPLETED" for r in mine), bl
    # the oracle never makes an invalid call and B0 never suppresses
    assert all(r["invalid_calls"] == 0 for r in rows if r["baseline"] == "B2")
    assert all(r["suppressed_valid"] == 0 for r in rows if r["baseline"] == "B0")
    summary = json.loads((a / "summary.json").read_text())
    assert "B5" in summary["overall"] and "vs_B0_invalid_calls_diff" in summary["overall"]["B5"]
    assert summary["models"]["B5/release_sim"]["behavioral_precision"] is not None
    assert (a / "report.md").read_text().startswith("# Benchmark report")
    assert (a / "summary.csv").exists() and (a / "models" / "B5_release_sim.json").exists()


def test_aalpy_sul_respects_budget_and_falls_back_to_traces():
    sb = ReleaseSandbox("v1")
    spec = get_spec("release_sim", "v1")
    fid = sb.fixtures(0, 1)[0]
    res = learn_mealy(sb, spec, fid, budget=5, seed=0)
    assert res.probes <= 5
    assert res.converged is False and "budget" in (res.error or "")
    assert res.table  # trace fallback still gives guidance


def test_aalpy_learns_a_mealy_machine_on_the_clean_fixture():
    sb = ReleaseSandbox("v1")
    spec = get_spec("release_sim", "v1")
    ids = sb.fixtures(0, 11)
    fid = next(f for f in ids if sb._specs[f].status == "DRAFT" and not sb._specs[f].tests_passed)
    # measured: L* needs ~870 probes (247 resets) to converge on release_sim; the benchmark
    # budget of 150 is far below that, which is reported, not hidden.
    res = learn_mealy(sb, spec, fid, budget=2000, seed=0)
    assert res.converged, res.error
    assert res.states >= 3  # DRAFT/untested, DRAFT/tested, PUBLISHED, ... are distinguishable
    assert res.probes > 400
