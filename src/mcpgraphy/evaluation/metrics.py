"""Benchmark metrics and paired bootstrap confidence intervals. Pure Python."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(
    diffs: Sequence[float], seed: int, resamples: int = 2000, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of paired differences."""
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * resamples)]
    hi = means[min(int((1 - alpha / 2) * resamples), resamples - 1)]
    return (lo, hi)


def summarize(
    episodes: list[dict[str, Any]], scenarios: dict[str, dict[str, Any]], seed: int
) -> dict[str, Any]:
    """Per (baseline, server, category) means, plus paired differences vs B0 by scenario_id."""
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for ep in episodes:
        sc = scenarios[ep["scenario_id"]]
        groups.setdefault((ep["baseline"], sc["server_id"], sc["category"]), []).append(ep)
    rows: list[dict[str, Any]] = []
    for (baseline, server, cat), eps in sorted(groups.items()):
        rows.append(
            {
                "baseline": baseline,
                "server": server,
                "category": cat,
                "n": len(eps),
                "completion_rate": mean([e["result"] == "COMPLETED" for e in eps]),
                "achievable_completion_rate": _rate(
                    [
                        e["result"] == "COMPLETED"
                        for e in eps
                        if scenarios[e["scenario_id"]]["achievable"]
                    ]
                ),
                "abstain_rate": mean([e["result"] == "ABSTAINED" for e in eps]),
                "correct_abstain_rate": _rate(
                    [e["abstain_was_correct"] for e in eps if e["result"] == "ABSTAINED"]
                ),
                "exhausted_rate": mean([e["result"] == "EXHAUSTED" for e in eps]),
                "invalid_calls_mean": mean([e["invalid_calls"] for e in eps]),
                "calls_mean": mean([e["calls"] for e in eps]),
                "false_suppression_rate": _rate_sum(
                    [e["suppressed_valid"] for e in eps],
                    [e["suppression_opportunities"] for e in eps],
                ),
            }
        )
    # overall per baseline
    by_baseline: dict[str, list[dict[str, Any]]] = {}
    for ep in episodes:
        by_baseline.setdefault(ep["baseline"], []).append(ep)
    overall = {}
    b0 = {e["scenario_id"]: e for e in by_baseline.get("B0", [])}
    for baseline, eps in sorted(by_baseline.items()):
        paired_invalid = [
            e["invalid_calls"] - b0[e["scenario_id"]]["invalid_calls"]
            for e in eps
            if e["scenario_id"] in b0
        ]
        paired_completed = [
            float(e["result"] == "COMPLETED") - float(b0[e["scenario_id"]]["result"] == "COMPLETED")
            for e in eps
            if e["scenario_id"] in b0
        ]
        overall[baseline] = {
            "n": len(eps),
            "completion_rate": mean([e["result"] == "COMPLETED" for e in eps]),
            "achievable_completion_rate": _rate(
                [
                    e["result"] == "COMPLETED"
                    for e in eps
                    if scenarios[e["scenario_id"]]["achievable"]
                ]
            ),
            "abstain_rate": mean([e["result"] == "ABSTAINED" for e in eps]),
            "correct_abstain_rate": _rate(
                [e["abstain_was_correct"] for e in eps if e["result"] == "ABSTAINED"]
            ),
            "invalid_calls_mean": mean([e["invalid_calls"] for e in eps]),
            "invalid_calls_total": sum(e["invalid_calls"] for e in eps),
            "calls_mean": mean([e["calls"] for e in eps]),
            "false_suppression_rate": _rate_sum(
                [e["suppressed_valid"] for e in eps], [e["suppression_opportunities"] for e in eps]
            ),
            "vs_B0_invalid_calls_diff": _paired(paired_invalid, seed),
            "vs_B0_completion_diff": _paired(paired_completed, seed + 1),
        }
    return {"by_category": rows, "overall": overall}


def _rate(xs: Sequence[bool | None]) -> float | None:
    vals = [bool(x) for x in xs if x is not None]
    return mean(vals) if vals else None


def _rate_sum(num: Sequence[int], den: Sequence[int]) -> float | None:
    d = sum(den)
    return (sum(num) / d) if d else None


def _paired(diffs: list[float], seed: int) -> dict[str, Any]:
    lo, hi = bootstrap_ci(diffs, seed)
    return {"mean": mean(diffs), "ci95": [lo, hi], "n": len(diffs)}


def to_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for r in summary["by_category"]:
        for metric in (
            "completion_rate",
            "achievable_completion_rate",
            "abstain_rate",
            "correct_abstain_rate",
            "invalid_calls_mean",
            "calls_mean",
            "false_suppression_rate",
        ):
            out.append(
                {
                    "baseline": r["baseline"],
                    "server": r["server"],
                    "category": r["category"],
                    "metric": metric,
                    "value": r[metric],
                    "n": r["n"],
                }
            )
    return out
