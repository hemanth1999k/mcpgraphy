"""A real LLM agent on the benchmark scenarios, with and without mcpgraphy in front of the tools.

    OPENAI_API_KEY=... mcpgraphy benchmark llm --model gpt-5.4-mini --limit 40

Same scenarios, same simulated servers, same oracle as `benchmark run`. Two arms:

* `raw`: the model sees the server's tools and acts on them directly.
* `mcpgraphy`: the same tools behind the proxy's block-mode gate, using the passive-trace graph
  (what the shipped proxy learns from traffic), plus the `mcpgraphy_affordances` tool. Every
  tool is pre-approved, so the gate acts on evidence only.

Per task: completion, invalid calls (judged by the oracle before each call is executed), blocked
calls, tool calls, prompt / cached / completion tokens, dollar cost at list prices, wall time.
The model samples, so runs are not byte-identical; the model id and date are recorded.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcpgraphy.client_adapter import InMemoryToolClient
from mcpgraphy.evaluation.agents import Action
from mcpgraphy.evaluation.baselines import LearnedGuidance
from mcpgraphy.evaluation.metrics import bootstrap_ci, mean
from mcpgraphy.evaluation.runner import SuiteConfig, build_scenarios
from mcpgraphy.evaluation.tasks import Scenario
from mcpgraphy.graph.store import GraphStore
from mcpgraphy.learn.abstract import features_of_raw
from mcpgraphy.models import EffectClass, OutcomeKind, canonical_json, content_id
from mcpgraphy.offers.selector import inline_hint, resolve_state, routes_from, select_offers
from mcpgraphy.registry import get_spec
from mcpgraphy.sandbox import make_sandbox

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
# USD per 1M tokens: input, cached input, output. OpenAI list prices as of 2026-11-01.
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.5": (5.00, 0.50, 30.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
}
ARMS = ("raw", "mcpgraphy", "gate", "inline", "route")
GUIDED = {"mcpgraphy", "gate", "inline", "route"}
MAX_ROUNDS = 16  # model turns per task, a hard stop independent of the step budget

SYSTEM_RAW = (
    "You operate a {server} service through tools. Each tool call costs money, so be "
    "efficient. The resource is {resource_class} '{local_id}'. Goal: {goal}. Read the resource "
    "first with {obs}. When the goal is reached, reply exactly GOAL_REACHED. If the goal cannot "
    "be reached from the current state with the available tools, reply exactly "
    "GOAL_UNREACHABLE. Otherwise always call a tool."
)
SYSTEM_MCPGRAPHY = SYSTEM_RAW + (
    " mcpgraphy sits in front of the service: it refuses calls that were observed to fail from "
    "the resource's current state and tells you why. mcpgraphy_affordances(resource_ref) lists, "
    "for a resource's current state, which actions are candidates, uncertain, untested, or not "
    "supported from this state; use it before mutating."
)
SYSTEM_GATE = SYSTEM_RAW + (
    " mcpgraphy sits in front of the service: it refuses calls that were observed to fail from "
    "the resource's current state and tells you why."
)
SYSTEM_INLINE = SYSTEM_GATE + (
    " Every read of the resource ends with one line from mcpgraphy listing which actions are "
    "ok, not supported, or untested from the current state; use it to choose the next call."
)
SYSTEM_ROUTE = SYSTEM_GATE + (
    " Every read of the resource, and the result of every action on it, ends with lines from "
    "mcpgraphy: the resource's state, which actions are ok from it and what each changes, "
    "which are not supported, and routes to other states. Plan from those lines, act, and do "
    "not re-read the resource unless an action's result lacks them."
)
SYSTEMS = {
    "raw": SYSTEM_RAW,
    "route": SYSTEM_ROUTE,
    "mcpgraphy": SYSTEM_MCPGRAPHY,
    "gate": SYSTEM_GATE,
    "inline": SYSTEM_INLINE,
}
AFFORDANCES_TOOL = {
    "type": "function",
    "function": {
        "name": "mcpgraphy_affordances",
        "description": "Learned action offers for a resource's current observed state.",
        "parameters": {
            "type": "object",
            "properties": {
                "resource_ref": {"type": "string", "description": "<resource_class>:<id>"}
            },
            "required": ["resource_ref"],
        },
    },
}


@dataclass
class Usage:
    prompt: int = 0
    cached: int = 0
    completion: int = 0
    requests: int = 0

    def add(self, u: dict[str, Any]) -> None:
        self.prompt += int(u.get("prompt_tokens", 0))
        self.cached += int((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        self.completion += int(u.get("completion_tokens", 0))
        self.requests += 1

    def cost(self, model: str) -> float:
        p_in, p_cached, p_out = PRICES.get(model, (0.0, 0.0, 0.0))
        return (
            (self.prompt - self.cached) * p_in + self.cached * p_cached + self.completion * p_out
        ) / 1e6


@dataclass
class LLMEpisode:
    scenario_id: str
    arm: str
    result: str = "EXHAUSTED"  # COMPLETED | ABSTAINED | FALSE_DONE | EXHAUSTED | ERROR
    steps: list[dict[str, Any]] = field(default_factory=list)
    invalid_calls: int = 0
    blocked_calls: int = 0
    suppressed_valid: int = 0
    reads: int = 0
    guidance_calls: int = 0
    usage: Usage = field(default_factory=Usage)
    wall_s: float = 0.0
    abstain_was_correct: bool | None = None
    error: str | None = None
    final_text: str = ""
    turns: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, model: str) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "arm": self.arm,
            "result": self.result,
            "steps": self.steps,
            "calls": len(self.steps),
            "invalid_calls": self.invalid_calls,
            "blocked_calls": self.blocked_calls,
            "suppressed_valid": self.suppressed_valid,
            "reads": self.reads,
            "guidance_calls": self.guidance_calls,
            "prompt_tokens": self.usage.prompt,
            "cached_tokens": self.usage.cached,
            "completion_tokens": self.usage.completion,
            "total_tokens": self.usage.prompt + self.usage.completion,
            "requests": self.usage.requests,
            "cost_usd": round(self.usage.cost(model), 6),
            "wall_s": round(self.wall_s, 2),
            "abstain_was_correct": self.abstain_was_correct,
            "error": self.error,
            "final_text": self.final_text[:300],
            "turns": self.turns,
            "tokens_by_turn_kind": _tokens_by_kind(self.turns),
        }


def _tokens_by_kind(turns: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in turns:
        out[t["kind"]] = out.get(t["kind"], 0) + t["prompt"] + t["completion"]
    return out


def _post(payload: dict[str, Any], key: str) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(5):
        req = urllib.request.Request(OPENAI_URL, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"OpenAI HTTP {e.code}: {body[:300]}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt < 4:
                time.sleep(2**attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def _openai_tools(specs: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.input_schema,
            },
        }
        for s in specs
    ]


def _compact_offers(offers: list[Any]) -> list[dict[str, Any]]:
    out = []
    for o in offers:
        d = o.to_dict()
        beh = d.get("behavior") or {}
        out.append(
            {
                "tool": d.get("tool"),
                "availability": d.get("availability"),
                "status": beh.get("status"),
                "predicted_effects": beh.get("predicted_effects"),
                "requires_approval": (d.get("authorization") or {}).get("required"),
            }
        )
    return out


async def run_llm_episode(
    arm: str,
    scenario: Scenario,
    model: str,
    key: str,
    store: GraphStore | None,
    model_version: str | None,
) -> LLMEpisode:
    from mcpgraphy.bench.ground_truth import true_outcome

    ep = LLMEpisode(scenario.scenario_id, arm)
    t0 = time.perf_counter()
    sandbox = make_sandbox(scenario.server_id, scenario.server_version)
    ids = sandbox.fixtures(scenario.seed, scenario.fixture_index + 1)
    manifest = sandbox.reset_fixture(ids[scenario.fixture_index])
    spec_live = get_spec(scenario.server_id, scenario.server_version)
    spec_v1 = get_spec(scenario.server_id, "v1")
    local_id = manifest.focal_ref.local_id
    guidance = (
        LearnedGuidance("B4", spec_v1, store, model_version)
        if arm in GUIDED and store is not None and model_version
        else None
    )
    goal_text = ", ".join(f"{k} = {v}" for k, v in scenario.goal.items())
    system = SYSTEMS[arm].format(
        server=scenario.server_id,
        resource_class=spec_live.resource_class,
        local_id=local_id,
        goal=goal_text,
        obs=spec_live.observation_tool,
    )
    injected = False
    async with InMemoryToolClient(sandbox.mcp) as client:
        tools = _openai_tools(await client.list_tools())
        live_names = {t["function"]["name"] for t in tools}
        if arm == "mcpgraphy":
            tools.append(AFFORDANCES_TOOL)

        async def goal_met() -> bool:
            rec = await client.call(spec_live.observation_tool, {spec_live.id_param: local_id})
            return bool(rec.raw) and all(rec.raw.get(k) == v for k, v in scenario.goal.items())

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Reach the goal for {spec_live.resource_class} {local_id}.",
            },
        ]
        nudges = 0
        try:
            for _round in range(MAX_ROUNDS):
                payload = {
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                }
                resp = await asyncio.to_thread(_post, payload, key)
                usage = resp.get("usage") or {}
                ep.usage.add(usage)
                turn = {
                    "prompt": int(usage.get("prompt_tokens", 0)),
                    "completion": int(usage.get("completion_tokens", 0)),
                    "kind": "final",
                }
                ep.turns.append(turn)
                msg = resp["choices"][0]["message"]
                entry: dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
                if msg.get("tool_calls"):
                    entry["tool_calls"] = msg["tool_calls"]
                messages.append(entry)
                if not msg.get("tool_calls"):
                    text = (msg.get("content") or "").strip()
                    ep.final_text = text
                    verdict = _verdict(text)
                    if verdict == "IMPOSSIBLE":
                        ep.result = "ABSTAINED"
                        ep.abstain_was_correct = not scenario.achievable
                        break
                    if verdict == "DONE":
                        ep.result = "COMPLETED" if await goal_met() else "FALSE_DONE"
                        break
                    turn["kind"] = "nudge"
                    nudges += 1
                    if nudges > 1:
                        break
                    messages.append(
                        {
                            "role": "user",
                            "content": "Reply GOAL_REACHED, GOAL_UNREACHABLE, or call a tool.",
                        }
                    )
                    continue
                stop = False
                kinds: list[str] = []
                for tc in msg["tool_calls"]:
                    name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"] or "{}")
                    except ValueError:
                        args = None
                    before = (
                        len(ep.steps),
                        ep.invalid_calls,
                        ep.blocked_calls,
                        ep.reads,
                        ep.guidance_calls,
                    )
                    content = await _handle_call(
                        name,
                        args,
                        ep,
                        scenario,
                        sandbox,
                        manifest,
                        client,
                        spec_live,
                        spec_v1,
                        live_names,
                        guidance,
                        store,
                        model_version,
                        local_id,
                        true_outcome,
                        injected,
                    )
                    kinds.append(_turn_kind(before, ep))
                    if name == spec_live.observation_tool and not injected:
                        injected = True
                        if scenario.concurrent_op:
                            sandbox.inject_concurrent_update(
                                ids[scenario.fixture_index], scenario.concurrent_op
                            )
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content})
                    if len(ep.steps) >= scenario.max_steps:
                        stop = True
                turn["kind"] = _worst(kinds)
                if stop:
                    break
            if ep.result == "EXHAUSTED" and await goal_met():
                ep.result = "COMPLETED"
        except Exception as e:  # noqa: BLE001 - the run must survive one bad episode
            ep.result = "ERROR"
            ep.error = f"{type(e).__name__}: {e}"[:300]
    ep.wall_s = time.perf_counter() - t0
    return ep


async def _handle_call(
    name,
    args,
    ep,
    scenario,
    sandbox,
    manifest,
    client,
    spec_live,
    spec_v1,
    live_names,
    guidance,
    store,
    model_version,
    local_id,
    true_outcome,
    injected,
) -> str:
    if args is None:
        return "Error: tool arguments were not valid JSON."
    if name == "mcpgraphy_affordances":
        ep.guidance_calls += 1
        if store is None or not model_version:
            return json.dumps({"offers": [], "note": "nothing learned"})
        ref = str(args.get("resource_ref", local_id))
        rid = ref.split(":", 1)[1] if ":" in ref else ref
        resolved = await resolve_state(client, spec_v1, rid)
        offers = select_offers(store, spec_v1, model_version, resolved)
        return json.dumps(
            {
                "resource": rid,
                "exists": resolved.exists,
                "observed_state": dict(resolved.state.features),
                "offers": _compact_offers(offers),
            }
        )
    if name not in live_names:
        return f"Error: unknown tool {name!r}."
    policy = spec_live.tool(name)
    if policy.effect is EffectClass.READ_ONLY:
        ep.reads += 1
        rec = await client.call(name, args)
        text = _render(rec)
        if (
            ep.arm in ("inline", "route")
            and name == spec_live.observation_tool
            and store
            and model_version
        ):
            rid = str(args.get(spec_live.id_param, local_id))
            text += "\n" + await _hint(
                client, store, spec_v1, model_version, rid, ep.arm != "inline"
            )
        return text
    valid = true_outcome(sandbox, name, args) is OutcomeKind.OK
    if guidance is not None:
        focal = await client.call(spec_live.observation_tool, {spec_live.id_param: local_id})
        feats = (
            features_of_raw(spec_v1.resource_class, focal.raw) if focal.raw else {"exists": False}
        )
        label = (await guidance.rank(client, manifest, feats, [Action(name, args)]))[0]
        if label == "NOT_SUPPORTED":
            ep.blocked_calls += 1
            ep.suppressed_valid += int(valid)
            pretty = ", ".join(f"{k}={v}" for k, v in feats.items())
            msg = (
                f"mcpgraphy blocked this call: {name} was observed to fail from state "
                f"[{pretty}] in prior calls."
            )
            if ep.arm == "mcpgraphy":
                return msg + (
                    f" Call mcpgraphy_affordances('{spec_v1.resource_class}:"
                    f"{args.get(spec_v1.id_param, local_id)}') for alternatives."
                )
            rid = str(args.get(spec_v1.id_param, local_id))
            return (
                msg
                + "\n"
                + await _hint(client, store, spec_v1, model_version, rid, ep.arm == "route")
            )
    rec = await client.call(name, args)
    ep.steps.append({"tool": name, "args": args, "valid": valid, "outcome": str(rec.outcome.kind)})
    if not valid:
        ep.invalid_calls += 1
    text = _render(rec)
    if ep.arm in ("route",) and store and model_version:
        rid = str(args.get(spec_v1.id_param, local_id))
        text += "\n" + await _hint(client, store, spec_v1, model_version, rid, True)
    return text


async def _hint(client, store, spec, model_version, rid, with_routes: bool) -> str:
    """The resource's state looked up in the graph, with routes for the route arm."""
    resolved = await resolve_state(client, spec, rid)
    offers = select_offers(store, spec, model_version, resolved)
    routes = (
        routes_from(store, spec, model_version, resolved.state.state_id) if with_routes else None
    )
    return inline_hint(
        offers, spec, dict(resolved.state.features), routes=routes, compact=with_routes
    )


GIVE_UP = (
    "GOAL_UNREACHABLE",
    "UNREACHABLE",
    "IMPOSSIBLE",
    "CANNOT BE REACHED",
    "CAN'T BE REACHED",
    "NOT ACHIEVABLE",
    "NOT POSSIBLE",
)


def _verdict(text: str) -> str | None:
    """DONE, IMPOSSIBLE, or None for a message that is neither (the model gets one nudge).

    The keywords avoid every state name on the simulated servers (a ticket state is called
    DONE), so a state name in the reply can never be read as a verdict."""
    t = text.upper()
    if any(g in t for g in GIVE_UP):
        return "IMPOSSIBLE"
    if "GOAL_REACHED" in t or "GOAL REACHED" in t:
        return "DONE"
    return None


_KIND_RANK = ["invalid", "blocked", "action", "guidance", "read", "other"]


def _turn_kind(before: tuple[int, ...], ep: LLMEpisode) -> str:
    steps, invalid, blocked, reads, guidance = before
    if ep.invalid_calls > invalid:
        return "invalid"
    if ep.blocked_calls > blocked:
        return "blocked"
    if len(ep.steps) > steps:
        return "action"
    if ep.guidance_calls > guidance:
        return "guidance"
    if ep.reads > reads:
        return "read"
    return "other"


def _worst(kinds: list[str]) -> str:
    return min(kinds, key=_KIND_RANK.index) if kinds else "other"


def _render(rec: Any) -> str:
    if rec.outcome.kind.is_ok:
        return json.dumps(rec.raw) if rec.raw is not None else "ok"
    return f"Error {rec.outcome.kind}: {rec.outcome.message or rec.outcome.code or ''}".strip()


def summarize_llm(rows: list[dict[str, Any]], scenarios: dict[str, dict], model: str, seed: int):
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    out: dict[str, Any] = {"model": model, "arms": {}, "paired_vs_raw": {}}
    for arm, rs in sorted(by_arm.items()):
        ach = [r for r in rs if scenarios[r["scenario_id"]]["achievable"]]
        unach = [r for r in rs if not scenarios[r["scenario_id"]]["achievable"]]
        out["arms"][arm] = {
            "n": len(rs),
            "completion_achievable": mean([r["result"] == "COMPLETED" for r in ach])
            if ach
            else None,
            "correct_abstain_unachievable": mean([r["result"] == "ABSTAINED" for r in unach])
            if unach
            else None,
            "false_done_rate": mean([r["result"] == "FALSE_DONE" for r in rs]),
            "error_rate": mean([r["result"] == "ERROR" for r in rs]),
            "invalid_calls_per_task": mean([r["invalid_calls"] for r in rs]),
            "blocked_calls_per_task": mean([r["blocked_calls"] for r in rs]),
            "suppressed_valid_total": sum(r["suppressed_valid"] for r in rs),
            "tool_calls_per_task": mean(
                [r["calls"] + r["reads"] + r["guidance_calls"] for r in rs]
            ),
            "requests_per_task": mean([r["requests"] for r in rs]),
            "prompt_tokens_per_task": mean([r["prompt_tokens"] for r in rs]),
            "cached_tokens_per_task": mean([r["cached_tokens"] for r in rs]),
            "completion_tokens_per_task": mean([r["completion_tokens"] for r in rs]),
            "total_tokens_per_task": mean([r["total_tokens"] for r in rs]),
            "cost_usd_per_task": mean([r["cost_usd"] for r in rs]),
            "cost_usd_total": sum(r["cost_usd"] for r in rs),
            "wall_s_per_task": mean([r["wall_s"] for r in rs]),
            "tokens_by_turn_kind_per_task": {
                k: round(sum(r["tokens_by_turn_kind"].get(k, 0) for r in rs) / len(rs))
                for k in _KIND_RANK + ["nudge", "final"]
            },
            "tokens_per_completed_task": (
                sum(r["total_tokens"] for r in rs)
                / max(1, sum(r["result"] == "COMPLETED" for r in rs))
            ),
        }
    raw = {r["scenario_id"]: r for r in by_arm.get("raw", [])}
    for arm, rs in sorted(by_arm.items()):
        if arm == "raw":
            continue
        paired = [(r, raw[r["scenario_id"]]) for r in rs if r["scenario_id"] in raw]
        res = {}
        for metric in ("invalid_calls", "total_tokens", "cost_usd"):
            diffs = [a[metric] - b[metric] for a, b in paired]
            lo, hi = bootstrap_ci(diffs, seed)
            res[metric] = {
                "mean_diff": mean(diffs) if diffs else 0.0,
                "ci95": [lo, hi],
                "n": len(diffs),
            }
        out["paired_vs_raw"][arm] = res
    return out


def _wasted(s: dict[str, Any]) -> int:
    k = s["tokens_by_turn_kind_per_task"]
    return k["invalid"] + k["blocked"]


def render_llm_report(summary: dict[str, Any]) -> str:
    a = summary["arms"]
    lines = [
        f"# LLM agent: {summary['model']} ({summary.get('date', '')})",
        "",
        "| arm | n | completion (achievable) | correct abstain (unachievable) | "
        "invalid calls / task | blocked / task | tool calls / task | tokens / task | "
        "tokens on failed+blocked turns | tokens / completed task | cost / task | errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def f(x):
        return "" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))

    for arm, s in a.items():
        lines.append(
            f"| {arm} | {s['n']} | {f(s['completion_achievable'])} | "
            f"{f(s['correct_abstain_unachievable'])} | {f(s['invalid_calls_per_task'])} | "
            f"{f(s['blocked_calls_per_task'])} | {f(s['tool_calls_per_task'])} | "
            f"{s['total_tokens_per_task']:.0f} | "
            f"{_wasted(s)} | "
            f"{s['tokens_per_completed_task']:.0f} | ${s['cost_usd_per_task']:.4f} | "
            f"{f(s['error_rate'])} |"
        )
    for arm, p in summary["paired_vs_raw"].items():
        lines.append("")
        lines.append(
            f"Paired differences, {arm} minus raw (95% bootstrap CI, n={p['invalid_calls']['n']}):"
        )
        for m, v in p.items():
            lines.append(f"- {m}: {v['mean_diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    return "\n".join(lines) + "\n"


def sample_scenarios(scenarios: list[Scenario], n: int) -> list[Scenario]:
    """n scenarios spread round-robin across (server, category) cells, lowest seeds first."""
    cells: dict[tuple[str, int], list[Scenario]] = {}
    for s in scenarios:
        cells.setdefault((s.server_id, s.category), []).append(s)
    out: list[Scenario] = []
    i = 0
    while len(out) < n and any(len(c) > i for c in cells.values()):
        for key in sorted(cells):
            if len(cells[key]) > i and len(out) < n:
                out.append(cells[key][i])
        i += 1
    return sorted(out, key=lambda s: s.scenario_id)


def run_llm_suite(
    suite: str,
    model: str,
    out_root: str | Path,
    arms: list[str],
    limit: int | None,
    concurrency: int,
    seed: int = 42,
    sample: int | None = None,
) -> Path:
    from mcpgraphy.evaluation.passive import record_passive_corpus

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set (read from the environment only)")
    cfg = SuiteConfig.load(suite)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = content_id(
        "llm", {"suite": cfg.name, "model": model, "arms": arms, "limit": limit, "stamp": stamp}
    )
    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenarios(cfg, limit)
    if sample:
        scenarios = sample_scenarios(scenarios, sample)
    (run_dir / "scenarios.json").write_text(
        json.dumps([s.to_dict() for s in scenarios], indent=2, sort_keys=True)
    )
    db = run_dir / "models.sqlite"
    versions: dict[str, str] = {}
    if GUIDED & set(arms):
        L = cfg.learner
        for server in sorted({s.server_id for s in scenarios}):
            versions[server] = asyncio.run(
                record_passive_corpus(
                    make_sandbox(server, "v1"),
                    get_spec(server, "v1"),
                    db,
                    run_dir / f"journal_{server}_B4.jsonl",
                    seed=L["seed"],
                    fixtures=L["fixtures"],
                )
            )

    async def _all() -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(concurrency)
        with GraphStore(db) as store:

            async def one(sc: Scenario, arm: str) -> dict[str, Any]:
                async with sem:
                    ep = await run_llm_episode(
                        arm,
                        sc,
                        model,
                        key,
                        store if arm in GUIDED else None,
                        versions.get(sc.server_id),
                    )
                    row = {
                        **ep.to_dict(model),
                        "server_id": sc.server_id,
                        "category": sc.category,
                        "achievable": sc.achievable,
                    }
                    print(
                        f"  {sc.scenario_id:<22} {arm:<10} {ep.result:<10} "
                        f"invalid={ep.invalid_calls} blocked={ep.blocked_calls} "
                        f"tokens={ep.usage.prompt + ep.usage.completion} "
                        f"${ep.usage.cost(model):.4f}",
                        flush=True,
                    )
                    return row

            return await asyncio.gather(*[one(sc, arm) for sc in scenarios for arm in arms])

    t0 = time.perf_counter()
    rows = asyncio.run(_all())
    rows.sort(key=lambda r: (r["scenario_id"], r["arm"]))
    with open(run_dir / "results.jsonl", "w") as fh:
        for r in rows:
            fh.write(canonical_json(r) + "\n")
    summary = summarize_llm(rows, {s.scenario_id: s.to_dict() for s in scenarios}, model, seed)
    summary.update(
        {
            "run_id": run_id,
            "suite": cfg.name,
            "date": stamp,
            "scenarios": len(scenarios),
            "wall_s": round(time.perf_counter() - t0, 1),
            "prices_usd_per_1m": PRICES.get(model),
        }
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (run_dir / "report.md").write_text(render_llm_report(summary))
    (Path(out_root) / "latest").write_text(run_id)
    return run_dir
