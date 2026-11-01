"""mcpgraphy command line interface.

argparse only: no extra dependencies, and `main(argv)` is directly testable in-process.
Exit codes: 0 ok, 2 usage error, 3 exploration aborted (reset failure / budget).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from mcpgraphy import __version__
from mcpgraphy.registry import SERVER_IDS, get_spec

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ABORT = 3
DEFAULT_DB = os.environ.get("MCPGRAPHY_DB", "mcpgraphy.sqlite")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mcpgraphy",
        description="Learned, uncertainty-aware state-action graphs for MCP tools.",
    )
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    sub.add_parser("version", help="print the package version")

    insp = sub.add_parser("inspect", help="list a sandbox server's tools and declared metadata")
    _server_args(insp)
    insp.add_argument("--json", action="store_true", help="machine-readable output")

    learn = sub.add_parser(
        "learn", help="run safe, resettable experiments and materialize a behavior graph"
    )
    _server_args(learn)
    learn.add_argument("--fixtures", type=int, default=20, help="number of sandbox fixtures")
    learn.add_argument("--budget", type=int, default=150, help="max effectful probes")
    learn.add_argument("--seed", type=int, default=0)
    _db_args(learn)
    learn.add_argument("--journal", help="journal path (default: <db>.journal.jsonl)")
    learn.add_argument(
        "--base-model", help="model version to inherit from and retest (drift handling)"
    )
    learn.add_argument(
        "--ablation", choices=("no_counterexamples", "no_arg_classes"), help="learner ablation"
    )

    graph = sub.add_parser("graph", help="export a learned graph")
    gsub = graph.add_subparsers(dest="graph_command", metavar="SUBCOMMAND")
    gexp = gsub.add_parser("export", help="export as JSON (authoritative) or Mermaid")
    gexp.add_argument(
        "--server", required=True, help="server id (a simulator or any learned/wrapped server)"
    )
    gexp.add_argument("--model", default="latest")
    gexp.add_argument("--format", choices=("json", "mermaid"), default="json")
    gexp.add_argument("--out", help="write to a file instead of stdout")
    _db_args(gexp)

    offers = sub.add_parser("offers", help="action offers for a resource's current observed state")
    _server_args(offers)
    offers.add_argument("--resource", required=True, help="e.g. release:demo-7-0 or ticket:tk-7-0")
    offers.add_argument("--fixture-seed", type=int, help="override the seed parsed from the id")
    offers.add_argument("--goal", help="feature=value to promote, e.g. status=PUBLISHED")
    offers.add_argument("--model", default="latest")
    offers.add_argument("--json", action="store_true")
    _db_args(offers)

    serve = sub.add_parser("serve", help="run the action-offer MCP server")
    serve.add_argument("--transport", choices=("stdio",), default="stdio")
    _server_args(serve)
    serve.add_argument("--fixture-seed", type=int, default=0)
    serve.add_argument("--fixture-index", type=int, default=0)
    serve.add_argument("--model", default="latest")
    serve.add_argument(
        "--execute-policy",
        choices=("deny", "read_only", "approved"),
        default="deny",
        help="host policy for execute_offered_action (default deny: no effectful execution)",
    )
    serve.add_argument(
        "--approve", default="", help="comma list of tools the host pre-approves (approved mode)"
    )
    serve.add_argument(
        "--allow-uncertain", action="store_true", help="let UNCERTAIN_CANDIDATE offers execute"
    )
    _db_args(serve)

    bench = sub.add_parser("benchmark", help="run or report the deterministic benchmark")
    bsub = bench.add_subparsers(dest="bench_command", metavar="SUBCOMMAND")
    brun = bsub.add_parser("run", help="learn models, run all scenarios x baselines, write results")
    brun.add_argument("--suite", default="stateful-v1", help="suite name or path to its JSON")
    brun.add_argument("--seed", type=int, default=42)
    brun.add_argument("--baselines", help="comma list, e.g. B0,B2,B3,B4,B5 (default: suite)")
    brun.add_argument("--ablations", help="comma list (default: suite); '' for none")
    brun.add_argument("--out", default="experiments/runs")
    brun.add_argument("--limit", type=int, help="only the first N scenarios (smoke runs)")
    brep = bsub.add_parser("report", help="print the report of a run")
    brep.add_argument("--run", default="latest")
    brep.add_argument("--out", default="experiments/runs")
    bllm = bsub.add_parser(
        "llm",
        help="a real LLM agent on the scenarios, raw vs behind mcpgraphy (needs OPENAI_API_KEY)",
    )
    bllm.add_argument("--suite", default="stateful-v1")
    bllm.add_argument("--model", default="gpt-5.4-mini")
    bllm.add_argument("--arms", default="raw,gate,inline,mcpgraphy")
    bllm.add_argument("--limit", type=int, help="only the first N scenarios")
    bllm.add_argument("--concurrency", type=int, default=6)
    bllm.add_argument("--out", default="experiments/llm")

    sbx = sub.add_parser(
        "sandbox", help="run a simulated server itself over stdio (for demos/tests)"
    )
    _server_args(sbx)
    sbx.add_argument("--fixture-seed", type=int, default=0)
    sbx.add_argument("--fixture-index", type=int, default=0)

    wrap = sub.add_parser(
        "wrap",
        help="proxy mode: sit between an agent and a real MCP server, learn from its calls, "
        "serve offers, warn or block",
    )
    wrap.add_argument("--adapter", required=True, help="adapter JSON (observation tool, id param)")
    wrap.add_argument("--policy", choices=("observe", "warn", "block"), default="warn")
    wrap.add_argument(
        "--approve", default="", help="comma list of tools pre-approved in block mode"
    )
    wrap.add_argument("--refresh-every", type=int, default=5, help="re-learn after N calls")
    wrap.add_argument(
        "--no-hints",
        action="store_true",
        help="do not append the one-line affordance summary to read-backs",
    )
    wrap.add_argument("--url", help="downstream Streamable HTTP URL instead of a command")
    wrap.add_argument("downstream", nargs=argparse.REMAINDER, help="-- command args... (stdio)")
    _db_args(wrap)
    wrap.add_argument("--journal", help="journal path (default: <db>.journal.jsonl)")
    return p


def _server_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--server", required=True, choices=SERVER_IDS)
    p.add_argument("--version", dest="server_version", default="v1", choices=("v1", "v2"))


def _db_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", default=DEFAULT_DB, help=f"sqlite path (default {DEFAULT_DB})")


def main(argv: list[str] | None = None) -> int:
    logging.getLogger("mcp").setLevel(logging.ERROR)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_OK
    if args.command == "version":
        print(__version__)
        return EXIT_OK
    if args.command == "inspect":
        return cmd_inspect(args)
    if args.command == "learn":
        return cmd_learn(args)
    if args.command == "graph":
        if args.graph_command != "export":
            parser.parse_args(["graph", "--help"])
            return EXIT_USAGE
        return cmd_graph_export(args)
    if args.command == "offers":
        return cmd_offers(args)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "sandbox":
        return cmd_sandbox(args)
    if args.command == "wrap":
        return cmd_wrap(args)
    if args.command == "benchmark":
        if args.bench_command == "run":
            return cmd_benchmark_run(args)
        if args.bench_command == "report":
            return cmd_benchmark_report(args)
        if args.bench_command == "llm":
            return cmd_benchmark_llm(args)
        parser.parse_args(["benchmark", "--help"])
        return EXIT_USAGE
    parser.error(f"unknown command {args.command}")
    return EXIT_USAGE  # pragma: no cover


# ---------------------------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------------------------


def cmd_inspect(args: argparse.Namespace) -> int:
    report = asyncio.run(inspect_server(args.server, args.server_version))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return EXIT_OK
    print(f"{report['server_id']} {report['version']}  fingerprint={report['fingerprint']}")
    print(f"observation tool: {report['observation_tool']}  resource: {report['resource_class']}")
    for t in report["tools"]:
        flags = [t["effect_class"]]
        if t["approval_required"]:
            flags.append("APPROVAL_REQUIRED")
        params = ", ".join(t["params"])
        print(f"  {t['name']}({params})  [{' '.join(flags)}]")
        print(f"      {t['description']}")
    print("note: effect classes and approval flags are declared hints, not learned behavior.")
    return EXIT_OK


async def inspect_server(server_id: str, version: str) -> dict[str, Any]:
    from mcpgraphy.client_adapter import InMemoryToolClient
    from mcpgraphy.models import ServerVersion
    from mcpgraphy.sandbox import make_sandbox

    spec = get_spec(server_id, version)
    sandbox = make_sandbox(server_id, version)
    async with InMemoryToolClient(sandbox.mcp) as client:
        tools = await client.list_tools()
    sv = ServerVersion.build(server_id, version, tools)
    return {
        "server_id": server_id,
        "version": version,
        "fingerprint": sv.fingerprint,
        "server_version_id": sv.server_version_id,
        "observation_tool": spec.observation_tool,
        "resource_class": spec.resource_class,
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "params": sorted(t.input_schema.get("properties", {})),
                "input_schema": t.input_schema,
                "effect_class": str(spec.tool(t.name).effect),
                "approval_required": spec.tool(t.name).approval_required,
            }
            for t in tools
        ],
    }


# ---------------------------------------------------------------------------------------------
# learn
# ---------------------------------------------------------------------------------------------


def cmd_learn(args: argparse.Namespace) -> int:
    from mcpgraphy.learn.pipeline import run_learn

    journal = args.journal or f"{args.db}.journal.jsonl"
    result = asyncio.run(
        run_learn(
            server_id=args.server,
            version=args.server_version,
            seed=args.seed,
            fixtures=args.fixtures,
            budget=args.budget,
            db_path=args.db,
            journal_path=journal,
            base_model=args.base_model,
            ablation=args.ablation,
        )
    )
    s = result.summary
    print(f"model_version: {result.model_version}")
    print(f"run_id:        {s.run_id}")
    fp = s.server_version.fingerprint
    print(f"server:        {args.server} {args.server_version} fingerprint={fp}")
    print(f"experiments:   {s.experiments}  phases={s.phases}")
    b = s.budget
    print(
        f"budget:        {b['used']}/{b['total']} probes, {b['resets']} resets, "
        f"{b['skipped']} skipped, by_effect={b['by_effect']}"
    )
    print(f"journal:       {result.journal_path}")
    print(f"db:            {result.db_path}")
    if s.aborted:
        print(f"ABORTED:       {s.aborted}")
        return EXIT_ABORT
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# graph export
# ---------------------------------------------------------------------------------------------


def cmd_graph_export(args: argparse.Namespace) -> int:
    from mcpgraphy.graph.export import dumps, export_json, export_mermaid
    from mcpgraphy.graph.store import GraphStore

    with GraphStore(args.db) as store:
        try:
            mv = store.resolve_model_version(args.server, args.model)
        except LookupError as e:
            print(f"error: {e}", file=sys.stderr)
            return EXIT_USAGE
        data = export_json(store, mv, strip_timestamps=True)
    text = dumps(data) if args.format == "json" else export_mermaid(data)
    if args.out:
        Path(args.out).write_text(text + ("\n" if not text.endswith("\n") else ""))
        print(f"wrote {args.out} ({mv})", file=sys.stderr)
    else:
        print(text)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# offers / serve
# ---------------------------------------------------------------------------------------------


def cmd_offers(args: argparse.Namespace) -> int:
    from mcpgraphy.mcp_server import OfferServer
    from mcpgraphy.offers import load_demo_fixture
    from mcpgraphy.sandbox import make_sandbox

    sandbox = make_sandbox(args.server, args.server_version)
    load_demo_fixture(sandbox, args.resource, fixture_seed=args.fixture_seed)
    server = OfferServer(args.db, {args.server: sandbox}, args.model)
    try:
        result = asyncio.run(server.list_affordances(args.server, args.resource, args.goal))
    except Exception as e:  # ToolError carries the NOT_FOUND reason
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
        return EXIT_OK
    state = ", ".join(f"{k}={v}" for k, v in result.observed_state.items())
    print(f"Resource: {result.resource_ref}   (server {args.server} {args.server_version})")
    print(f"Observed state: {state}" if result.exists else "Observed state: NOT FOUND")
    print(
        f"Model: {result.model_version}"
        + ("   [schema drift!]" if result.fingerprint_drift else "")
    )
    print()
    print("Action offers:")
    for i, o in enumerate(result.offers, 1):
        flags = [str(o.availability), str(o.behavior.status)]
        if o.authorization.required:
            flags.append("APPROVAL_REQUIRED")
        argv_ = ", ".join(f"{k}={v}" for k, v in o.argument_template.items())
        print(f"  {i}. {o.tool}({argv_})  [{'; '.join(flags)}]")
        if o.behavior.required_predicates:
            print(f"       needs: {', '.join(o.behavior.required_predicates)}")
        if o.behavior.predicted_effects:
            print(f"       predicted: {', '.join(o.behavior.predicted_effects)}")
        if o.uncertainty.alternative_outcomes:
            alts = ", ".join(
                f"{a.outcome_kind} x{a.evidence_count}" for a in o.uncertainty.alternative_outcomes
            )
            print(f"       observed outcomes: {alts}")
        print(
            f"       evidence: {len(o.behavior.evidence_ids)} experiment(s)   offer_id={o.offer_id}"
        )
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    from mcpgraphy.mcp_server import build_offer_server
    from mcpgraphy.offers.executor import ExecutionPolicy
    from mcpgraphy.sandbox import make_sandbox

    sandbox = make_sandbox(args.server, args.server_version)
    ids = sandbox.fixtures(args.fixture_seed, args.fixture_index + 1)
    sandbox.reset_fixture(ids[args.fixture_index])
    # The approval token is a host-configured confirmation secret; it is read from the
    # environment only, never from the command line (process lists are world-readable).
    policy = ExecutionPolicy(
        mode=args.execute_policy,
        approved_tools=frozenset(t for t in args.approve.split(",") if t),
        approval_token=os.environ.get("MCPGRAPHY_APPROVAL_TOKEN") or None,
        allow_uncertain=args.allow_uncertain,
    )
    server = build_offer_server(args.db, {args.server: sandbox}, args.model, policy)
    server.run(args.transport)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# sandbox / wrap (proxy mode)
# ---------------------------------------------------------------------------------------------


def cmd_sandbox(args: argparse.Namespace) -> int:
    from mcpgraphy.sandbox import make_sandbox

    sandbox = make_sandbox(args.server, args.server_version)
    ids = sandbox.fixtures(args.fixture_seed, args.fixture_index + 1)
    sandbox.reset_fixture(ids[args.fixture_index])
    sandbox.mcp.run("stdio")
    return EXIT_OK


def cmd_wrap(args: argparse.Namespace) -> int:
    from mcp import StdioServerParameters

    from mcpgraphy.proxy import Proxy, ProxyPolicy, serve_stdio
    from mcpgraphy.proxy.adapter import AdapterSpec

    cmd = [c for c in args.downstream if c != "--"]
    if bool(args.url) == bool(cmd):
        print("error: give either --url URL or -- command args...", file=sys.stderr)
        return EXIT_USAGE
    downstream = args.url or StdioServerParameters(command=cmd[0], args=cmd[1:])
    adapter = AdapterSpec.load(args.adapter)
    policy = ProxyPolicy(
        mode=args.policy,
        approved_tools=frozenset(t for t in args.approve.split(",") if t),
        hints=not args.no_hints,
    )
    proxy = Proxy(
        adapter,
        downstream,
        db_path=args.db,
        journal_path=args.journal or f"{args.db}.journal.jsonl",
        policy=policy,
        refresh_every=args.refresh_every,
    )
    asyncio.run(serve_stdio(proxy))
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------------------------


def cmd_benchmark_run(args: argparse.Namespace) -> int:
    from mcpgraphy.evaluation.runner import SuiteConfig, run_suite

    cfg = SuiteConfig.load(args.suite)
    baselines = args.baselines.split(",") if args.baselines else None
    ablations = None if args.ablations is None else [a for a in args.ablations.split(",") if a]
    run_dir = run_suite(cfg, args.seed, args.out, baselines, ablations, args.limit)
    summary = json.loads((run_dir / "summary.json").read_text())
    timing = json.loads((run_dir / "timing.json").read_text())
    print(f"run_id:    {summary['run_id']}")
    print(f"scenarios: {summary['scenarios']}  baselines: {', '.join(summary['baselines'])}")
    print(f"results:   {run_dir / 'results.jsonl'}")
    print(f"report:    {run_dir / 'report.md'}")
    print(
        f"wall time: {timing['total_s']:.1f}s (models {timing['models_s']:.1f}s, "
        f"episodes {timing['episodes_s']:.1f}s)"
    )
    return EXIT_OK


def cmd_benchmark_llm(args: argparse.Namespace) -> int:
    from mcpgraphy.evaluation.llm_agent import run_llm_suite

    run_dir = run_llm_suite(
        args.suite, args.model, args.out, args.arms.split(","), args.limit, args.concurrency
    )
    print((run_dir / "report.md").read_text())
    print(f"results: {run_dir / 'results.jsonl'}")
    return EXIT_OK


def cmd_benchmark_report(args: argparse.Namespace) -> int:
    from mcpgraphy.evaluation.runner import resolve_run

    try:
        run_dir = resolve_run(args.out, args.run)
    except (LookupError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    print((run_dir / "report.md").read_text())
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
