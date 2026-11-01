"""The learner cannot see the oracle or hidden state."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from mcpgraphy.sandbox import ReleaseSandbox, RestrictedSandboxView

SRC = Path(__file__).resolve().parents[2] / "src" / "mcpgraphy"
# The evaluation harness (scenario generation, validity judging, the B2 oracle baseline) must
# see the ground truth by definition. Everything else in the package must not.
HARNESS = ("evaluation", "bench")
HIDDEN_ATTRS = {
    "world",
    "_specs",
    "rules",
    "locked",
    "tested_version",
    "write_denied",
    "inject_concurrent_update",
}


def _modules():
    return sorted(p for p in SRC.rglob("*.py") if not set(HARNESS) & set(p.relative_to(SRC).parts))


def test_no_learner_module_imports_bench():
    offenders = []
    for path in _modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n == "mcpgraphy.bench" or n.startswith("mcpgraphy.bench.") for n in names):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert offenders == []


def test_learner_modules_never_touch_hidden_attrs():
    offenders = []
    for path in _modules():
        if path.parent.name == "sandbox":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in HIDDEN_ATTRS:
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno} .{node.attr}")
    assert offenders == []


def test_learner_runtime_never_loads_the_oracle_package():
    """A learn run plus the offer server in a fresh interpreter must not import `benchmarks`."""
    code = (
        "import asyncio, sys, tempfile, pathlib\n"
        "from mcpgraphy.learn.pipeline import run_learn\n"
        "from mcpgraphy.mcp_server import build_offer_server\n"
        "from mcpgraphy.sandbox import make_sandbox\n"
        "import mcpgraphy.cli\n"
        "d = pathlib.Path(tempfile.mkdtemp())\n"
        "r = asyncio.run(run_learn('release_sim', 'v1', 0, 4, 20, d / 'g.sqlite', d / 'j.jsonl'))\n"
        "build_offer_server(str(d / 'g.sqlite'), {'release_sim': make_sandbox('release_sim')})\n"
        "bad = sorted(m for m in sys.modules if m.startswith('mcpgraphy.bench'))\n"
        "assert not bad, bad\n"
        "print('clean', r.summary.experiments)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.startswith("clean")


def test_restricted_view_hides_world_and_control_internals():
    sb = ReleaseSandbox()
    view = RestrictedSandboxView(sb)
    assert view.server_id == "release_sim"
    ids = view.fixtures(0, 1)
    manifest = view.reset_fixture(ids[0])
    assert manifest.focal_ref.local_id == "demo-0-0"
    for name in ("world", "_specs", "rules", "inject_concurrent_update"):
        with pytest.raises(AttributeError):
            getattr(view, name)
    with pytest.raises(AttributeError):
        view.server_id = "x"
