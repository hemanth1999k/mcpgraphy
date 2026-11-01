"""Demo: mcpgraphy in front of the official filesystem MCP server (no credentials).

    python examples/filesystem_demo.py /tmp/mcpgraphy-demo     # pip install mcpgraphy first

Act 1: a naive agent works on files through the proxy in `warn` mode and makes the usual
mistakes (editing and moving files that are not there). The proxy learns from those calls only.
Act 2: the proxy restarts in `block` mode with what it learned; the same mistakes are now
stopped before they reach the server, overwriting an existing file needs approval, and
`mcpgraphy_affordances` shows what is safe to do with a file in its current state.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

from mcp import Client, StdioServerParameters

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="mcpgraphy-demo-"))
HERE = Path(__file__).resolve().parent
ADAPTER = HERE / "adapters" / "filesystem.json"


def proxy_params(policy: str, approve: str = "") -> StdioServerParameters:
    args = [
        "-m",
        "mcpgraphy",
        "wrap",
        "--adapter",
        str(ADAPTER),
        "--policy",
        policy,
        "--refresh-every",
        "2",
        "--db",
        str(ROOT / ".mcpgraphy.sqlite"),
    ]
    if approve:
        args += ["--approve", approve]
    args += ["--", "npx", "-y", "@modelcontextprotocol/server-filesystem", str(ROOT)]
    return StdioServerParameters(command=sys.executable, args=args)


def say(label: str, res) -> None:
    text = res.content[0].text if res.content else ""
    flag = (
        "BLOCKED"
        if res.is_error and "mcpgraphy blocked" in text
        else ("error" if res.is_error else "ok")
    )
    first = text.strip().splitlines()[0] if text.strip() else ""
    notes = [
        c.text
        for c in res.content[1:]
        if getattr(c, "type", "") == "text" and c.text.startswith("mcpgraphy:")
    ]
    print(f"  {label:<46} -> {flag:<7} {first[:70]}")
    for n in notes:
        print(f"      ! {n[:110]}")


async def act1() -> None:
    print("\nACT 1  warn mode: the agent works, mcpgraphy watches and learns\n")
    async with Client(proxy_params("warn")) as c:
        notes = ROOT / "notes.md"
        draft = ROOT / "draft.md"
        say(
            "edit_file notes.md (does not exist yet)",
            await c.call_tool(
                "edit_file", {"path": str(notes), "edits": [{"oldText": "a", "newText": "b"}]}
            ),
        )
        say(
            "move_file draft.md -> final.md (no draft yet)",
            await c.call_tool(
                "move_file", {"source": str(draft), "destination": str(ROOT / "final.md")}
            ),
        )
        say(
            "write_file notes.md",
            await c.call_tool(
                "write_file", {"path": str(notes), "content": "# Notes\n\ntodo: ship\n"}
            ),
        )
        say(
            "edit_file notes.md (now it exists)",
            await c.call_tool(
                "edit_file",
                {
                    "path": str(notes),
                    "edits": [{"oldText": "todo: ship", "newText": "todo: ship mcpgraphy"}],
                },
            ),
        )
        say(
            "write_file draft.md",
            await c.call_tool("write_file", {"path": str(draft), "content": "draft\n"}),
        )
        say(
            "move_file draft.md -> final.md",
            await c.call_tool(
                "move_file", {"source": str(draft), "destination": str(ROOT / "final.md")}
            ),
        )
        say(
            "edit_file gone.md (does not exist)",
            await c.call_tool(
                "edit_file",
                {"path": str(ROOT / "gone.md"), "edits": [{"oldText": "x", "newText": "y"}]},
            ),
        )
        status = (await c.call_tool("mcpgraphy_status", {})).structured_content
        print(
            f"\n  learned from {status['calls_recorded']} calls, warned {status['warned']}x, "
            f"model {status['model_version'][-16:]}"
        )


async def act2() -> None:
    print("\nACT 2  block mode, same knowledge: the mistakes stop before the server sees them\n")
    async with Client(proxy_params("block", approve="write_file")) as c:
        say(
            "edit_file missing.md",
            await c.call_tool(
                "edit_file",
                {"path": str(ROOT / "missing.md"), "edits": [{"oldText": "x", "newText": "y"}]},
            ),
        )
        say(
            "move_file missing.md -> x.md",
            await c.call_tool(
                "move_file", {"source": str(ROOT / "missing.md"), "destination": str(ROOT / "x.md")}
            ),
        )
        say(
            "edit_file notes.md (exists: allowed)",
            await c.call_tool(
                "edit_file",
                {
                    "path": str(ROOT / "notes.md"),
                    "edits": [{"oldText": "mcpgraphy", "newText": "mcpgraphy today"}],
                },
            ),
        )
        say(
            "move_file final.md -> done.md (not pre-approved)",
            await c.call_tool(
                "move_file",
                {"source": str(ROOT / "final.md"), "destination": str(ROOT / "done.md")},
            ),
        )
        aff = (
            await c.call_tool(
                "mcpgraphy_affordances", {"resource_ref": f"file:{ROOT / 'notes.md'}"}
            )
        ).structured_content
        print(f"\n  offers for notes.md  (observed: {aff['observed_state']})")
        for o in aff["offers"]:
            if o["tool"] in ("edit_file", "move_file", "write_file", "read_text_file"):
                flags = o["availability"] + (" +approval" if o["authorization"]["required"] else "")
                print(f"    {o['tool']:<16} {flags}")
        aff = (
            await c.call_tool("mcpgraphy_affordances", {"resource_ref": f"file:{ROOT / 'nope.md'}"})
        ).structured_content
        print(f"  offers for nope.md   (observed: {aff['observed_state']})")
        for o in aff["offers"]:
            if o["tool"] in ("edit_file", "move_file", "write_file"):
                print(f"    {o['tool']:<16} {o['availability']}")
        status = (await c.call_tool("mcpgraphy_status", {})).structured_content
        print(
            f"\n  blocked {status['blocked']} call(s); audit + graph in {ROOT / '.mcpgraphy.sqlite'}"
        )


async def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for p in ROOT.iterdir():
        p.unlink() if p.is_file() else shutil.rmtree(p)
    print(
        f"mcpgraphy demo  |  downstream: @modelcontextprotocol/server-filesystem  |  root: {ROOT}"
    )
    await act1()
    await act2()


if __name__ == "__main__":
    asyncio.run(main())
