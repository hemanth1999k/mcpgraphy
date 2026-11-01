"""The single module that touches MCP SDK result shapes.

Verified against mcp 2.3.0:
  * `Client(server)` runs in-memory with no transport.
  * `call_tool` returns `CallToolResult` with `is_error`, `content`, `structured_content`.
  * A `ToolError` inside a tool arrives as `is_error=True` with a text block
    "Error executing tool <name>: <message>"; pydantic argument validation errors look the same.
  * Only typed (pydantic model) returns populate `structured_content`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mcp import Client
from mcp.server import MCPServer

from mcpgraphy.models import Outcome, ToolSpec
from mcpgraphy.observe.normalize import normalize_error, ok_outcome


@dataclass(frozen=True)
class ToolCallRecord:
    tool: str
    args: dict[str, Any]
    outcome: Outcome
    raw: Any


def record_from_result(tool: str, args: dict[str, Any], result: Any) -> ToolCallRecord:
    """Normalize a CallToolResult from any transport into a ToolCallRecord."""
    text = _first_text(result.content)
    if result.is_error:
        return ToolCallRecord(tool=tool, args=dict(args), outcome=normalize_error(text), raw=None)
    raw: Any = result.structured_content
    if raw is None and text:
        try:
            raw = json.loads(text)
        except ValueError:
            raw = {"text": text}
    return ToolCallRecord(tool=tool, args=dict(args), outcome=ok_outcome(), raw=raw)


def tool_specs(result: Any) -> list[ToolSpec]:
    specs = []
    for t in result.tools:
        ann = t.annotations.model_dump(mode="json") if t.annotations else {}
        specs.append(
            ToolSpec(
                name=t.name,
                description=t.description or "",
                input_schema=t.input_schema,
                annotations=ann,
            )
        )
    return sorted(specs, key=lambda s: s.name)


class RemoteToolClient:
    """`.call()` / `.list_tools()` over an already-entered `mcp.Client` of any transport."""

    def __init__(self, client: Client) -> None:
        self.client = client

    async def list_tools(self) -> list[ToolSpec]:
        return tool_specs(await self.client.list_tools())

    async def call(self, tool: str, args: dict[str, Any]) -> ToolCallRecord:
        return record_from_result(tool, args, await self.client.call_tool(tool, args))


class InMemoryToolClient:
    """Async context manager around an in-memory MCP client session."""

    def __init__(self, server: MCPServer) -> None:
        self._server = server
        self._client: Client | None = None

    async def __aenter__(self) -> InMemoryToolClient:
        self._client = Client(self._server)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        assert self._client is not None
        await self._client.__aexit__(*exc)
        self._client = None

    @property
    def client(self) -> Client:
        if self._client is None:
            raise RuntimeError("client not entered")
        return self._client

    async def list_tools(self) -> list[ToolSpec]:
        result = await self.client.list_tools()
        specs = []
        for t in result.tools:
            ann = t.annotations.model_dump(mode="json") if t.annotations else {}
            specs.append(
                ToolSpec(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.input_schema,
                    annotations=ann,
                )
            )
        return sorted(specs, key=lambda s: s.name)

    async def call(self, tool: str, args: dict[str, Any]) -> ToolCallRecord:
        result = await self.client.call_tool(tool, args)
        text = _first_text(result.content)
        if result.is_error:
            return ToolCallRecord(
                tool=tool, args=dict(args), outcome=normalize_error(text), raw=None
            )
        raw: Any = result.structured_content
        if raw is None and text:
            try:
                raw = json.loads(text)
            except ValueError:
                raw = {"text": text}
        return ToolCallRecord(tool=tool, args=dict(args), outcome=ok_outcome(), raw=raw)


def _first_text(content: list[Any]) -> str:
    for block in content:
        if getattr(block, "type", None) == "text":
            return str(block.text)
    return ""
