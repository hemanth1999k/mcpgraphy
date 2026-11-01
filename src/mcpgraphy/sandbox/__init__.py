"""Deterministic, resettable simulated MCP servers."""

from __future__ import annotations

from mcpgraphy.sandbox.base import (
    CallCounter,
    FixtureManifest,
    NoResetSandbox,
    RestrictedSandboxView,
    SandboxControl,
)
from mcpgraphy.sandbox.release_sim import ReleaseSandbox
from mcpgraphy.sandbox.ticket_sim import TicketSandbox


def make_sandbox(server_id: str, version: str = "v1") -> SandboxControl:
    if server_id == "release_sim":
        return ReleaseSandbox(version)  # type: ignore[arg-type]
    if server_id == "ticket_sim":
        return TicketSandbox(version)  # type: ignore[arg-type]
    raise KeyError(f"unknown server {server_id}")


__all__ = [
    "CallCounter",
    "FixtureManifest",
    "NoResetSandbox",
    "ReleaseSandbox",
    "RestrictedSandboxView",
    "SandboxControl",
    "TicketSandbox",
    "make_sandbox",
]
