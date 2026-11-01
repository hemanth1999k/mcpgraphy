"""Shared fixtures. Every test runs with network access disabled."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from mcpgraphy.sandbox import ReleaseSandbox, TicketSandbox


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise RuntimeError("network access is forbidden in tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.fixture
def release_sandbox() -> ReleaseSandbox:
    return ReleaseSandbox("v1")


@pytest.fixture
def release_sandbox_v2() -> ReleaseSandbox:
    return ReleaseSandbox("v2")


@pytest.fixture
def ticket_sandbox() -> TicketSandbox:
    return TicketSandbox("v1")


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "graph.sqlite"


@pytest.fixture
def tmp_journal(tmp_path: Path) -> Path:
    return tmp_path / "journal.jsonl"
