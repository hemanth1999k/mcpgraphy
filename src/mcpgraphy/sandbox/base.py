"""Sandbox control API shared by the simulators.

The control API (`fixtures`, `reset_fixture`, `inject_concurrent_update`) is plain Python and is
never registered as an MCP tool. The learner only ever sees a `RestrictedSandboxView`, which hides
the world object and therefore every hidden flag.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from mcp.server import MCPServer

from mcpgraphy.models import Frozen, ResourceRef
from mcpgraphy.safety import ResetError


class FixtureManifest(Frozen):
    """What the learner is told after a reset: where the focal resource is and which ids exist
    for each argument class. No hidden state is ever included."""

    fixture_id: str
    server_id: str
    version: str
    focal_ref: ResourceRef
    expected_visible: dict[str, Any]
    id_pools: dict[str, list[str]]


def fixture_id_for(server_id: str, version: str, seed: int, index: int) -> str:
    h = hashlib.sha256(f"{server_id}|{version}|{seed}|{index}".encode()).hexdigest()
    return f"fx_{h[:12]}"


@dataclass
class CallCounter:
    calls: dict[str, int] = field(default_factory=dict)
    resets: int = 0

    def hit(self, tool: str) -> None:
        self.calls[tool] = self.calls.get(tool, 0) + 1

    @property
    def total(self) -> int:
        return sum(self.calls.values())

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": dict(sorted(self.calls.items())),
            "total": self.total,
            "resets": self.resets,
        }

    def clear(self) -> None:
        self.calls.clear()
        self.resets = 0


@runtime_checkable
class SandboxControl(Protocol):
    server_id: str
    version: str
    mcp: MCPServer
    counter: CallCounter

    def fixtures(self, seed: int, n: int) -> list[str]: ...

    def reset_fixture(self, fixture_id: str) -> FixtureManifest: ...

    def inject_concurrent_update(self, fixture_id: str, op: str) -> None: ...

    @property
    def reset_supported(self) -> bool: ...


class RestrictedSandboxView:
    """The only handle the learner receives. Anything beyond these names raises."""

    __slots__ = ("_inner",)
    _ALLOWED = frozenset(
        {"server_id", "version", "mcp", "counter", "fixtures", "reset_fixture", "reset_supported"}
    )

    def __init__(self, inner: SandboxControl) -> None:
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name: str) -> Any:
        if name not in self._ALLOWED:
            raise AttributeError(f"learner may not access sandbox.{name}")
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("sandbox view is read-only")


class NoResetSandbox:
    """Wraps a sandbox whose reset must be treated as unavailable (benchmark category 10)."""

    def __init__(self, inner: SandboxControl) -> None:
        self._inner = inner
        self.server_id = inner.server_id
        self.version = inner.version
        self.mcp = inner.mcp
        self.counter = inner.counter

    @property
    def reset_supported(self) -> bool:
        return False

    def fixtures(self, seed: int, n: int) -> list[str]:
        return self._inner.fixtures(seed, n)

    def reset_fixture(self, fixture_id: str) -> FixtureManifest:
        raise ResetError("reset not supported by this sandbox")

    def inject_concurrent_update(self, fixture_id: str, op: str) -> None:
        self._inner.inject_concurrent_update(fixture_id, op)


def counted(counter: CallCounter, tool: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a tool implementation so every invocation (success or failure) is counted."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        counter.hit(tool)
        return fn(*args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    wrapper.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    return wrapper
