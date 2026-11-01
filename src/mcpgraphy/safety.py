"""Deterministic safety policy and experiment budgets.

This is code, not a natural-language instruction. The explorer asks `SafetyPolicy.check()` before
every effectful probe and `Budget` accounts for every probe and reset. A failed reset raises
`ResetError`, which must abort exploration (never "continue and hope").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mcpgraphy.models import EffectClass


class ResetError(RuntimeError):
    """The sandbox could not prove a reset happened. Exploration must stop."""


class BudgetExhausted(RuntimeError):
    """No probes left."""


@dataclass
class Budget:
    total: int
    destructive_cap_fraction: float = 0.2
    used: int = 0
    resets: int = 0
    by_effect: dict[str, int] = field(default_factory=dict)
    skipped: int = 0

    @property
    def remaining(self) -> int:
        return max(self.total - self.used, 0)

    @property
    def destructive_cap(self) -> int:
        return int(self.total * self.destructive_cap_fraction)

    def would_exceed(self, effect: EffectClass) -> str | None:
        """Return a reason string if a probe of this effect class is not allowed now."""
        if effect is EffectClass.READ_ONLY:
            return None
        if self.remaining <= 0:
            return "budget exhausted"
        if (
            effect is EffectClass.DESTRUCTIVE
            and self.by_effect.get(effect, 0) >= self.destructive_cap
        ):
            return f"destructive cap reached ({self.destructive_cap})"
        return None

    def consume(self, effect: EffectClass) -> None:
        if effect is EffectClass.READ_ONLY:
            return
        self.used += 1
        self.by_effect[effect] = self.by_effect.get(effect, 0) + 1

    def count_reset(self) -> None:
        self.resets += 1

    def to_dict(self) -> dict[str, int | float | dict[str, int]]:
        return {
            "total": self.total,
            "used": self.used,
            "resets": self.resets,
            "skipped": self.skipped,
            "destructive_cap": self.destructive_cap,
            "by_effect": {str(k): v for k, v in sorted(self.by_effect.items())},
        }


@dataclass(frozen=True)
class SafetyPolicy:
    """Allow-list based policy: only declared tools, only resettable sandboxes, bounded effects."""

    allowed_tools: frozenset[str]
    require_reset: bool = True

    def check(
        self, tool: str, effect: EffectClass, budget: Budget, reset_supported: bool
    ) -> str | None:
        if tool not in self.allowed_tools:
            return f"tool {tool} not in allow-list"
        if effect is not EffectClass.READ_ONLY and self.require_reset and not reset_supported:
            return "effectful probe without a proven reset is forbidden"
        return budget.would_exceed(effect)
