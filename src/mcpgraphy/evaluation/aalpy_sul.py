"""B3: off-the-shelf active automata learning (AALpy L*) under the same probe budget.

The SUL resets to ONE canonical fixture (a clean untested draft / an assigned open ticket) because
L* needs a single initial state. Inputs are the core action classes (concretized against the
canonical fixture) plus an `observe` symbol whose output is the visible feature tuple, so the
learned Mealy machine exposes observable state. Outputs of action inputs are outcome kinds.

What this baseline cannot do by construction, and we report rather than hide: it never sees
fixtures with hidden flags, so it is confidently wrong on hidden-state categories; it cannot
start from arbitrary states (e.g. ARCHIVED), so those stay UNKNOWN; and it has no notion of
evidence counts or contradiction.

Threading note: AALpy's learner is synchronous while the MCP client is async. The client stays
in one coroutine (anyio cancel scopes are task-bound) and L* runs in a worker thread that
submits each tool call back to the event loop.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from mcpgraphy.client_adapter import InMemoryToolClient
from mcpgraphy.evaluation.baselines import TraceGuidance, _fkey
from mcpgraphy.learn.abstract import features_of_raw
from mcpgraphy.models import ActionClass, EffectClass
from mcpgraphy.registry import ServerSpec, concretize
from mcpgraphy.safety import BudgetExhausted
from mcpgraphy.sandbox.base import SandboxControl

OBSERVE = "observe"
Bridge = Callable[[Coroutine[Any, Any, Any]], Any]


@dataclass
class MealyResult:
    converged: bool
    probes: int
    resets: int
    states: int
    error: str | None
    table: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    automaton: Any = None


class SandboxSUL:
    """AALpy SUL over a sandbox. aalpy is imported lazily."""

    def __init__(
        self,
        sandbox: SandboxControl,
        spec: ServerSpec,
        fixture_id: str,
        budget: int,
        alphabet: dict[str, ActionClass],
        client: InMemoryToolClient,
        bridge: Bridge,
    ) -> None:
        from aalpy.base import SUL

        outer = self

        class _SUL(SUL):
            def pre(self) -> None:
                outer._pre()

            def post(self) -> None:
                pass

            def step(self, letter):
                return outer._step(letter)

        self.sandbox = sandbox
        self.spec = spec
        self.fixture_id = fixture_id
        self.budget = budget
        self.alphabet = alphabet
        self.client = client
        self.bridge = bridge
        self.probes = 0
        self.resets = 0
        self.manifest = None
        self.table: dict[tuple[str, str], set[str]] = {}
        self.sul = _SUL()

    def _pre(self) -> None:
        self.manifest = self.sandbox.reset_fixture(self.fixture_id)
        self.resets += 1

    def _observe(self) -> str:
        rec = self.bridge(
            self.client.call(
                self.spec.observation_tool, {self.spec.id_param: self.manifest.focal_ref.local_id}
            )
        )
        feats = features_of_raw(self.spec.resource_class, rec.raw) if rec.raw else {"exists": False}
        return _fkey(feats)

    def _step(self, letter: str) -> str:
        if letter == OBSERVE:
            return self._observe()
        ac = self.alphabet[letter]
        if ac.effect_class is not EffectClass.READ_ONLY:
            if self.probes >= self.budget:
                raise BudgetExhausted("B3 probe budget exhausted")
            self.probes += 1
        before = self._observe()
        args = concretize(self.manifest, ac)
        rec = self.bridge(self.client.call(ac.tool, args))
        out = str(rec.outcome.kind)
        self.table.setdefault((before, ac.action_class_id), set()).add(out)
        return out


def learn_mealy(
    sandbox: SandboxControl, spec: ServerSpec, fixture_id: str, budget: int, seed: int = 0
) -> MealyResult:
    alphabet = {
        ac.label: ac
        for ac in spec.action_classes()
        if spec.is_core(ac) and ac.effect_class is not EffectClass.READ_ONLY
    }
    inputs = [OBSERVE] + sorted(alphabet)

    async def main() -> MealyResult:
        async with InMemoryToolClient(sandbox.mcp) as client:
            loop = asyncio.get_running_loop()

            def bridge(coro: Coroutine[Any, Any, Any]) -> Any:
                return asyncio.run_coroutine_threadsafe(coro, loop).result()

            sul = SandboxSUL(sandbox, spec, fixture_id, budget, alphabet, client, bridge)
            return await asyncio.to_thread(_learn_sync, sul, inputs, alphabet, budget, seed)

    return asyncio.run(main())


def _learn_sync(
    sul: SandboxSUL, inputs: list[str], alphabet: dict[str, ActionClass], budget: int, seed: int
) -> MealyResult:
    automaton = None
    converged = False
    error: str | None = None
    try:
        from aalpy.learning_algs import run_Lstar
        from aalpy.oracles import RandomWalkEqOracle

        random.seed(seed)
        eq = RandomWalkEqOracle(
            inputs, sul.sul, num_steps=max(budget // 3, 10), reset_after_cex=True, reset_prob=0.2
        )
        automaton = run_Lstar(
            inputs,
            sul.sul,
            eq_oracle=eq,
            automaton_type="mealy",
            cache_and_non_det_check=True,
            print_level=0,
        )
        converged = True
    except BudgetExhausted as e:
        error = str(e)
    except Exception as e:  # AALpy raises plain exceptions on nondeterminism etc.
        error = f"{type(e).__name__}: {e}"
    table = dict(sul.table)
    if automaton is not None:
        # fold the learned machine into the same (features, action) -> outcomes table so the
        # guidance is comparable: a Mealy state "is" the observation it emits.
        for state in automaton.states:
            obs = state.output_fun.get(OBSERVE)
            if obs is None:
                continue
            for letter, out in state.output_fun.items():
                if letter == OBSERVE:
                    continue
                table.setdefault((obs, alphabet[letter].action_class_id), set()).add(str(out))
    return MealyResult(
        converged=converged,
        probes=sul.probes,
        resets=sul.resets,
        states=len(automaton.states) if automaton is not None else 0,
        error=error,
        table=table,
        automaton=automaton,
    )


def mealy_guidance(spec: ServerSpec, result: MealyResult) -> TraceGuidance:
    return TraceGuidance("B3", spec, result.table)
