"""Scenario generation for the `stateful-v1` suite.

200 scenario configurations = 2 servers x 10 categories x 10 seeds. Scenario seeds start at
1000 so benchmark fixtures are disjoint from the learner's fixtures (seeds 0..19).

The scenario *recipes* name visible and hidden conditions (that is benchmark-author knowledge);
the learner never sees this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CATEGORY_NAMES = {
    1: "normal valid sequence",
    2: "invalid action due to state guard",
    3: "wrong or missing resource identifiers",
    4: "unobservable hidden state",
    5: "alternate valid sequence",
    6: "schema changed between versions",
    7: "behavior changed without schema change",
    8: "concurrent update / stale snapshot",
    9: "nondeterministic result under same observed state",
    10: "insufficient reset capability",
}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    server_id: str
    server_version: str  # the server the agent acts on
    learner_version: str  # the server version the learned models come from
    category: int
    seed: int
    fixture_index: int
    local_id: str
    goal: dict[str, Any]
    achievable: bool
    max_steps: int
    no_reset: bool = False
    concurrent_op: str | None = None
    distractors: dict[str, str] = field(default_factory=dict)
    requirement: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "server_id": self.server_id,
            "server_version": self.server_version,
            "learner_version": self.learner_version,
            "category": self.category,
            "category_name": CATEGORY_NAMES[self.category],
            "seed": self.seed,
            "fixture_index": self.fixture_index,
            "local_id": self.local_id,
            "goal": self.goal,
            "achievable": self.achievable,
            "max_steps": self.max_steps,
            "no_reset": self.no_reset,
            "concurrent_op": self.concurrent_op,
            "distractors": self.distractors,
            "requirement": self.requirement,
        }


# Recipe = (fixture requirement on the private spec, goal, server version, concurrent op)
# Requirements are matched against the sandbox's fixture spec fields (benchmark-side only).


def release_recipe(cat: int, k: int) -> dict[str, Any]:
    odd = k % 2 == 1
    table: dict[int, dict[str, Any]] = {
        1: {"req": {"status": "DRAFT", "tests_passed": False}, "goal": {"status": "PUBLISHED"}},
        2: {"req": {"status": "PUBLISHED", "locked": False}, "goal": {"status": "DRAFT"}},
        3: {
            "req": {"status": "DRAFT", "tests_passed": False},
            "goal": {"status": "PUBLISHED"},
            "distractors": {"release_id": "STALE_DELETED"},
        },
        4: {"req": {"status": "PUBLISHED", "locked": True}, "goal": {"status": "ARCHIVED"}},
        5: {
            "req": {"status": "DRAFT", "tests_passed": True, "mismatch": True},
            "goal": {"status": "PUBLISHED"},
        },
        6: {
            "req": {"status": "DRAFT", "tests_passed": False},
            "goal": {"status": "PUBLISHED"},
            "version": "v2",
        },
        7: {
            "req": {"status": "DRAFT", "tests_passed": False},
            "goal": {"status": "ARCHIVED"},
            "version": "v2",
        },
        8: {
            "req": {"status": "PUBLISHED", "locked": False},
            "goal": {"status": "ARCHIVED"},
            "concurrent_op": "lock",
        },
        9: {"req": {"status": "PUBLISHED", "locked": odd}, "goal": {"status": "ARCHIVED"}},
        10: {
            "req": {"status": "DRAFT", "tests_passed": False},
            "goal": {"status": "PUBLISHED"},
            "no_reset": True,
        },
    }
    return table[cat]


def ticket_recipe(cat: int, k: int) -> dict[str, Any]:
    odd = k % 2 == 1
    table: dict[int, dict[str, Any]] = {
        1: {
            "req": {"state": "OPEN", "has_assignee": True, "write_denied": False},
            "goal": {"state": "DONE"},
        },
        2: {"req": {"state": "CLOSED"}, "goal": {"state": "IN_PROGRESS"}},
        3: {
            "req": {"state": "OPEN", "has_assignee": False, "write_denied": False},
            "goal": {"state": "IN_PROGRESS"},
            "distractors": {"user_id": "USER_OTHER_PROJECT", "ticket_id": "STALE_DELETED"},
        },
        4: {
            "req": {"state": "OPEN", "has_assignee": False, "write_denied": True},
            "goal": {"state": "DONE"},
        },
        5: {
            "req": {"state": "OPEN", "has_assignee": False, "write_denied": False},
            "goal": {"state": "IN_PROGRESS"},
        },
        6: {
            "req": {"state": "OPEN", "has_assignee": True, "write_denied": False},
            "goal": {"state": "DONE"},
            "version": "v2",
        },
        7: {
            "req": {"state": "IN_PROGRESS", "has_assignee": True},
            "goal": {"state": "CLOSED"},
            "version": "v2",
        },
        8: {
            "req": {"state": "DONE", "has_assignee": True, "write_denied": False},
            "goal": {"state": "CLOSED"},
            "concurrent_op": "deny",
        },
        9: {
            "req": {"state": "DONE", "has_assignee": True, "write_denied": odd},
            "goal": {"state": "CLOSED"},
        },
        10: {
            "req": {"state": "OPEN", "has_assignee": True, "write_denied": False},
            "goal": {"state": "DONE"},
            "no_reset": True,
        },
    }
    return table[cat]


RECIPES = {"release_sim": release_recipe, "ticket_sim": ticket_recipe}


def scenario_seed(seed_base: int, cat: int, k: int) -> int:
    return seed_base + 10 * cat + k
