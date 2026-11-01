"""Sandbox server registry: safety metadata for exploration only.

Everything here is *declared* metadata: effect classes, approval requirements, the read-only
observation tool, and the argument-class grammar. None of it is behavioral evidence; a unit test
proves a declaration alone can never raise a transition above UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcpgraphy.models import ActionClass, EffectClass
from mcpgraphy.sandbox.base import FixtureManifest


@dataclass(frozen=True)
class ParamSpec:
    name: str
    classes: tuple[str, ...]  # all argument classes, in deterministic order
    core: tuple[str, ...]  # classes probed on every fixture (state-dependent core)


@dataclass(frozen=True)
class ToolPolicy:
    name: str
    effect: EffectClass
    approval_required: bool
    params: tuple[ParamSpec, ...]


@dataclass(frozen=True)
class ServerSpec:
    server_id: str
    version: str
    resource_class: str
    tenant: str
    observation_tool: str
    id_param: str
    tools: tuple[ToolPolicy, ...]

    def tool(self, name: str) -> ToolPolicy:
        for t in self.tools:
            if t.name == name:
                return t
        raise KeyError(name)

    @property
    def allowed_tools(self) -> frozenset[str]:
        return frozenset(t.name for t in self.tools)

    def action_classes(self) -> list[ActionClass]:
        """Every (tool, argument-class combination), deterministic order."""
        out: list[ActionClass] = []
        for t in self.tools:
            for combo in _product([p.classes for p in t.params]):
                arg_classes = {p.name: c for p, c in zip(t.params, combo, strict=True)}
                out.append(ActionClass.build(self.server_id, t.name, arg_classes, t.effect))
        return out

    def is_core(self, ac: ActionClass) -> bool:
        t = self.tool(ac.tool)
        return all(ac.arg_classes[p.name] in p.core for p in t.params)


def _product(lists: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = [()]
    for choices in lists:
        result = [r + (c,) for r in result for c in choices]
    return result


# ---------------------------------------------------------------------------------------------
# Argument-class grammar
# ---------------------------------------------------------------------------------------------

ID_CLASSES = ("EXISTING", "NONEXISTENT", "STALE_DELETED", "OTHER_TENANT", "MALFORMED")
USER_CLASSES = ("USER_IN_PROJECT", "USER_OTHER_PROJECT", "USER_NONEXISTENT")
STATE_CLASSES = ("TARGET_OPEN", "TARGET_IN_PROGRESS", "TARGET_DONE", "TARGET_INVALID")
PROJECT_CLASSES = ("EXISTING_PROJECT", "SIMILAR_NAME_OTHER_PROJECT", "NONEXISTENT_PROJECT")
TEXT_CLASSES = ("VALID", "EMPTY")

CONSTANT_VALUES: dict[str, Any] = {
    "VALID": "Edited by probe",
    "EMPTY": "",
    "TARGET_OPEN": "OPEN",
    "TARGET_IN_PROGRESS": "IN_PROGRESS",
    "TARGET_DONE": "DONE",
    "TARGET_INVALID": "LIMBO",
    "DEFAULT": "default",
    "DEFAULT_FALSE": False,
}


def concretize(manifest: FixtureManifest, ac: ActionClass) -> dict[str, Any] | None:
    """Turn an action class into concrete arguments for this fixture, or None if a class has no
    value available (e.g. the pool is empty)."""
    args: dict[str, Any] = {}
    for param, cls in ac.arg_classes.items():
        if cls in CONSTANT_VALUES:
            args[param] = CONSTANT_VALUES[cls]
            continue
        pool = manifest.id_pools.get(cls)
        if not pool:
            return None
        args[param] = pool[0]
    return args


def classify_value(
    manifest: FixtureManifest, spec: ServerSpec, tool: str, param: str, value: Any
) -> str:
    """Reverse lookup: which argument class does a concrete value belong to for this fixture?

    Values not recognised from the fixture's pools or constants fall into the parameter's
    "unrecognised" bucket (VALID/EMPTY by emptiness for text, NONEXISTENT-style classes for
    identifiers, TARGET_INVALID for unknown state literals). Never silently EXISTING."""
    pspec = next(p for p in spec.tool(tool).params if p.name == param)
    if param == spec.id_param and value == manifest.focal_ref.local_id:
        return "EXISTING"
    for cls in pspec.classes:
        if cls in CONSTANT_VALUES and CONSTANT_VALUES[cls] == value:
            return cls
        if value in manifest.id_pools.get(cls, []):
            return cls
    if set(pspec.classes) == set(TEXT_CLASSES):
        return "EMPTY" if not isinstance(value, str) or not value.strip() else "VALID"
    for fallback in ("TARGET_INVALID", "USER_NONEXISTENT", "NONEXISTENT_PROJECT", "NONEXISTENT"):
        if fallback in pspec.classes:
            return fallback
    return pspec.classes[-1]


def _id(name: str) -> ParamSpec:
    return ParamSpec(name, ID_CLASSES, ("EXISTING",))


def _text(name: str) -> ParamSpec:
    return ParamSpec(name, TEXT_CLASSES, ("VALID",))


def release_spec(version: str = "v1") -> ServerSpec:
    run_tests_params: tuple[ParamSpec, ...] = (_id("release_id"),)
    if version == "v2":
        run_tests_params += (ParamSpec("suite", ("DEFAULT",), ("DEFAULT",)),)
    return ServerSpec(
        server_id="release_sim",
        version=version,
        resource_class="release",
        tenant="t1",
        observation_tool="get_release",
        id_param="release_id",
        tools=(
            ToolPolicy("create_release", EffectClass.MUTATING, False, (_text("name"),)),
            ToolPolicy("get_release", EffectClass.READ_ONLY, False, (_id("release_id"),)),
            ToolPolicy(
                "edit_release", EffectClass.MUTATING, False, (_id("release_id"), _text("title"))
            ),
            ToolPolicy("run_tests", EffectClass.MUTATING, False, run_tests_params),
            ToolPolicy("publish_release", EffectClass.MUTATING, True, (_id("release_id"),)),
            ToolPolicy("archive_release", EffectClass.DESTRUCTIVE, True, (_id("release_id"),)),
        ),
    )


def ticket_spec(version: str = "v1") -> ServerSpec:
    list_users_params: tuple[ParamSpec, ...] = (
        ParamSpec("project_id", PROJECT_CLASSES, ("EXISTING_PROJECT",)),
    )
    if version == "v2":
        list_users_params += (
            ParamSpec("include_inactive", ("DEFAULT_FALSE",), ("DEFAULT_FALSE",)),
        )
    return ServerSpec(
        server_id="ticket_sim",
        version=version,
        resource_class="ticket",
        tenant="p-alpha",
        observation_tool="get_ticket",
        id_param="ticket_id",
        tools=(
            ToolPolicy(
                "create_ticket",
                EffectClass.MUTATING,
                False,
                (ParamSpec("project_id", PROJECT_CLASSES, ("EXISTING_PROJECT",)), _text("title")),
            ),
            ToolPolicy("get_ticket", EffectClass.READ_ONLY, False, (_id("ticket_id"),)),
            ToolPolicy("list_users", EffectClass.READ_ONLY, False, list_users_params),
            ToolPolicy(
                "assign_ticket",
                EffectClass.MUTATING,
                False,
                (_id("ticket_id"), ParamSpec("user_id", USER_CLASSES, ("USER_IN_PROJECT",))),
            ),
            ToolPolicy(
                "move_ticket",
                EffectClass.MUTATING,
                False,
                (_id("ticket_id"), ParamSpec("state", STATE_CLASSES, STATE_CLASSES[:3])),
            ),
            ToolPolicy("close_ticket", EffectClass.DESTRUCTIVE, True, (_id("ticket_id"),)),
        ),
    )


SERVER_IDS = ("release_sim", "ticket_sim")


def get_spec(server_id: str, version: str = "v1") -> ServerSpec:
    if server_id == "release_sim":
        return release_spec(version)
    if server_id == "ticket_sim":
        return ticket_spec(version)
    raise KeyError(f"unknown server {server_id}")
