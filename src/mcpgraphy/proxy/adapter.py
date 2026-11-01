"""Adapter: how to observe a resource on an arbitrary MCP server (proxy mode).

An adapter is a small JSON document the operator writes once per downstream server. It names
the read-only observation tool, the identifier parameter, and optionally which fields of the
observation count as state. Everything else (tool list, schemas, effect hints) comes from the
server's own `tools/list`.

Zero-config feature detection: when `features` is omitted, booleans and short enum-like strings
in the observation result are used as state features; identifiers, free text and numbers are
not (they would make every resource its own state).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mcpgraphy.learn import abstract
from mcpgraphy.models import ActionClass, EffectClass, Features, ToolSpec
from mcpgraphy.registry import ParamSpec, ServerSpec, ToolPolicy

VALUE = "VALUE"
ENUM_LIKE = re.compile(r"^[A-Za-z][A-Za-z0-9_\- ]{0,31}$")


class AdapterSpec(BaseModel):
    server_id: str
    resource_class: str
    observation_tool: str
    id_param: str
    tenant: str = "default"
    features: list[str] | None = None
    effect_classes: dict[str, str] = Field(default_factory=dict)
    approval_required: list[str] = Field(default_factory=list)
    version: str = "live"
    # Some servers return the observation as text ("key: value" lines) rather than fields.
    observation_format: str = "fields"  # "fields" | "kv_text"
    # Tools that address the resource through a differently named parameter (e.g. move_file's
    # `source`). The first alias present in a tool's schema is treated as its identifier.
    id_param_aliases: list[str] = Field(default_factory=list)

    def id_param_for(self, tool_params: list[str]) -> str | None:
        for name in [self.id_param, *self.id_param_aliases]:
            if name in tool_params:
                return name
        return None

    def parse_observation(self, raw: Any) -> dict[str, Any] | None:
        """Turn the observation tool's result into a flat dict of fields."""
        if raw is None:
            return None
        if self.observation_format == "kv_text":
            text = raw.get("content", raw.get("text", "")) if isinstance(raw, dict) else str(raw)
            out: dict[str, Any] = {}
            for line in str(text).splitlines():
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                v = v.strip()
                out[k.strip()] = (
                    True if v.lower() == "true" else False if v.lower() == "false" else v
                )
            return out
        return raw if isinstance(raw, dict) else {"value": raw}

    @classmethod
    def load(cls, path: str | Path) -> AdapterSpec:
        return cls.model_validate(json.loads(Path(path).read_text()))

    # ---- state abstraction ----------------------------------------------------------------
    def extract(self, raw: dict[str, Any]) -> Features:
        raw = self.parse_observation(raw) or {}
        fields = (
            self.features
            if self.features is not None
            else sorted(k for k, v in raw.items() if _enum_like(k, v))
        )
        feats: Features = {}
        for k in fields:
            v = raw.get(k)
            if isinstance(v, bool) or v is None:
                feats[k] = v
            elif isinstance(v, (int, float)):
                feats[k] = "PRESENT"
            elif isinstance(v, str):
                feats[k] = v if ENUM_LIKE.match(v) else "TEXT"
            else:
                feats[k] = "PRESENT"
        return feats

    def register_extractor(self) -> None:
        abstract.EXTRACTORS[self.resource_class] = self.extract

    # ---- dynamic server spec ---------------------------------------------------------------
    def build_spec(self, tools: list[ToolSpec]) -> ServerSpec:
        policies = []
        for t in sorted(tools, key=lambda t: t.name):
            props = (t.input_schema or {}).get("properties", {}) or {}
            tool_id = self.id_param_for(sorted(props))
            params = tuple(
                ParamSpec(name, ("EXISTING", "NONEXISTENT"), ("EXISTING",))
                if name == tool_id
                else ParamSpec(name, (VALUE,), (VALUE,))
                for name in sorted(props)
            )
            policies.append(
                ToolPolicy(
                    t.name,
                    self.effect_of(t),
                    t.name in self.approval_required,
                    params,
                )
            )
        return ServerSpec(
            server_id=self.server_id,
            version=self.version,
            resource_class=self.resource_class,
            tenant=self.tenant,
            observation_tool=self.observation_tool,
            id_param=self.id_param,
            tools=tuple(policies),
        )

    def effect_of(self, t: ToolSpec) -> EffectClass:
        if t.name in self.effect_classes:
            return EffectClass(self.effect_classes[t.name])
        if t.name == self.observation_tool:
            return EffectClass.READ_ONLY
        ann = t.annotations or {}
        if ann.get("readOnlyHint") or ann.get("read_only_hint"):
            return EffectClass.READ_ONLY
        if ann.get("destructiveHint") or ann.get("destructive_hint"):
            return EffectClass.DESTRUCTIVE
        return EffectClass.MUTATING


def _enum_like(key: str, value: Any) -> bool:
    if key.lower() in {"id", "uuid", "name", "title", "description", "url", "path"} or key.endswith(
        "_id"
    ):
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return bool(ENUM_LIKE.match(value)) and " " not in value
    return False


def action_class_for_call(
    spec: ServerSpec, tool: str, args: dict[str, Any], target_exists: bool | None
) -> ActionClass:
    """Argument abstraction for proxied calls: the identifier is EXISTING or NONEXISTENT based
    on the read-back; every other argument is a single VALUE class."""
    policy = spec.tool(tool)
    classes = {}
    for p in policy.params:
        if "EXISTING" in p.classes:
            classes[p.name] = "EXISTING" if target_exists else "NONEXISTENT"
        else:
            classes[p.name] = VALUE
    return ActionClass.build(spec.server_id, tool, classes, policy.effect)


def id_value(spec: ServerSpec, tool: str, args: dict[str, Any]) -> Any:
    """The identifier argument of a proxied call, whichever parameter carries it."""
    try:
        policy = spec.tool(tool)
    except KeyError:
        return None
    for p in policy.params:
        if "EXISTING" in p.classes:
            return args.get(p.name)
    return None


EXAMPLE_ADAPTER = {
    "server_id": "release_sim",
    "resource_class": "release",
    "observation_tool": "get_release",
    "id_param": "release_id",
    "features": ["status", "tests_passed"],
    "effect_classes": {"archive_release": "DESTRUCTIVE"},
    "approval_required": ["publish_release", "archive_release"],
}
