"""Outcome normalization and trace redaction.

Error grammar: simulated servers raise `ToolError("CODE: message")`. The SDK wraps that as
`Error executing tool <name>: CODE: message` in a text content block with `is_error=True`.
Pydantic argument validation failures come back the same way with "validation error" text.
Anything unparsable is ERR_OTHER: we never guess a nicer category.
"""

from __future__ import annotations

import re
from typing import Any

from mcpgraphy.models import Outcome, OutcomeKind

CODE_MAP: dict[str, OutcomeKind] = {
    "NOT_FOUND": OutcomeKind.ERR_NOT_FOUND,
    "STATE_CONFLICT": OutcomeKind.ERR_STATE,
    "PERMISSION_DENIED": OutcomeKind.ERR_PERMISSION,
    "INVALID_ARGUMENT": OutcomeKind.ERR_VALIDATION,
    # POSIX / Node style codes, as emitted by e.g. the official filesystem MCP server
    "ENOENT": OutcomeKind.ERR_NOT_FOUND,
    "EACCES": OutcomeKind.ERR_PERMISSION,
    "EPERM": OutcomeKind.ERR_PERMISSION,
    "EEXIST": OutcomeKind.ERR_STATE,
    "ENOTEMPTY": OutcomeKind.ERR_STATE,
    "EISDIR": OutcomeKind.ERR_STATE,
    "ENOTDIR": OutcomeKind.ERR_STATE,
    "EINVAL": OutcomeKind.ERR_VALIDATION,
}

_SDK_PREFIX = re.compile(r"^Error executing tool [\w.-]+: ")
_CODE = re.compile(r"^([A-Z][A-Z_]+): ?(.*)$", re.DOTALL)
_VALIDATION = re.compile(r"validation error", re.IGNORECASE)


def normalize_error(text: str) -> Outcome:
    body = _SDK_PREFIX.sub("", text.strip(), count=1)
    m = _CODE.match(body)
    if m and m.group(1) in CODE_MAP:
        return Outcome(
            kind=CODE_MAP[m.group(1)],
            code=m.group(1),
            message=redact_text(m.group(2).strip()[:200]),
        )
    if _VALIDATION.search(body):
        return Outcome(
            kind=OutcomeKind.ERR_VALIDATION,
            code="SCHEMA_VALIDATION",
            message=redact_text(body.splitlines()[0][:200]),
        )
    return Outcome(kind=OutcomeKind.ERR_OTHER, code=None, message=redact_text(body[:200]))


def ok_outcome() -> Outcome:
    return Outcome(kind=OutcomeKind.OK, code=None, message="")


# ---------------------------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------------------------

_SECRET_KEY = re.compile(
    r"(secret|token|password|passwd|authorization|api[_-]?key|cookie)", re.IGNORECASE
)
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE)
_LONG_TOKEN = re.compile(r"\b(?:sk|ghp|xoxb|AKIA)[A-Za-z0-9_\-]{12,}\b")
REDACTED = "[REDACTED]"


def redact_text(text: str) -> str:
    text = _BEARER.sub(REDACTED, text)
    return _LONG_TOKEN.sub(REDACTED, text)


def redact(obj: Any) -> Any:
    """Recursively mask secret-looking keys and token-looking values before journaling."""
    if isinstance(obj, dict):
        return {k: (REDACTED if _SECRET_KEY.search(str(k)) else redact(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj)
    return obj
