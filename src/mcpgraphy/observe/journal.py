"""Append-only, fsynced JSONL evidence journal.

The journal is the source of truth: the graph is always materialized *from* it, so a crash
between a journal write and graph materialization loses nothing. A truncated final line (power
loss mid-write) is skipped and reported, never parsed as evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcpgraphy.models import canonical_json
from mcpgraphy.observe.normalize import redact


class Journal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="ascii")  # noqa: SIM115
        self.seq = _count_lines(self.path)

    def append(self, kind: str, payload: dict[str, Any]) -> int:
        self.seq += 1
        record = {"kind": kind, "seq": self.seq, **redact(payload)}
        self._fh.write(canonical_json(record) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        return self.seq

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @staticmethod
    def read(path: str | Path) -> tuple[list[dict[str, Any]], int]:
        """Return (events, skipped_lines). Malformed lines are skipped, not guessed."""
        events: list[dict[str, Any]] = []
        skipped = 0
        p = Path(path)
        if not p.exists():
            return events, skipped
        with open(p, encoding="ascii") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except ValueError:
                    skipped += 1
        return events, skipped


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="ascii") as fh:
        return sum(1 for line in fh if line.strip())
