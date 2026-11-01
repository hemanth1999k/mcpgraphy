"""Affordance selection and explanation."""

from __future__ import annotations

import re

from mcpgraphy.sandbox.base import SandboxControl


def load_demo_fixture(
    sandbox: SandboxControl,
    resource_ref: str,
    fixture_seed: int | None = None,
    fixture_index: int = 0,
) -> str | None:
    """Reset the sandbox to the fixture a demo resource id names.

    `release:demo-7-0` and `ticket:tk-7-0` mean seed 7, fixture index 0. Explicit
    `fixture_seed` overrides the parsed seed. Returns the fixture id, or None when the id does
    not name a demo fixture (the resource will then simply not exist: NEEDS_OBSERVATION)."""
    local = resource_ref.split(":", 1)[1] if ":" in resource_ref else resource_ref
    m = re.fullmatch(r"(?:demo|tk)-(\d+)-(\d+)", local)
    seed = fixture_seed if fixture_seed is not None else (int(m.group(1)) if m else None)
    index = int(m.group(2)) if m else fixture_index
    if seed is None:
        return None
    ids = sandbox.fixtures(seed, index + 1)
    fid = ids[index]
    sandbox.reset_fixture(fid)
    return fid
