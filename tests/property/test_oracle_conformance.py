"""The handwritten oracle guard tables agree with the simulators on every cell."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from mcpgraphy.bench.ground_truth import release_oracle, ticket_oracle
from mcpgraphy.registry import concretize, release_spec, ticket_spec
from mcpgraphy.sandbox import ReleaseSandbox, TicketSandbox


def _release_fixture(version: str, cell_index: int) -> tuple[ReleaseSandbox, str]:
    sb = ReleaseSandbox(version)  # type: ignore[arg-type]
    ids = sb.fixtures(seed=11, n=11)
    return sb, ids[cell_index % 11]


@settings(max_examples=150, deadline=None)
@given(
    version=st.sampled_from(["v1", "v2"]), cell=st.integers(0, 17), ac_index=st.integers(0, 10_000)
)
def test_release_handwritten_oracle_matches_simulation(version, cell, ac_index):
    sb, fid = _release_fixture(version, cell)
    manifest = sb.reset_fixture(fid)
    spec = release_spec(version)
    acs = spec.action_classes()
    ac = acs[ac_index % len(acs)]
    args = concretize(manifest, ac)
    assert args is not None
    simulated, _ = release_oracle.true_outcome(sb, ac.tool, args)
    rid = args.get("release_id")
    full = release_oracle.full_features(sb, rid) if rid is not None else None
    assert release_oracle.oracle_outcome(version, ac.tool, ac.arg_classes, full) is simulated


@settings(max_examples=150, deadline=None)
@given(
    version=st.sampled_from(["v1", "v2"]), cell=st.integers(0, 15), ac_index=st.integers(0, 10_000)
)
def test_ticket_handwritten_oracle_matches_simulation(version, cell, ac_index):
    sb = TicketSandbox(version)  # type: ignore[arg-type]
    fid = sb.fixtures(seed=11, n=10)[cell % 10]
    manifest = sb.reset_fixture(fid)
    spec = ticket_spec(version)
    acs = spec.action_classes()
    ac = acs[ac_index % len(acs)]
    args = concretize(manifest, ac)
    assert args is not None
    simulated, _ = ticket_oracle.true_outcome(sb, ac.tool, args)
    tid = args.get("ticket_id")
    full = ticket_oracle.full_features(sb, tid) if tid is not None else None
    assert ticket_oracle.oracle_outcome(version, ac.tool, ac.arg_classes, full) is simulated
