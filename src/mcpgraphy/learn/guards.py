"""Guard induction and the four-way behavioral status.

Pure and deterministic: iterates sorted containers only, no randomness, no I/O.

Rules, in plain words:
  * A transition is OBSERVED_SUPPORTED only where a probe at *exactly that abstract state*
    reproduced it, and no probe at that state disagreed. Supported failures count too: a state
    where `publish_release` always fails has an OBSERVED_SUPPORTED failure edge.
  * Where probes at one abstract state disagree (same visible features, different outcomes), the
    earliest outcome was the hypothesis and later ones are counterexamples: the earliest row is
    CONTRADICTED, the others CONDITIONAL with a "hidden condition suspected" note. Nothing is
    overwritten; every successor stays in the graph.
  * A state never probed for an action gets a CONDITIONAL hypothesis only if an induced guard
    (a conjunction over visible features that separates successes from failures) covers it; its
    provenance says HYPOTHESIZED_BY_GUARD and its evidence is the experiments the guard came from.
  * Everything else is simply absent, and the selector reports UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mcpgraphy.learn.abstract import diff_effects
from mcpgraphy.models import (
    AbstractState,
    BehaviorStatus,
    Counterexample,
    Experiment,
    Features,
    Guard,
    OutcomeKind,
    Predicate,
    Provenance,
    Transition,
)


@dataclass
class Induced:
    transitions: list[Transition] = field(default_factory=list)
    counterexamples: list[Counterexample] = field(default_factory=list)
    guards: dict[str, Guard | None] = field(default_factory=dict)  # action_class_id -> guard


def induce(
    model_version: str,
    experiments: list[Experiment],
    states: dict[str, AbstractState],
    action_class_ids: list[str],
) -> Induced:
    out = Induced()
    exps = sorted(experiments, key=lambda e: e.seq)
    known = {sid: st.features for sid, st in sorted(states.items())}
    for ac_id in sorted(action_class_ids):
        by_state: dict[str, list[Experiment]] = {}
        for e in exps:
            if e.action_class_id == ac_id:
                by_state.setdefault(e.before_state_id, []).append(e)
        s_ok = {sid for sid, es in by_state.items() if any(e.result.kind.is_ok for e in es)}
        s_err = {sid for sid, es in by_state.items() if any(not e.result.kind.is_ok for e in es)}
        guard = _induce_guard(known, s_ok, s_err) if s_ok else None
        if guard is not None:
            probed = set(by_state)
            satisfying = [sid for sid, f in known.items() if guard.holds(f)]
            guard = guard.model_copy(
                update={
                    "coverage_total": len(satisfying),
                    "coverage_tested": len([sid for sid in satisfying if sid in probed]),
                    "note": (
                        "hidden condition suspected: mixed outcomes under identical observed features"  # noqa: E501
                        if s_ok & s_err
                        else guard.note
                    ),
                }
            )
        out.guards[ac_id] = guard

        # ---- observed rows
        for sid in sorted(by_state):
            es = by_state[sid]
            groups: dict[tuple[OutcomeKind, str | None], list[Experiment]] = {}
            for e in es:
                groups.setdefault((e.result.kind, e.after_state_id), []).append(e)
            mixed = sid in s_ok and sid in s_err
            earliest_key = min(groups, key=lambda k: groups[k][0].seq)
            for key in sorted(groups, key=lambda k: (str(k[0]), k[1] or "")):
                kind, target = key
                evidence = tuple(e.experiment_id for e in groups[key])
                status = BehaviorStatus.OBSERVED_SUPPORTED
                note_guard = guard if kind.is_ok else None
                if mixed:
                    status = (
                        BehaviorStatus.CONTRADICTED
                        if key == earliest_key
                        else BehaviorStatus.CONDITIONAL
                    )
                tid = Transition.make_id(model_version, sid, ac_id, kind, target)
                cx_ids: list[str] = []
                if mixed and key == earliest_key:
                    for other_key, other_es in groups.items():
                        if other_key == key:
                            continue
                        for e in other_es:
                            cx = Counterexample(
                                counterexample_id=Counterexample.make_id(
                                    model_version, tid, e.experiment_id
                                ),
                                model_version=model_version,
                                transition_id=tid,
                                experiment_id=e.experiment_id,
                                predicted={"outcome": str(kind), "target_state_id": target},
                                observed={
                                    "outcome": str(e.result.kind),
                                    "target_state_id": e.after_state_id,
                                },
                            )
                            out.counterexamples.append(cx)
                            cx_ids.append(cx.counterexample_id)
                out.transitions.append(
                    Transition(
                        transition_id=tid,
                        model_version=model_version,
                        source_state_id=sid,
                        action_class_id=ac_id,
                        outcome_kind=kind,
                        target_state_id=target,
                        status=status,
                        guard=note_guard,
                        provenance=Provenance.SANDBOX_PROBE,
                        evidence_ids=evidence,
                        counterexample_ids=tuple(cx_ids),
                        predicted_effects=diff_effects(
                            known.get(sid, {}), known.get(target or "", None) if target else None
                        ),
                        created_at=groups[key][0].seq,
                    )
                )

        # ---- hypothesized rows for unprobed states covered by the guard
        if guard is None or s_ok & s_err:
            continue  # no hypothesis from mixed evidence
        ok_exps = [e for e in exps if e.action_class_id == ac_id and e.result.kind.is_ok]
        effect_map = _common_effects(ok_exps, known)
        for sid in sorted(known):
            if sid in by_state or not guard.holds(known[sid]):
                continue
            target_feats = (
                _apply_effects(known[sid], effect_map) if effect_map is not None else None
            )
            target_id = None
            if target_feats is not None:
                target_state = AbstractState.build(
                    states[next(iter(states))].server_id,
                    states[next(iter(states))].resource_class,
                    target_feats,
                )
                target_id = target_state.state_id
                states.setdefault(target_id, target_state)
                known.setdefault(target_id, target_feats)
            tid = Transition.make_id(model_version, sid, ac_id, OutcomeKind.OK, target_id)
            out.transitions.append(
                Transition(
                    transition_id=tid,
                    model_version=model_version,
                    source_state_id=sid,
                    action_class_id=ac_id,
                    outcome_kind=OutcomeKind.OK,
                    target_state_id=target_id,
                    status=BehaviorStatus.CONDITIONAL,
                    guard=guard,
                    provenance=Provenance.HYPOTHESIZED_BY_GUARD,
                    evidence_ids=tuple(e.experiment_id for e in ok_exps),
                    predicted_effects=diff_effects(known[sid], target_feats),
                    created_at=ok_exps[0].seq if ok_exps else 0,
                )
            )
    out.transitions.sort(key=lambda t: t.transition_id)
    out.counterexamples.sort(key=lambda c: c.counterexample_id)
    return out


# -------------------------------------------------------------------------------------------
# guard induction
# -------------------------------------------------------------------------------------------


def _induce_guard(known: dict[str, Features], s_ok: set[str], s_err: set[str]) -> Guard:
    """Conjunction of features constant across successful states, greedily minimized while it
    still excludes every purely-failing state. With no negatives, nothing is dropped."""
    ok_feats = [known[s] for s in sorted(s_ok) if s in known]
    if not ok_feats:
        return Guard()
    keys = sorted(set.intersection(*(set(f) for f in ok_feats)))
    constant = {
        k: ok_feats[0][k] for k in keys if all(f.get(k) == ok_feats[0][k] for f in ok_feats)
    }
    negatives = [known[s] for s in sorted(s_err - s_ok) if s in known]
    current = dict(constant)
    if negatives:
        for k in sorted(constant):
            candidate = {kk: v for kk, v in current.items() if kk != k}
            if all(not _holds(candidate, n) for n in negatives):
                current = candidate
    preds = tuple(Predicate(feature=k, op="==", value=v) for k, v in sorted(current.items()))
    return Guard(all_of=preds)


def _holds(conj: Features, feats: Features) -> bool:
    return all(feats.get(k) == v for k, v in conj.items())


def _common_effects(
    ok_exps: list[Experiment], known: dict[str, Features]
) -> dict[str, object] | None:
    """Per-feature 'becomes X' effects shared by every successful observation, or None when the
    successes disagree (then the hypothesized successor is left unobserved)."""
    effects: dict[str, object] | None = None
    for e in ok_exps:
        src = known.get(e.before_state_id)
        dst = known.get(e.after_state_id or "")
        if src is None or dst is None:
            return None
        this = {k: dst.get(k) for k in sorted(set(src) | set(dst)) if src.get(k) != dst.get(k)}
        if effects is None:
            effects = this
        elif effects != this:
            return None
    return effects if effects is not None else {}


def _apply_effects(feats: Features, effects: dict[str, object]) -> Features:
    out = dict(feats)
    out.update(effects)  # type: ignore[arg-type]
    return dict(sorted(out.items()))
