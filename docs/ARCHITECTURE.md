# Architecture in one page

## Data flow

```
                 OFFLINE (learning laboratory)                      LIVE (serving)
 sandbox.fixtures(seed, n)                                   MCP client (any host)
        |                                                             |
        v                                                             v
 reset_fixture(fid) ──► observe ──► safety.check ──► call ──► observe  mcpgraphy serve
        |                                                             |
        └──────────────► journal.jsonl (append-only, fsynced)        list_affordances
                                  |                                   |  inspect_resource via
                                  v                                   |  the read-only tool
                        builder.materialize()  ── guards.induce ──►   |
                                  |              counterexamples      v
                                  v                                selector.select_offers
                         SQLite (models, transitions, ...)  ◄─────────┘
                                  |
                                  v
                         export_json / export_mermaid
```

The journal is the only input to the graph. Re-running `materialize` on the same journal gives a
byte-identical export; re-running `learn` with the same seed gives a byte-identical journal.

## Identity

Every id is `prefix_` + the first 16 hex chars of sha256 over canonical JSON (sorted keys, no
whitespace). States hash `(server_id, resource_class, features)`; action classes hash
`(server_id, tool, arg_classes)`; transitions hash `(model_version, source, action, outcome,
target)`. No wall-clock time, memory address or insertion order ever enters a hash.

## Trust boundaries

- The learner receives a `RestrictedSandboxView`: `fixtures`, `reset_fixture`, `mcp`, `counter`,
  `reset_supported`. Anything else (`world`, `rules`, `_specs`, hidden flags) raises.
- The oracle lives in `benchmarks/ground_truth/`. `mcpgraphy` (except `evaluation/`) never
  imports it; a test walks the AST and another runs a learn in a fresh interpreter and inspects
  `sys.modules`.
- Declared metadata (`registry.py`) can gate *exploration* (effect classes, approval flags) but
  can never produce behavioral evidence: a declaration alone leaves a transition UNKNOWN.
- The offer server only ever calls the declared read-only observation tool.

## Status rules (learn/guards.py)

1. Probe at state S with action A reproduced outcome O, no probe at S disagreed →
   `OBSERVED_SUPPORTED` (also for failures).
2. Probes at S disagree → earliest row `CONTRADICTED` with linked counterexamples, later rows
   `CONDITIONAL` with note "hidden condition suspected". All successors kept.
3. S never probed with A but an induced guard (minimal conjunction over visible features that
   separates successes from failures) covers S → `CONDITIONAL`, provenance
   `HYPOTHESIZED_BY_GUARD`.
4. Otherwise no row → `UNKNOWN` at offer time.
5. Across server versions: inherited rows are `UNKNOWN` + `retest_required`; rows for tools whose
   input schema changed are dropped; a re-observed disagreement contradicts the inherited row.
