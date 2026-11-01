# Changelog

## 0.1.0 (2026-11-01)

First release.

- `mcpgraphy wrap`: a stdio proxy for any MCP server (stdio command or Streamable HTTP URL) that learns a behavior graph from the agent's own calls, with `observe`, `warn` and `block` policies, `--approve` for pre-approved tools, and four extra tools for the agent: `mcpgraphy_affordances`, `mcpgraphy_explain`, `mcpgraphy_graph`, `mcpgraphy_status`.
- Adapter JSON per server: observation tool, id parameter and aliases, `fields` or `kv_text` observations, feature selection, effect classes, approval-required tools. Adapters included for the official filesystem server and the two bundled simulators.
- Behavior graph with four edge statuses (`OBSERVED_SUPPORTED`, `CONDITIONAL`, `CONTRADICTED`, `UNKNOWN`), content-addressed ids, an fsynced JSONL journal, SQLite store, JSON and Mermaid export, drift detection by tool-schema fingerprint with retest of inherited edges.
- Research mode: two deterministic simulators with hidden state and versioned drift (`release_sim`, `ticket_sim`), a safe resettable explorer with budgets and guard induction, an action-offer MCP server (`mcpgraphy serve`) with controlled delegated execution off by default.
- Benchmark `stateful-v1`: 200 scenarios, baselines including AALpy L* and a passive-trace learner, ablations, paired bootstrap confidence intervals, byte-identical results per seed. See `docs/BENCHMARK.md`.
- 100 tests (unit, integration, property), CI on Python 3.11 and 3.13.
