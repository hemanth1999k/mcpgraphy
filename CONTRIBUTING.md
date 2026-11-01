# Contributing

Bug reports, adapters for other MCP servers, and benchmark scenarios are all welcome.

## Setup

```bash
git clone https://github.com/hemanth1999k/mcpgraphy && cd mcpgraphy
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Adding an adapter

Put a JSON file in `examples/adapters/`, run the proxy against the real server in `warn` mode, and include in the pull request the `mcpgraphy_status` output and one `mcpgraphy_affordances` answer so the behavior can be checked. Say which version of the server you tested.

## Rules that keep the graph honest

- An edge is `OBSERVED_SUPPORTED` only with a direct observation at that abstract state and no disagreeing observation there.
- Nothing elevates `UNKNOWN` without evidence. Tool names, descriptions and annotations are hints for safety, never evidence of behavior.
- Authorization is a separate field and is never inferred from behavior.
- Conflicting observations are kept as separate edges with their counterexamples, never averaged away.
- Outputs are deterministic for a seed: no randomness in the explorer, no set iteration reaching an export.

## Benchmark changes

The oracle under `src/mcpgraphy/bench/ground_truth/` may be imported only by `evaluation/` and `bench/`; a test checks this in a fresh interpreter. If you add a scenario category, add it for both servers and update `docs/BENCHMARK.md` with the new table from `mcpgraphy benchmark report`.
