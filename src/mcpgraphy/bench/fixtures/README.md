# Fixtures

Fixtures are not stored. They are generated deterministically by each simulator from a seed
(`sandbox.fixtures(seed, n)`), so the complete benchmark world is reproducible from
`benchmarks/suites/*.json` alone. Learner fixtures use seeds 0..19; benchmark scenarios use
seeds >= 1000; held-out precision checks use seed 2000.
