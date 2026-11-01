# Security

mcpgraphy runs locally and makes no network calls of its own. In proxy mode it forwards the agent's calls to the downstream server you name and reads resources back through that server's own read-only tool. It never adds probes, resets or writes.

## What it stores

The SQLite database and the JSONL journal next to it contain every call the proxy saw: tool name, arguments, the result text, and the read-back of the touched resource. Treat them like logs: keep them out of version control (the default names are gitignored) and delete them when you no longer need them. Known secret-shaped values are redacted before writing, but redaction is best effort.

## Approval

In `block` mode, destructive and approval-required tools run only if listed in `--approve`. In research mode, delegated execution is off by default and the approval token is read only from the `MCPGRAPHY_APPROVAL_TOKEN` environment variable, never from the command line.

## Reporting

If you find a way to make the proxy forward a blocked call, leak the journal, or execute an offer without the host policy allowing it, open a private security advisory on GitHub or email the maintainer listed on PyPI. Please do not open a public issue for it.
