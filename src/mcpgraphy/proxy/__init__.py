"""Proxy mode: sit between an agent and a real MCP server.

The proxy exposes the downstream server's tools unchanged, logs every call with a read-back of
the touched resource (passive evidence, no probing, no resets), materializes the behavior graph
from that log, serves `mcpgraphy_*` tools for offers and explanations, and depending on the
policy annotates (warn) or refuses (block) calls the evidence says will fail or that need host
approval.
"""

from __future__ import annotations

from mcpgraphy.proxy.wrap import Proxy, ProxyPolicy, serve_stdio

__all__ = ["Proxy", "ProxyPolicy", "serve_stdio"]
