"""Runs the MediShield MCP server standalone over stdio — connect any real
MCP client (e.g. the MCP Inspector, or Claude Desktop's MCP config) to this
process. No Anthropic API calls happen in this process at all; it only
serves `retrieve_policy_clauses` and `lookup_claim_history` over the MCP
protocol.

Usage:
    uv run python -m backend.scripts.run_mcp_server
"""

import anyio

from backend.mcp_server.server import mcp_server


def main() -> None:
    anyio.run(mcp_server.run_stdio_async)


if __name__ == "__main__":
    main()
