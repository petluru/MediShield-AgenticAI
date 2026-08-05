"""MCP server (PROJECT_PLAN.md §5: "Fraud Detection's tools... and Policy
RAG's tools... should be exposed through the MCP server rather than
hardcoded Python functions") exposing the two existing agent tools:
`retrieve_policy_clauses` (backend/rag/retrieval.py) and
`lookup_claim_history` (backend/fraud/claim_history.py).

Single source of truth: this module does not reimplement either tool's
logic. It wraps the *same* underlying callables the LangChain agents
already use (`.func`, the unwrapped function inside each `@tool`-decorated
`StructuredTool` — same unwrap pattern `backend/security/tool_scanning.py`
uses for the same reason: the wrapper adds framework metadata, the real
logic lives in `.func`). If either tool's behavior changes, both the
LangChain-direct path and this MCP server change together automatically —
there's nothing to keep in sync manually.

Run standalone (stdio transport, for a real MCP client/inspector to
connect to):
    uv run python -m backend.scripts.run_mcp_server
"""

from mcp.server.mcpserver import MCPServer

from backend.fraud.claim_history import lookup_claim_history as _lookup_claim_history_tool
from backend.rag.retrieval import retrieve_policy_clauses as _retrieve_policy_clauses_tool

mcp_server = MCPServer(
    name="medishield-tools",
    description="MediShield policy retrieval and claim-history lookup tools.",
)


@mcp_server.tool(
    name="retrieve_policy_clauses",
    description=_retrieve_policy_clauses_tool.description,
)
def retrieve_policy_clauses(query: str, plan: str | None = None, n_results: int = 5) -> str:
    return _retrieve_policy_clauses_tool.func(query, plan, n_results)


@mcp_server.tool(
    name="lookup_claim_history",
    description=_lookup_claim_history_tool.description,
)
def lookup_claim_history(patient_id: str) -> str:
    return _lookup_claim_history_tool.func(patient_id)
