"""LangChain-compatible tool adapters that call `backend.mcp_server.server`'s
tools over the real MCP protocol, instead of importing the underlying
Python functions directly (`backend/rag/retrieval.py`,
`backend/fraud/claim_history.py`).

**Wired into the production agents since 2026-08-05** — `backend/agents/
policy_rag.py` and `backend/agents/fraud_detection.py` both bind
`mcp_policy_rag_tools()`/`mcp_fraud_detection_tools()` from this module as
their tool source (see those files' own docstrings for the validation that
preceded the switch: a real end-to-end call compared against the prior
direct-import path's result before it became the default). Built and
protocol-tested first with the switch deliberately deferred for a live
review, since it changes what's actually happening on every real Claude
call for two already-carefully-tuned agents (prompt caching, recursion
limits, cost telemetry, the reuse-not-reasoning escalation fix) — not
something to flip in an unsupervised session without checking.

The synchronous/async bridge below (`_call_mcp_tool_sync`) is what makes
this possible at all: MCP's client API (`mcp` 2.0.0, installed but not the
version originally pinned as `mcp>=1.1.2`) is fully async, while the whole
agent pipeline runs synchronously (`agent.invoke()`, not `ainvoke()` —
every Policy RAG/Fraud Detection test depends on that).

Connects fresh for every call rather than keeping one persistent session
alive across an agent's tool-calling loop — the MCP server here has zero
external resources to manage (it's an in-process wrapper around local
function calls, not a real network service), so the reconnect overhead is
negligible, and "connect once, once, get an answer, disconnect" is far
simpler to reason about correctly than a long-lived background-thread
session, which matters more than the small performance cost given this
needs to be right without live iteration."""

import anyio
from langchain_core.tools import StructuredTool
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from backend.fraud.claim_history import lookup_claim_history as _lookup_claim_history_tool
from backend.mcp_server.server import mcp_server
from backend.rag.retrieval import retrieve_policy_clauses as _retrieve_policy_clauses_tool


async def _run_server(server_read, server_write) -> None:
    await mcp_server._lowlevel_server.run(
        server_read, server_write, mcp_server._lowlevel_server.create_initialization_options()
    )


async def _call_mcp_tool_async(name: str, arguments: dict) -> str:
    # `tg.cancel_scope.cancel()` must happen strictly *after* the result is
    # fully captured, and the `return` must happen strictly *after* both
    # `async with` blocks have unwound — returning a value from inside a
    # task group you're in the middle of cancelling is unreliable (a real
    # bug found here: it silently produced `None`, not the real result).
    result_text: str | None = None
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        async with anyio.create_task_group() as tg:
            tg.start_soon(_run_server, server_read, server_write)
            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                if result.is_error:
                    raise RuntimeError(f"MCP tool {name!r} returned an error: {result.content}")
                result_text = result.content[0].text
            tg.cancel_scope.cancel()
    assert result_text is not None
    return result_text


def _call_mcp_tool_sync(name: str, arguments: dict) -> str:
    """Only safe to call from a thread with no asyncio event loop already
    running — true for every current call site (the agent pipeline is
    fully synchronous). Would need `anyio.from_thread`/a background-loop
    bridge instead of `anyio.run` if ever called from an async context
    (e.g. directly inside a FastAPI async route) — not needed today."""
    return anyio.run(_call_mcp_tool_async, name, arguments)


def mcp_retrieve_policy_clauses(query: str, plan: str | None = None, n_results: int = 5) -> str:
    return _call_mcp_tool_sync("retrieve_policy_clauses", {"query": query, "plan": plan, "n_results": n_results})


def mcp_lookup_claim_history(patient_id: str) -> str:
    return _call_mcp_tool_sync("lookup_claim_history", {"patient_id": patient_id})


def mcp_policy_rag_tools() -> list[StructuredTool]:
    # Reuses the original LangChain tool's own `.description` (its
    # docstring) as the single source of truth — not the MCP-server
    # function's `__doc__`, which was never set (backend/mcp_server/server.py
    # passes the description via the `@mcp_server.tool(description=...)`
    # decorator argument instead of a docstring).
    return [
        StructuredTool.from_function(
            func=mcp_retrieve_policy_clauses,
            name="retrieve_policy_clauses",
            description=_retrieve_policy_clauses_tool.description,
        )
    ]


def mcp_fraud_detection_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=mcp_lookup_claim_history,
            name="lookup_claim_history",
            description=_lookup_claim_history_tool.description,
        )
    ]
