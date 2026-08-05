"""Protocol-level tests for the MCP server — real MCP client/server
round-trips over in-memory streams, zero Anthropic API cost (MCP is a
separate protocol between tool caller and tool server; no LLM is involved
in listing or calling a tool)."""

import anyio
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from backend.mcp_server.server import mcp_server


async def _run_server(server_read, server_write):
    await mcp_server._lowlevel_server.run(
        server_read, server_write, mcp_server._lowlevel_server.create_initialization_options()
    )


async def test_lists_both_tools_with_schemas():
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        async with anyio.create_task_group() as tg:
            tg.start_soon(_run_server, server_read, server_write)
            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert names == {"retrieve_policy_clauses", "lookup_claim_history"}
                tg.cancel_scope.cancel()


async def test_lookup_claim_history_returns_real_data():
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        async with anyio.create_task_group() as tg:
            tg.start_soon(_run_server, server_read, server_write)
            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                result = await session.call_tool("lookup_claim_history", {"patient_id": "PT_20322"})
                text = result.content[0].text
                assert "claim_PT_20322" in text
                assert "claim_PT_20322_dup" in text
                tg.cancel_scope.cancel()


async def test_lookup_claim_history_no_record_for_unknown_patient():
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        async with anyio.create_task_group() as tg:
            tg.start_soon(_run_server, server_read, server_write)
            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                result = await session.call_tool("lookup_claim_history", {"patient_id": "PT_NOT_REAL"})
                assert "No claim history on record" in result.content[0].text
                tg.cancel_scope.cancel()


async def test_retrieve_policy_clauses_returns_real_retrieval():
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        async with anyio.create_task_group() as tg:
            tg.start_soon(_run_server, server_read, server_write)
            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "retrieve_policy_clauses", {"query": "cosmetic surgery exclusions", "plan": "gold"}
                )
                text = result.content[0].text
                assert "plan=gold" in text
                tg.cancel_scope.cancel()
