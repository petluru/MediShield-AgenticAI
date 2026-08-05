"""Tests for the LangChain-compatible MCP client tool adapters. Real MCP
protocol round-trips (in-process), zero Anthropic API cost — these are the
sync wrapper functions the (not-yet-wired-in, see client_tools.py's
docstring) production agents would call."""

from backend.mcp_server.client_tools import (
    mcp_fraud_detection_tools,
    mcp_lookup_claim_history,
    mcp_policy_rag_tools,
    mcp_retrieve_policy_clauses,
)


def test_mcp_lookup_claim_history_returns_real_data():
    result = mcp_lookup_claim_history("PT_20322")
    assert "claim_PT_20322" in result
    assert "claim_PT_20322_dup" in result


def test_mcp_lookup_claim_history_no_history():
    result = mcp_lookup_claim_history("PT_NOT_REAL")
    assert "No claim history on record" in result


def test_mcp_retrieve_policy_clauses_returns_real_retrieval():
    result = mcp_retrieve_policy_clauses("cosmetic surgery exclusions", plan="gold")
    assert "plan=gold" in result


def test_mcp_policy_rag_tools_has_correct_name_and_description():
    tools = mcp_policy_rag_tools()
    assert len(tools) == 1
    assert tools[0].name == "retrieve_policy_clauses"
    assert "Semantic search" in tools[0].description


def test_mcp_fraud_detection_tools_has_correct_name_and_description():
    tools = mcp_fraud_detection_tools()
    assert len(tools) == 1
    assert tools[0].name == "lookup_claim_history"
    assert "claim submission" in tools[0].description.lower()


def test_mcp_policy_rag_tool_is_invocable_and_matches_direct_call():
    # The MCP-transport tool and the direct-Python-call tool must return
    # identical data for the same input — proves the MCP round-trip
    # doesn't lose or alter anything, which is the whole point of it being
    # a safe drop-in replacement.
    from backend.rag.retrieval import retrieve_policy_clauses as direct_tool

    mcp_tools = mcp_policy_rag_tools()
    mcp_result = mcp_tools[0].invoke({"query": "cosmetic surgery exclusions", "plan": "gold"})
    direct_result = direct_tool.invoke({"query": "cosmetic surgery exclusions", "plan": "gold"})
    assert mcp_result == direct_result


def test_mcp_fraud_detection_tool_is_invocable_and_matches_direct_call():
    from backend.fraud.claim_history import lookup_claim_history as direct_tool

    mcp_tools = mcp_fraud_detection_tools()
    mcp_result = mcp_tools[0].invoke({"patient_id": "PT_20322"})
    direct_result = direct_tool.invoke({"patient_id": "PT_20322"})
    assert mcp_result == direct_result
