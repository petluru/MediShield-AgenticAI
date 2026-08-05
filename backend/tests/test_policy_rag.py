from unittest.mock import MagicMock, patch

from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langgraph.errors import GraphRecursionError

from backend.agents.policy_rag import _RECURSION_LIMIT, _build_query, determine_coverage
from backend.config import Settings
from backend.models import PolicyOutput


def make_settings():
    return Settings(_env_file=None, ANTHROPIC_API_KEY="sk-ant-test")


def test_build_query_includes_plan_cpt_and_icd10():
    query = _build_query("gold", ["29827", "90837"], ["K35.80"])
    assert "gold" in query
    assert "29827" in query and "90837" in query
    assert "K35.80" in query


def test_build_query_omits_diagnosis_clause_when_no_icd10_codes():
    query = _build_query("silver", ["21920"], None)
    assert "diagnosis codes" not in query


@patch("backend.agents.policy_rag.create_agent")
@patch("backend.agents.policy_rag.build_chat_anthropic")
def test_determine_coverage_returns_structured_response(mock_build_llm, mock_create_agent):
    expected = PolicyOutput(
        covered=False,
        coverage_percentage=0.0,
        policy_clause="Section 4.1: liposuction excluded",
        exclusions=["Cosmetic exclusion S4(a)"],
        confidence=0.9,
    )
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {"messages": [], "structured_response": expected}
    mock_create_agent.return_value = fake_agent

    result = determine_coverage("gold", ["21920"], settings=make_settings())

    assert result is expected
    invoke_args = fake_agent.invoke.call_args[0][0]
    assert "21920" in invoke_args["messages"][0]["content"]


@patch("backend.agents.policy_rag.create_agent")
@patch("backend.agents.policy_rag.build_chat_anthropic")
def test_determine_coverage_caps_the_tool_calling_loop(mock_build_llm, mock_create_agent):
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": PolicyOutput(covered=True, coverage_percentage=80.0, policy_clause="ok", confidence=0.9)
    }
    mock_create_agent.return_value = fake_agent

    determine_coverage("gold", ["99213"], settings=make_settings())

    call_kwargs = fake_agent.invoke.call_args.kwargs
    assert call_kwargs["config"] == {"recursion_limit": _RECURSION_LIMIT}


@patch("backend.agents.policy_rag.create_agent")
@patch("backend.agents.policy_rag.build_chat_anthropic")
def test_determine_coverage_enables_prompt_caching_middleware(mock_build_llm, mock_create_agent):
    # Token-cost optimization: the multi-turn tool loop resends its full
    # growing history every turn (measured: one real 5-turn case cost 32,540
    # input tokens, see TOKEN_OPTIMIZATION_PLAN.md). This middleware caches
    # the repeated prefix so repeat turns pay Anthropic's ~0.1x cache-read
    # rate instead of full price.
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": PolicyOutput(covered=True, coverage_percentage=80.0, policy_clause="ok", confidence=0.9)
    }
    mock_create_agent.return_value = fake_agent

    determine_coverage("gold", ["99213"], settings=make_settings())

    middleware = mock_create_agent.call_args.kwargs["middleware"]
    assert any(isinstance(m, AnthropicPromptCachingMiddleware) for m in middleware)


@patch("backend.agents.policy_rag.create_agent")
@patch("backend.agents.policy_rag.build_chat_anthropic")
def test_determine_coverage_falls_back_safely_when_the_loop_never_converges(mock_build_llm, mock_create_agent):
    # Regression test: a real run hit 103 tool-calling iterations on a query
    # the policy documents couldn't answer cleanly (see IMPLEMENTATION_CHALLENGES.md).
    # This must degrade to a low-confidence answer, not crash the case pipeline.
    fake_agent = MagicMock()
    fake_agent.invoke.side_effect = GraphRecursionError("recursion limit reached")
    mock_create_agent.return_value = fake_agent

    result = determine_coverage("gold", ["59400"], ["K35.80"], settings=make_settings())

    assert result.covered is False
    assert result.confidence == 0.0
