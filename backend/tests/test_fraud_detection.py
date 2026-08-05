from unittest.mock import MagicMock, patch

from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError

from backend.agents.fraud_detection import (
    _RECURSION_LIMIT,
    _build_query,
    _FraudAssessment,
    _risk_level_for_score,
    assess_fraud_risk,
)
from backend.config import Settings
from backend.models import RiskLevel


def make_tool_message(content: str) -> ToolMessage:
    return ToolMessage(content=content, name="lookup_claim_history", tool_call_id="call_1")


def make_settings():
    return Settings(_env_file=None, ANTHROPIC_API_KEY="sk-ant-test")


def test_risk_level_low_below_escalate_min():
    settings = make_settings()
    assert _risk_level_for_score(0.1, settings) == RiskLevel.LOW


def test_risk_level_medium_between_thresholds():
    settings = make_settings()
    assert _risk_level_for_score(0.3, settings) == RiskLevel.MEDIUM
    assert _risk_level_for_score(0.59, settings) == RiskLevel.MEDIUM


def test_risk_level_high_at_or_above_threshold():
    settings = make_settings()
    assert _risk_level_for_score(0.6, settings) == RiskLevel.HIGH
    assert _risk_level_for_score(1.0, settings) == RiskLevel.HIGH


def test_build_query_includes_all_provided_fields():
    query = _build_query("PT_1", 4095.0, ["90837", "93000"], "11/30/2025", "NPI-1234567890")
    assert "PT_1" in query
    assert "$4,095.00" in query
    assert "90837" in query and "93000" in query
    assert "11/30/2025" in query
    assert "NPI-1234567890" in query
    assert "untrusted" in query


def test_build_query_omits_unset_optional_fields():
    query = _build_query("PT_1", None, [], None, None)
    assert "Claim amount" not in query
    assert "CPT/procedure codes" not in query
    assert "Service date" not in query
    assert "Provider NPI" not in query


@patch("backend.agents.fraud_detection.create_agent")
@patch("backend.agents.fraud_detection.build_chat_anthropic")
def test_assess_fraud_risk_does_not_escalate_for_clear_cut_score(mock_build_llm, mock_create_agent):
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": _FraudAssessment(fraud_score=0.05, anomalies=[])
    }
    mock_create_agent.return_value = fake_agent

    result = assess_fraud_risk("PT_1", settings=make_settings())

    assert result.fraud_score == 0.05
    assert result.risk_level == RiskLevel.LOW
    assert result.escalated_to_opus is False
    mock_create_agent.assert_called_once()  # only one pass — no escalation


@patch("backend.agents.fraud_detection.create_agent")
@patch("backend.agents.fraud_detection.build_chat_anthropic")
def test_assess_fraud_risk_escalates_without_repeating_the_tool_loop(mock_build_llm, mock_create_agent):
    # Token-cost optimization (TOKEN_OPTIMIZATION_PLAN.md): escalation must
    # reuse the deterministic claim-history lookup instead of re-running a
    # second full tool-calling loop. Only one create_agent pass (Sonnet's)
    # should happen; Opus's escalation is a single-shot call.
    sonnet_agent = MagicMock()
    sonnet_agent.invoke.return_value = {
        "structured_response": _FraudAssessment(fraud_score=0.35, anomalies=["duplicate submission found"]),
        "messages": [make_tool_message("2 claim submission(s) on record for patient PT_1: - doc_a - doc_b")],
    }
    mock_create_agent.return_value = sonnet_agent

    opus_llm = MagicMock()
    opus_llm.with_structured_output.return_value.invoke.return_value = _FraudAssessment(
        fraud_score=0.42, anomalies=["confirmed duplicate submission"]
    )
    mock_build_llm.return_value = opus_llm

    result = assess_fraud_risk("PT_1", settings=make_settings())

    mock_create_agent.assert_called_once()  # no second tool loop for Opus
    assert result.fraud_score == 0.42  # uses the escalated (opus) result, not the sonnet one
    assert result.escalated_to_opus is True
    assert result.risk_level == RiskLevel.MEDIUM

    # Opus must see the reused deterministic fact...
    opus_messages = opus_llm.with_structured_output.return_value.invoke.call_args[0][0]
    opus_prompt_text = str(opus_messages)
    assert "2 claim submission(s) on record for patient PT_1" in opus_prompt_text
    # ...but never Sonnet's own reasoning/narrative.
    assert "duplicate submission found" not in opus_prompt_text
    assert "0.35" not in opus_prompt_text


@patch("backend.agents.fraud_detection.create_agent")
@patch("backend.agents.fraud_detection.build_chat_anthropic")
def test_assess_fraud_risk_escalation_falls_back_to_full_loop_without_captured_history(
    mock_build_llm, mock_create_agent
):
    # If Sonnet's pass never captured a claim-history lookup (no matching
    # ToolMessage in its trace), escalation must not proceed with no facts —
    # it falls back to a full, independent tool loop for Opus.
    sonnet_agent = MagicMock()
    sonnet_agent.invoke.return_value = {
        "structured_response": _FraudAssessment(fraud_score=0.35, anomalies=[]),
        "messages": [],  # no lookup_claim_history call captured
    }
    opus_agent = MagicMock()
    opus_agent.invoke.return_value = {
        "structured_response": _FraudAssessment(fraud_score=0.42, anomalies=["confirmed duplicate submission"]),
        "messages": [],
    }
    mock_create_agent.side_effect = [sonnet_agent, opus_agent]

    result = assess_fraud_risk("PT_1", settings=make_settings())

    assert mock_create_agent.call_count == 2  # fallback: full loop for both passes
    assert result.fraud_score == 0.42
    assert result.escalated_to_opus is True


@patch("backend.agents.fraud_detection.create_agent")
@patch("backend.agents.fraud_detection.build_chat_anthropic")
def test_assess_fraud_risk_does_not_escalate_for_clearly_fraudulent_score(mock_build_llm, mock_create_agent):
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": _FraudAssessment(fraud_score=0.9, anomalies=["multiple red flags"])
    }
    mock_create_agent.return_value = fake_agent

    result = assess_fraud_risk("PT_1", settings=make_settings())

    assert result.escalated_to_opus is False
    assert result.risk_level == RiskLevel.HIGH
    mock_create_agent.assert_called_once()


@patch("backend.agents.fraud_detection.create_agent")
@patch("backend.agents.fraud_detection.build_chat_anthropic")
def test_assess_fraud_risk_caps_the_tool_calling_loop(mock_build_llm, mock_create_agent):
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {"structured_response": _FraudAssessment(fraud_score=0.05, anomalies=[])}
    mock_create_agent.return_value = fake_agent

    assess_fraud_risk("PT_1", settings=make_settings())

    call_kwargs = fake_agent.invoke.call_args.kwargs
    assert call_kwargs["config"] == {"recursion_limit": _RECURSION_LIMIT}


@patch("backend.agents.fraud_detection.create_agent")
@patch("backend.agents.fraud_detection.build_chat_anthropic")
def test_assess_fraud_risk_enables_prompt_caching_middleware(mock_build_llm, mock_create_agent):
    # Same optimization as Policy RAG's identical create_agent pattern —
    # see TOKEN_OPTIMIZATION_PLAN.md.
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {"structured_response": _FraudAssessment(fraud_score=0.05, anomalies=[])}
    mock_create_agent.return_value = fake_agent

    assess_fraud_risk("PT_1", settings=make_settings())

    middleware = mock_create_agent.call_args.kwargs["middleware"]
    assert any(isinstance(m, AnthropicPromptCachingMiddleware) for m in middleware)


@patch("backend.agents.fraud_detection.create_agent")
@patch("backend.agents.fraud_detection.build_chat_anthropic")
def test_assess_fraud_risk_falls_back_to_escalate_when_the_loop_never_converges(mock_build_llm, mock_create_agent):
    # Same regression class as Policy RAG's identical create_agent pattern
    # (see IMPLEMENTATION_CHALLENGES.md) — must degrade to a safe, reviewable
    # result rather than crash the case pipeline.
    fake_agent = MagicMock()
    fake_agent.invoke.side_effect = GraphRecursionError("recursion limit reached")
    mock_create_agent.return_value = fake_agent

    result = assess_fraud_risk("PT_1", settings=make_settings())

    assert result.fraud_score >= make_settings().fraud_escalate_min
    assert result.anomalies
