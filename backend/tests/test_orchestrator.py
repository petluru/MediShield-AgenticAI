from unittest.mock import MagicMock, patch

from backend.agents.orchestrator import (
    _OrchestratorNarrative,
    compute_decision,
    decide,
    requires_human_review,
)
from backend.config import Settings
from backend.models import (
    CaseState,
    ClaimsOutput,
    ClassifierOutput,
    Decision,
    DocType,
    ExtractedClaimFields,
    FraudOutput,
    KYCOutput,
    PolicyOutput,
    RiskLevel,
)


def make_settings():
    return Settings(_env_file=None, ANTHROPIC_API_KEY="sk-ant-test")


def make_case(**overrides) -> CaseState:
    defaults = {"case_id": "case-1", "file_path": "dataset/claim_forms/x.png", "content_type": "image/png"}
    defaults.update(overrides)
    return CaseState(**defaults)  # type: ignore[arg-type]


def make_claims_fields(**overrides) -> ExtractedClaimFields:
    defaults = {
        "claim_amount": 1000.0,
        "icd10_codes": ["K35.80"],
        "cpt_codes": ["99213"],
        "provider_npi": "1234567890",
        "service_date": "01/01/2026",
    }
    defaults.update(overrides)
    return ExtractedClaimFields(**defaults)  # type: ignore[arg-type]


def test_compute_decision_approves_a_clean_case_with_no_results():
    settings = make_settings()
    decision, reasons = compute_decision(make_case(), settings)
    assert decision == Decision.APPROVE
    assert reasons


def test_compute_decision_escalates_on_kyc_image_processing_error_even_with_low_fraud():
    # Real bug fix (see IMPLEMENTATION_CHALLENGES.md and vision_utils.py's
    # ImageProcessingError): kyc_passed=False alone would fall into the
    # REJECT branch below, and requires_human_review only pauses REJECT
    # when fraud_score is already elevated — so a corrupted upload with a
    # low fraud score would have auto-REJECTed with no human ever seeing
    # it. Must escalate regardless of fraud score.
    settings = make_settings()
    case = make_case(
        kyc_result=KYCOutput(kyc_passed=False, flags=["image_processing_error"], confidence=0.0),
        fraud_result=FraudOutput(fraud_score=0.0, anomalies=[], risk_level=RiskLevel.LOW),
    )
    decision, reasons = compute_decision(case, settings)
    assert decision == Decision.ESCALATE
    assert any("could not process" in r for r in reasons)


def test_compute_decision_escalates_on_claims_image_processing_error_even_with_low_fraud():
    settings = make_settings()
    case = make_case(
        claims_result=ClaimsOutput(
            extracted_fields=make_claims_fields(),
            schema_valid=False,
            validation_errors=["image_processing_error"],
            confidence=0.0,
        ),
        fraud_result=FraudOutput(fraud_score=0.0, anomalies=[], risk_level=RiskLevel.LOW),
    )
    decision, reasons = compute_decision(case, settings)
    assert decision == Decision.ESCALATE
    assert any("could not process" in r for r in reasons)


def test_compute_decision_rejects_failed_kyc():
    settings = make_settings()
    case = make_case(kyc_result=KYCOutput(kyc_passed=False, flags=["expired"], confidence=0.9))
    decision, reasons = compute_decision(case, settings)
    assert decision == Decision.REJECT
    assert any("KYC" in r for r in reasons)


def test_compute_decision_rejects_uncovered_procedure():
    settings = make_settings()
    case = make_case(
        policy_result=PolicyOutput(
            covered=False, coverage_percentage=0.0, policy_clause="excluded", confidence=0.9
        )
    )
    decision, reasons = compute_decision(case, settings)
    assert decision == Decision.REJECT
    assert any("not covered" in r for r in reasons)


def test_compute_decision_rejects_invalid_claim_schema():
    settings = make_settings()
    case = make_case(
        claims_result=ClaimsOutput(
            extracted_fields=make_claims_fields(provider_npi=None),
            schema_valid=False,
            validation_errors=["provider NPI is missing"],
            confidence=0.9,
        )
    )
    decision, reasons = compute_decision(case, settings)
    assert decision == Decision.REJECT
    assert any("schema invalid" in r for r in reasons)


def test_compute_decision_reject_takes_priority_over_escalate():
    settings = make_settings()
    case = make_case(
        kyc_result=KYCOutput(kyc_passed=False, flags=["tampered"], confidence=0.9),
        fraud_result=FraudOutput(fraud_score=0.9, anomalies=["many red flags"], risk_level=RiskLevel.HIGH),
    )
    decision, _ = compute_decision(case, settings)
    assert decision == Decision.REJECT


def test_compute_decision_escalates_on_high_fraud_score():
    settings = make_settings()
    case = make_case(fraud_result=FraudOutput(fraud_score=0.4, anomalies=[], risk_level=RiskLevel.MEDIUM))
    decision, reasons = compute_decision(case, settings)
    assert decision == Decision.ESCALATE
    assert any("fraud score" in r for r in reasons)


def test_compute_decision_escalates_on_low_agent_confidence():
    settings = make_settings()
    case = make_case(classifier_result=ClassifierOutput(doc_type=DocType.CLAIM_FORM, confidence=0.4))
    decision, reasons = compute_decision(case, settings)
    assert decision == Decision.ESCALATE
    assert any("confidence" in r for r in reasons)


def test_compute_decision_escalates_on_unknown_doc_type_even_with_high_confidence():
    # Real bug (see IMPLEMENTATION_CHALLENGES.md): a confidently-misclassified
    # UNKNOWN document (e.g. a bank statement) never escalated before this
    # fix, because low-confidence was the only ESCALATE trigger and the
    # classifier can be perfectly sure it's looking at something unusable.
    settings = make_settings()
    case = make_case(
        classifier_result=ClassifierOutput(doc_type=DocType.UNKNOWN, confidence=0.97, routing_tags=["bank_statement"]),
        fraud_result=FraudOutput(fraud_score=0.05, anomalies=[], risk_level=RiskLevel.LOW),
    )
    decision, reasons = compute_decision(case, settings)
    assert decision == Decision.ESCALATE
    assert any("classified" in r for r in reasons)


def test_compute_decision_does_not_escalate_for_low_confidence_unknown_via_the_unknown_reason_specifically():
    # Both triggers can independently explain the same ESCALATE outcome —
    # this just confirms the new UNKNOWN check doesn't accidentally
    # suppress or duplicate-count the existing low-confidence trigger.
    settings = make_settings()
    case = make_case(classifier_result=ClassifierOutput(doc_type=DocType.UNKNOWN, confidence=0.3))
    decision, reasons = compute_decision(case, settings)
    assert decision == Decision.ESCALATE
    assert any("classified" in r for r in reasons)
    assert any("confidence" in r for r in reasons)


def test_compute_decision_approves_when_everything_passes():
    settings = make_settings()
    case = make_case(
        classifier_result=ClassifierOutput(doc_type=DocType.CLAIM_FORM, confidence=0.95),
        claims_result=ClaimsOutput(
            extracted_fields=make_claims_fields(), schema_valid=True, validation_errors=[], confidence=0.9
        ),
        policy_result=PolicyOutput(
            covered=True, coverage_percentage=80.0, policy_clause="covered", confidence=0.9
        ),
        fraud_result=FraudOutput(fraud_score=0.1, anomalies=[], risk_level=RiskLevel.LOW),
    )
    decision, _ = compute_decision(case, settings)
    assert decision == Decision.APPROVE


def test_requires_human_review_true_for_escalate():
    settings = make_settings()
    assert requires_human_review(Decision.ESCALATE, fraud_score=0.0, settings=settings) is True


def test_requires_human_review_true_for_reject_with_high_fraud():
    settings = make_settings()
    assert requires_human_review(Decision.REJECT, fraud_score=0.5, settings=settings) is True


def test_requires_human_review_false_for_reject_with_low_fraud():
    settings = make_settings()
    assert requires_human_review(Decision.REJECT, fraud_score=0.1, settings=settings) is False


def test_requires_human_review_false_for_approve():
    settings = make_settings()
    assert requires_human_review(Decision.APPROVE, fraud_score=0.0, settings=settings) is False


@patch("backend.agents.orchestrator.build_chat_anthropic")
def test_decide_uses_escalation_model_when_escalating(mock_build_llm):
    fake_llm = MagicMock()
    fake_structured = MagicMock()
    fake_structured.invoke.return_value = _OrchestratorNarrative(
        confidence=0.5, justification="ambiguous case", agent_summaries={"fraud": "elevated score"}
    )
    fake_llm.with_structured_output.return_value = fake_structured
    mock_build_llm.return_value = fake_llm

    settings = make_settings()
    case = make_case(fraud_result=FraudOutput(fraud_score=0.4, anomalies=[], risk_level=RiskLevel.MEDIUM))

    result = decide(case, settings=settings)

    assert result.decision == Decision.ESCALATE
    assert result.escalated_to_opus is True
    mock_build_llm.assert_called_once_with(settings.escalation_model, settings, agent="orchestrator")


@patch("backend.agents.orchestrator.build_chat_anthropic")
def test_decide_uses_base_model_when_not_escalating(mock_build_llm):
    fake_llm = MagicMock()
    fake_structured = MagicMock()
    fake_structured.invoke.return_value = _OrchestratorNarrative(
        confidence=0.95, justification="clean case", agent_summaries={}
    )
    fake_llm.with_structured_output.return_value = fake_structured
    mock_build_llm.return_value = fake_llm

    settings = make_settings()
    result = decide(make_case(), settings=settings)

    assert result.decision == Decision.APPROVE
    assert result.escalated_to_opus is False
    mock_build_llm.assert_called_once_with(settings.orchestrator_model, settings, agent="orchestrator")
