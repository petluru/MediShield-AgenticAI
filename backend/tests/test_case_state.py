import pytest
from pydantic import ValidationError

from backend.models import (
    CaseState,
    CaseStatus,
    ClassifierOutput,
    Decision,
    DocType,
    ExtractedClaimFields,
    OrchestratorDecision,
)


def make_case() -> CaseState:
    return CaseState(
        case_id="case-1",
        file_path="dataset/claim_forms/claim_PT_19116.png",
        content_type="image/png",
    )


def test_case_state_defaults_to_received():
    case = make_case()
    assert case.status == CaseStatus.RECEIVED
    assert case.classifier_result is None
    assert case.errors == []


def test_case_state_accumulates_agent_outputs_independently():
    case = make_case()
    case.classifier_result = ClassifierOutput(doc_type=DocType.CLAIM_FORM, confidence=0.95, routing_tags=["claims"])
    case.status = CaseStatus.CLASSIFIED
    assert case.classifier_result.doc_type == DocType.CLAIM_FORM
    assert case.kyc_result is None  # untouched by the classifier write


def test_extracted_claim_fields_optional_by_default():
    fields = ExtractedClaimFields()
    assert fields.claim_amount is None
    assert fields.icd10_codes == []


def test_confidence_must_be_within_unit_interval():
    with pytest.raises(ValidationError):
        ClassifierOutput(doc_type=DocType.UNKNOWN, confidence=1.5)


def test_orchestrator_decision_requires_valid_enum():
    decision = OrchestratorDecision(decision=Decision.ESCALATE, confidence=0.5, justification="fraud score high")
    assert decision.decision == Decision.ESCALATE
    assert decision.escalated_to_opus is False
