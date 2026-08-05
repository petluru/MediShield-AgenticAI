from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver

from backend.config import Settings
from backend.evals.ground_truth import EvalCase
from backend.evals.harness import read_case_result_from_checkpoint, run_case, run_eval_suite
from backend.graph.pipeline import build_case_graph
from backend.models import (
    ClaimsOutput,
    ClassifierOutput,
    Decision,
    DocType,
    ExtractedClaimFields,
    FraudOutput,
    KYCOutput,
    OrchestratorDecision,
    RiskLevel,
)


def make_settings():
    return Settings(_env_file=None, ANTHROPIC_API_KEY="sk-ant-test")


def make_graph():
    return build_case_graph(settings=make_settings(), checkpointer=InMemorySaver())


def make_case(**overrides) -> EvalCase:
    defaults = {
        "doc_id": "id_PT_1",
        "category": "id_documents",
        "file_path": "dataset/id_documents/id_PT_1.png",
        "patient_id": "PT_1",
        "policy_number": "MED-GLD-1",
        "expected_doc_type": DocType.ID_DOCUMENT,
        "expected_decision": Decision.APPROVE,
    }
    defaults.update(overrides)
    return EvalCase(**defaults)


@patch("backend.graph.pipeline.decide")
@patch("backend.graph.pipeline.assess_fraud_risk")
@patch("backend.graph.pipeline.verify_kyc")
@patch("backend.graph.pipeline.classify_document")
def test_run_case_records_a_correct_match(mock_classify, mock_kyc, mock_fraud, mock_decide):
    mock_classify.return_value = ClassifierOutput(doc_type=DocType.ID_DOCUMENT, confidence=0.95)
    mock_kyc.return_value = KYCOutput(kyc_passed=True, flags=[], confidence=0.95)
    mock_fraud.return_value = FraudOutput(fraud_score=0.05, anomalies=[], risk_level=RiskLevel.LOW)
    mock_decide.return_value = OrchestratorDecision(decision=Decision.APPROVE, confidence=0.95, justification="clean")

    result = run_case(make_graph(), make_case())

    assert result.error is None
    assert result.actual_doc_type == DocType.ID_DOCUMENT
    assert result.actual_decision == Decision.APPROVE
    assert result.extracted_field_presence is None  # not a claim_forms case


@patch("backend.graph.pipeline.decide")
@patch("backend.graph.pipeline.assess_fraud_risk")
@patch("backend.graph.pipeline.determine_coverage")
@patch("backend.graph.pipeline.extract_claim")
@patch("backend.graph.pipeline.classify_document")
def test_run_case_records_extraction_field_presence_for_claim_forms(
    mock_classify, mock_claims, mock_policy, mock_fraud, mock_decide
):
    mock_classify.return_value = ClassifierOutput(doc_type=DocType.CLAIM_FORM, confidence=0.95)
    mock_claims.return_value = ClaimsOutput(
        extracted_fields=ExtractedClaimFields(
            claim_amount=100.0, icd10_codes=["K35.80"], cpt_codes=[], provider_npi=None
        ),
        schema_valid=False,
        validation_errors=["no CPT codes"],
        confidence=0.9,
    )
    mock_fraud.return_value = FraudOutput(fraud_score=0.05, anomalies=[], risk_level=RiskLevel.LOW)
    mock_decide.return_value = OrchestratorDecision(decision=Decision.REJECT, confidence=0.9, justification="invalid")

    case = make_case(
        doc_id="claim_PT_1",
        category="claim_forms",
        file_path="dataset/claim_forms/claim_PT_1.png",
        expected_doc_type=DocType.CLAIM_FORM,
        expected_decision=Decision.REJECT,
    )
    result = run_case(make_graph(), case)

    assert result.error is None
    assert result.extracted_field_presence == {
        "claim_amount": True,
        "icd10_codes": True,
        "cpt_codes": False,
        "provider_npi": False,
    }
    mock_policy.assert_not_called()  # invalid schema -> never reaches Policy RAG


@patch("backend.graph.pipeline.classify_document")
def test_run_case_records_error_without_crashing(mock_classify):
    mock_classify.side_effect = RuntimeError("simulated agent failure")

    result = run_case(make_graph(), make_case())

    assert result.error is not None
    assert "simulated agent failure" in result.error
    assert result.actual_doc_type is None
    assert result.actual_decision is None


@patch("backend.graph.pipeline.decide")
@patch("backend.graph.pipeline.assess_fraud_risk")
@patch("backend.graph.pipeline.verify_kyc")
@patch("backend.graph.pipeline.classify_document")
def test_run_eval_suite_continues_after_one_case_errors(mock_classify, mock_kyc, mock_fraud, mock_decide):
    mock_classify.side_effect = [
        RuntimeError("first case fails"),
        ClassifierOutput(doc_type=DocType.ID_DOCUMENT, confidence=0.95),
    ]
    mock_kyc.return_value = KYCOutput(kyc_passed=True, flags=[], confidence=0.95)
    mock_fraud.return_value = FraudOutput(fraud_score=0.05, anomalies=[], risk_level=RiskLevel.LOW)
    mock_decide.return_value = OrchestratorDecision(decision=Decision.APPROVE, confidence=0.95, justification="clean")

    cases = [make_case(doc_id="id_PT_1"), make_case(doc_id="id_PT_2")]
    results = run_eval_suite(make_graph(), cases)

    assert len(results) == 2
    assert results[0].error is not None
    assert results[1].error is None
    assert results[1].actual_decision == Decision.APPROVE


@patch("backend.graph.pipeline.decide")
@patch("backend.graph.pipeline.assess_fraud_risk")
@patch("backend.graph.pipeline.verify_kyc")
@patch("backend.graph.pipeline.classify_document")
def test_read_case_result_from_checkpoint_matches_the_original_run(mock_classify, mock_kyc, mock_fraud, mock_decide):
    # Same graph/checkpointer instance reused for both calls — this is what
    # lets separate --category eval stages be combined for free afterward.
    mock_classify.return_value = ClassifierOutput(doc_type=DocType.ID_DOCUMENT, confidence=0.95)
    mock_kyc.return_value = KYCOutput(kyc_passed=True, flags=[], confidence=0.95)
    mock_fraud.return_value = FraudOutput(fraud_score=0.05, anomalies=[], risk_level=RiskLevel.LOW)
    mock_decide.return_value = OrchestratorDecision(decision=Decision.APPROVE, confidence=0.95, justification="clean")

    graph = make_graph()
    case = make_case()
    original = run_case(graph, case)

    replayed = read_case_result_from_checkpoint(graph, case)

    assert replayed.actual_doc_type == original.actual_doc_type
    assert replayed.actual_decision == original.actual_decision
    assert replayed.error is None


def test_read_case_result_from_checkpoint_errors_for_a_case_never_run():
    graph = make_graph()
    result = read_case_result_from_checkpoint(graph, make_case(doc_id="never_ran"))

    assert result.error is not None
    assert "never run" in result.error
