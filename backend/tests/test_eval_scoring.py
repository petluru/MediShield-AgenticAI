from backend.evals.scoring import (
    CaseResult,
    classification_accuracy,
    decision_correctness,
    extraction_completeness,
    weighted_score,
)
from backend.models import Decision, DocType


def make_result(**overrides) -> CaseResult:
    defaults = {
        "doc_id": "x",
        "category": "id_documents",
        "expected_doc_type": DocType.ID_DOCUMENT,
        "actual_doc_type": DocType.ID_DOCUMENT,
        "expected_decision": Decision.APPROVE,
        "actual_decision": Decision.APPROVE,
    }
    defaults.update(overrides)
    return CaseResult(**defaults)


def test_classification_accuracy_all_correct():
    results = [make_result(), make_result()]
    assert classification_accuracy(results) == 1.0


def test_classification_accuracy_partial():
    results = [make_result(), make_result(actual_doc_type=DocType.CLAIM_FORM)]
    assert classification_accuracy(results) == 0.5


def test_classification_accuracy_empty_results():
    assert classification_accuracy([]) == 0.0


def test_classification_accuracy_none_actual_counts_as_wrong():
    results = [make_result(actual_doc_type=None)]
    assert classification_accuracy(results) == 0.0


def test_decision_correctness_all_correct():
    results = [make_result(), make_result()]
    assert decision_correctness(results) == 1.0


def test_decision_correctness_mismatch():
    results = [make_result(expected_decision=Decision.REJECT, actual_decision=Decision.APPROVE)]
    assert decision_correctness(results) == 0.0


def test_extraction_completeness_full_fields():
    results = [
        make_result(
            category="claim_forms",
            extracted_field_presence={
                "claim_amount": True,
                "icd10_codes": True,
                "cpt_codes": True,
                "provider_npi": True,
            },
        )
    ]
    assert extraction_completeness(results) == 1.0


def test_extraction_completeness_partial_fields():
    results = [
        make_result(
            category="claim_forms",
            extracted_field_presence={
                "claim_amount": True,
                "icd10_codes": False,
                "cpt_codes": True,
                "provider_npi": False,
            },
        )
    ]
    assert extraction_completeness(results) == 0.5


def test_extraction_completeness_ignores_non_claim_form_categories():
    results = [make_result(category="id_documents")]
    assert extraction_completeness(results) == 0.0


def test_extraction_completeness_missing_presence_dict_counts_as_all_missing():
    results = [make_result(category="claim_forms", extracted_field_presence=None)]
    assert extraction_completeness(results) == 0.0


def test_weighted_score_perfect_run():
    results = [
        make_result(category="claim_forms", extracted_field_presence={f: True for f in
                    ("claim_amount", "icd10_codes", "cpt_codes", "provider_npi")})
        for _ in range(3)
    ]
    score = weighted_score(results)
    assert score.classification_accuracy == 1.0
    assert score.extraction_completeness == 1.0
    assert score.decision_correctness == 1.0
    assert score.auto_scored_weight == 0.65
    assert abs(score.auto_scored_subtotal - 0.65) < 1e-9
    assert score.passes_decision_correctness_bar is True
    assert score.not_auto_scored == {
        "policy_retrieval_quality": 0.15,
        "ui_functionality": 0.10,
        "code_quality_and_structure": 0.10,
    }


def test_weighted_score_failing_run():
    results = [make_result(expected_decision=Decision.ESCALATE, actual_decision=Decision.APPROVE)]
    score = weighted_score(results)
    assert score.decision_correctness == 0.0
    assert score.passes_decision_correctness_bar is False
