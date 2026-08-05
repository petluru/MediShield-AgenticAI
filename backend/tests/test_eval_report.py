from backend.evals.report import format_report
from backend.evals.scoring import CaseResult, weighted_score
from backend.models import Decision, DocType


def make_result(**overrides) -> CaseResult:
    defaults = {
        "doc_id": "id_PT_1",
        "category": "id_documents",
        "expected_doc_type": DocType.ID_DOCUMENT,
        "actual_doc_type": DocType.ID_DOCUMENT,
        "expected_decision": Decision.APPROVE,
        "actual_decision": Decision.APPROVE,
    }
    defaults.update(overrides)
    return CaseResult(**defaults)


def test_format_report_includes_all_auto_scored_criteria():
    results = [make_result()]
    report = format_report(results, weighted_score(results))
    assert "Classification Accuracy" in report
    assert "Extraction Completeness" in report
    assert "Decision Correctness" in report


def test_format_report_flags_passing_decision_correctness():
    results = [make_result(), make_result(doc_id="id_PT_2")]
    report = format_report(results, weighted_score(results))
    assert "Decision Correctness >= 60%: **PASS**" in report


def test_format_report_flags_failing_decision_correctness():
    results = [make_result(expected_decision=Decision.REJECT, actual_decision=Decision.APPROVE)]
    report = format_report(results, weighted_score(results))
    assert "Decision Correctness >= 60%: **FAIL**" in report


def test_format_report_lists_mismatches():
    results = [
        make_result(doc_id="clean_case"),
        make_result(doc_id="bad_case", expected_decision=Decision.REJECT, actual_decision=Decision.APPROVE),
    ]
    report = format_report(results, weighted_score(results))
    assert "bad_case" in report
    assert "clean_case" not in report.split("## Mismatches")[1].split("## Category")[0]


def test_format_report_omits_mismatches_section_when_none():
    results = [make_result()]
    report = format_report(results, weighted_score(results))
    assert "## Mismatches" not in report


def test_format_report_includes_error_cases_as_mismatches():
    results = [make_result(actual_doc_type=None, actual_decision=None, error="RuntimeError: boom")]
    report = format_report(results, weighted_score(results))
    assert "## Mismatches" in report
    assert "boom" in report


def test_format_report_category_breakdown():
    results = [make_result(category="id_documents"), make_result(category="claim_forms", doc_id="claim_PT_1")]
    report = format_report(results, weighted_score(results))
    assert "id_documents" in report
    assert "claim_forms" in report
