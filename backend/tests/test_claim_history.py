import json

from backend.fraud.claim_history import (
    _clear_cache_for_tests,
    get_patient_claim_history,
    lookup_claim_history,
)

FIXTURE = [
    {
        "doc_id": "claim_PT_1_dup",
        "category": "claim_forms",
        "claim_form_type": "CMS1500",
        "case_cluster_id": "C_001",
        "patient_id": "PT_1",
        "policy_number": "MED-GLD-1",
        "fraud_label": True,
        "fraud_reason": "duplicate_claim",
        "edge_flags": ["duplicate_of_claim_PT_1"],
    },
    {
        "doc_id": "claim_PT_1",
        "category": "claim_forms",
        "claim_form_type": "CMS1500",
        "case_cluster_id": "C_001",
        "patient_id": "PT_1",
        "policy_number": "MED-GLD-1",
        "fraud_label": False,
        "fraud_reason": None,
        "edge_flags": [],
    },
    {
        "doc_id": "claim_PT_2",
        "category": "claim_forms",
        "claim_form_type": "UB04",
        "case_cluster_id": "C_002",
        "patient_id": "PT_2",
        "policy_number": "MED-GLD-2",
        "fraud_label": False,
        "fraud_reason": None,
        "edge_flags": [],
    },
    {
        "doc_id": "id_PT_2",
        "category": "id_documents",
        "case_cluster_id": "C_002",
        "patient_id": "PT_2",
        "policy_number": "MED-GLD-2",
    },
]


def write_fixture(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(FIXTURE), encoding="utf-8")
    _clear_cache_for_tests()
    return path


def test_get_patient_claim_history_finds_duplicate_submissions(tmp_path):
    path = write_fixture(tmp_path)
    history = get_patient_claim_history("PT_1", metadata_path=path)
    assert len(history) == 2
    assert {h["doc_id"] for h in history} == {"claim_PT_1", "claim_PT_1_dup"}


def test_get_patient_claim_history_never_exposes_fraud_ground_truth(tmp_path):
    path = write_fixture(tmp_path)
    history = get_patient_claim_history("PT_1", metadata_path=path)
    for entry in history:
        assert "fraud_label" not in entry
        assert "fraud_reason" not in entry
        assert "edge_flags" not in entry


def test_get_patient_claim_history_excludes_other_categories(tmp_path):
    path = write_fixture(tmp_path)
    history = get_patient_claim_history("PT_2", metadata_path=path)
    assert len(history) == 1
    assert history[0]["doc_id"] == "claim_PT_2"


def test_get_patient_claim_history_empty_for_unknown_patient(tmp_path):
    path = write_fixture(tmp_path)
    assert get_patient_claim_history("PT_999", metadata_path=path) == []


def test_lookup_claim_history_tool_reports_duplicate_count(tmp_path, monkeypatch):
    path = write_fixture(tmp_path)
    monkeypatch.setattr(
        "backend.fraud.claim_history.get_patient_claim_history",
        lambda patient_id: get_patient_claim_history(patient_id, metadata_path=path),
    )
    result = lookup_claim_history.invoke({"patient_id": "PT_1"})
    assert "2 claim submission(s)" in result
    assert "claim_PT_1_dup" in result


def test_lookup_claim_history_tool_reports_no_history():
    result = lookup_claim_history.invoke({"patient_id": "PT_UNKNOWN_XYZ"})
    assert "No claim history" in result
