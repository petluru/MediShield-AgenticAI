from backend.evals.ground_truth import derive_expected_decision, load_eval_cases
from backend.models import Decision, DocType


def test_derive_expected_decision_uses_explicit_field_when_present():
    doc = {"category": "unknown", "expected_decision": "ESCALATE"}
    assert derive_expected_decision(doc) == Decision.ESCALATE


def test_derive_expected_decision_expired_id_rejects():
    doc = {"category": "id_documents", "edge_flags": ["expired_id"], "fraud_label": False}
    assert derive_expected_decision(doc) == Decision.REJECT


def test_derive_expected_decision_tampered_id_rejects():
    doc = {"category": "id_documents", "edge_flags": ["tampered_id"], "fraud_label": False}
    assert derive_expected_decision(doc) == Decision.REJECT


def test_derive_expected_decision_expiring_soon_id_does_not_reject():
    # backend/agents/kyc.py's own system prompt: expiring soon still passes.
    doc = {"category": "id_documents", "edge_flags": ["expiring_soon_id"], "fraud_label": False}
    assert derive_expected_decision(doc) == Decision.APPROVE


def test_derive_expected_decision_missing_fields_on_claim_form_rejects():
    doc = {"category": "claim_forms", "edge_flags": ["missing_fields"], "fraud_label": False}
    assert derive_expected_decision(doc) == Decision.REJECT


def test_derive_expected_decision_missing_fields_on_prescription_does_not_reject():
    # Prescriptions never go through the Claims Agent (routing only checks
    # claim_forms) — missing_fields there has no REJECT-triggering effect.
    doc = {"category": "prescriptions", "edge_flags": ["missing_fields"], "fraud_label": False}
    assert derive_expected_decision(doc) == Decision.APPROVE


def test_derive_expected_decision_uncovered_procedure_rejects():
    doc = {"category": "claim_forms", "edge_flags": ["uncovered_procedure"], "fraud_label": False}
    assert derive_expected_decision(doc) == Decision.REJECT


def test_derive_expected_decision_fraud_label_escalates():
    doc = {"category": "discharge_summaries", "edge_flags": [], "fraud_label": True}
    assert derive_expected_decision(doc) == Decision.ESCALATE


def test_derive_expected_decision_clean_document_approves():
    doc = {"category": "policy_amendments", "edge_flags": ["has_supporting_docs"], "fraud_label": False}
    assert derive_expected_decision(doc) == Decision.APPROVE


def test_derive_expected_decision_reject_takes_priority_over_fraud_label():
    doc = {"category": "id_documents", "edge_flags": ["expired_id"], "fraud_label": True}
    assert derive_expected_decision(doc) == Decision.REJECT


def test_load_eval_cases_against_the_real_dataset():
    # Real local file, zero API cost — this is the same distribution
    # verified by hand against dataset/metadata.json before this module
    # was written.
    cases = load_eval_cases()
    assert len(cases) == 155

    from collections import Counter

    counts = Counter(c.expected_decision for c in cases)
    assert counts[Decision.APPROVE] == 128
    assert counts[Decision.ESCALATE] == 10
    assert counts[Decision.REJECT] == 17


def test_load_eval_cases_file_paths_all_exist():
    import os

    cases = load_eval_cases()
    missing = [c.doc_id for c in cases if not os.path.exists(c.file_path)]
    assert missing == []


def test_load_eval_cases_category_maps_to_correct_doctype():
    cases = load_eval_cases()
    by_id = {c.doc_id: c for c in cases}
    assert by_id["id_PT_19116"].expected_doc_type == DocType.ID_DOCUMENT
    assert by_id["claim_PT_19116"].expected_doc_type == DocType.CLAIM_FORM
    assert by_id["unknown_blurry_scan_001"].expected_doc_type == DocType.UNKNOWN
