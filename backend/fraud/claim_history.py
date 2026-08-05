"""Patient claim-history lookup for the Fraud Detection Agent (task #13's
MCP server will eventually wrap this same function as an MCP tool, rather
than duplicating the lookup logic — same pattern as backend/rag/retrieval.py).

Deliberately exposes only structural fields (doc ids, case clustering, claim
form type) from `dataset/metadata.json` — never `fraud_label`,
`fraud_reason`, or `edge_flags`, which are this project's eval ground truth
(task #11) and must not leak into what the agent reasons over. A patient
with two on-record claim submissions is a legitimate duplicate-detection
signal the agent should notice itself from the raw list, not be handed."""

import json
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from langchain_core.tools import tool

from backend.config import Settings, get_settings


class ClaimHistoryEntry(TypedDict):
    doc_id: str
    claim_form_type: str | None
    case_cluster_id: str | None
    policy_number: str | None


@lru_cache
def _load_metadata(metadata_path: str) -> list[dict]:
    with open(metadata_path, encoding="utf-8") as f:
        result: list[dict] = json.load(f)
        return result


def get_patient_claim_history(
    patient_id: str, settings: Settings | None = None, metadata_path: Path | None = None
) -> list[ClaimHistoryEntry]:
    """All claim_form submissions on record for a patient, structural fields
    only. `metadata_path` overrides the default (dataset/metadata.json) —
    used by tests to point at an isolated fixture."""
    settings = settings or get_settings()
    path = metadata_path or settings.resolved_path("dataset/metadata.json")
    records = _load_metadata(str(path))
    return [
        {
            "doc_id": record["doc_id"],
            "claim_form_type": record.get("claim_form_type"),
            "case_cluster_id": record.get("case_cluster_id"),
            "policy_number": record.get("policy_number"),
        }
        for record in records
        if record.get("category") == "claim_forms" and record.get("patient_id") == patient_id
    ]


def _clear_cache_for_tests() -> None:
    """Test-only helper — the metadata cache is keyed by path, so a test
    fixture pointing at a temp file needs a clean slate."""
    _load_metadata.cache_clear()


@tool
def lookup_claim_history(patient_id: str) -> str:
    """Look up this patient's on-record claim submissions by patient ID.
    Returns each submission's document id, claim form type, and case
    cluster. Use this to check for duplicate submissions (more than one
    claim on record) or an unusually high submission count. This is our
    own internal claims database, not externally-supplied document text.
    """
    history = get_patient_claim_history(patient_id)
    if not history:
        return f"No claim history on record for patient {patient_id}."
    lines = [
        f"- {entry['doc_id']} (form type: {entry['claim_form_type']}, cluster: {entry['case_cluster_id']})"
        for entry in history
    ]
    return f"{len(history)} claim submission(s) on record for patient {patient_id}:\n" + "\n".join(lines)
