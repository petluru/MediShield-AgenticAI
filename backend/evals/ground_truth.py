"""Derives eval ground truth from `dataset/metadata.json` — task #9's "scored
against dataset/metadata.json" requirement (PROJECT_PLAN.md SS2 item 9).

**One CaseState = one document** (see backend/graph/pipeline.py's own design
note), so ground truth is derived per document, not per case-cluster — each
of a cluster's 5 documents (id/claim/discharge/rx/amendment) is its own
independent case with its own expected decision, matching how the pipeline
actually runs them.

Decision derivation follows the assignment's own rule (assignment_02_
multimodal_ai.md SS8) and only the checks the current pipeline routing
actually performs for that document's category (backend/graph/pipeline.py's
`_route_after_classification`/`_route_after_claims`):
- `id_documents` are the only category KYC ever evaluates — `expired_id` or
  `tampered_id` fails KYC -> REJECT. `expiring_soon_id` does NOT fail KYC
  (backend/agents/kyc.py's own system prompt: still passes, just flagged).
- `claim_forms` are the only category Claims/Policy ever evaluate —
  `missing_fields` -> invalid schema -> REJECT; `uncovered_procedure` ->
  not covered -> REJECT.
- Any document whose OWN `fraud_label` is true -> ESCALATE (fraud only ever
  escalates in the assignment's rule table, never rejects on its own).
- `unknown/` category entries carry an explicit `expected_decision` in
  metadata already (all 4 are ESCALATE) — used directly, not re-derived.
- Everything else -> APPROVE.

**Known detectability gap, not a ground-truth error:** `missing_fields` and
`fraud_label` also appear on `prescriptions`/`discharge_summaries` entries
in the raw dataset (e.g. a discharge summary carrying a `readmission_30d`
signal) — those are deliberately NOT treated as REJECT/ESCALATE triggers
here, because the current pipeline routes those categories straight to
Fraud Detection (no Claims/KYC check ever runs on them), and
`backend/fraud/claim_history.py`'s tool only exposes claim_forms metadata
(doc_id/type/cluster/policy — no dates, amounts, or diagnosis codes), so
Fraud Detection has no way to see a discharge summary's own field values
regardless of which document triggered the case. Ground truth reflects
what the assignment's rule says SHOULD happen for that document; the eval
harness comparing this against the pipeline's ACTUAL decision is exactly
what surfaces this kind of structural detection gap — see the harness
report, not this module, for that analysis."""

from pathlib import Path

from pydantic import BaseModel

from backend.config import REPO_ROOT, Settings, get_settings
from backend.models import Decision, DocType

_CATEGORY_TO_DOCTYPE: dict[str, DocType] = {
    "claim_forms": DocType.CLAIM_FORM,
    "id_documents": DocType.ID_DOCUMENT,
    "discharge_summaries": DocType.DISCHARGE_SUMMARY,
    "prescriptions": DocType.PRESCRIPTION,
    "policy_amendments": DocType.POLICY_AMENDMENT,
    "unknown": DocType.UNKNOWN,
}


class EvalCase(BaseModel):
    doc_id: str
    category: str
    file_path: str
    patient_id: str | None
    policy_number: str | None
    expected_doc_type: DocType
    expected_decision: Decision


def derive_expected_decision(doc: dict) -> Decision:
    if "expected_decision" in doc:
        return Decision(doc["expected_decision"])

    category = doc["category"]
    flags = set(doc.get("edge_flags") or [])

    if category == "id_documents" and (flags & {"expired_id", "tampered_id"}):
        return Decision.REJECT
    if category == "claim_forms" and "missing_fields" in flags:
        return Decision.REJECT
    if category == "claim_forms" and "uncovered_procedure" in flags:
        return Decision.REJECT
    if doc.get("fraud_label"):
        return Decision.ESCALATE
    return Decision.APPROVE


def _real_file_path(doc: dict, settings: Settings) -> str:
    # dataset/metadata.json's own `file_path` field is stale (points at the
    # original sandbox's path) for every category except `unknown` — real
    # files live at dataset/<category>/<doc_id>.png (see
    # [[medishield-env-gotchas]] / IMPLEMENTATION_CHALLENGES.md).
    if doc["category"] == "unknown":
        return doc["file_path"]
    return str(settings.resolved_path(f"dataset/{doc['category']}/{doc['doc_id']}.png"))


def load_eval_cases(
    metadata_path: Path | None = None, settings: Settings | None = None
) -> list[EvalCase]:
    import json

    settings = settings or get_settings()
    path = metadata_path or (REPO_ROOT / "dataset" / "metadata.json")
    with path.open(encoding="utf-8") as f:
        raw_docs: list[dict] = json.load(f)

    cases = []
    for doc in raw_docs:
        cases.append(
            EvalCase(
                doc_id=doc["doc_id"],
                category=doc["category"],
                file_path=_real_file_path(doc, settings),
                patient_id=doc.get("patient_id"),
                policy_number=doc.get("policy_number"),
                expected_doc_type=_CATEGORY_TO_DOCTYPE[doc["category"]],
                expected_decision=derive_expected_decision(doc),
            )
        )
    return cases
