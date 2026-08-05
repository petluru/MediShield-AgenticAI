"""Weighted scoring against assignment_02_multimodal_ai.md's Evaluation
Criteria table (SS"Evaluation Criteria"). Pure functions over already-collected
per-case results — no Anthropic calls here, so this is fully unit-testable
with synthetic data (see backend/tests/test_eval_scoring.py) independent of
ever running the real harness.

Only 3 of the assignment's 6 weighted criteria are actually derivable from
`dataset/metadata.json` + the pipeline's own output — the other 3 require
things this repo doesn't have yet or that the assignment itself says need a
human:
- Policy Retrieval Quality (15%): the assignment's own description says
  "manual spot-check" — not something to fabricate an automated score for.
- UI Functionality (10%): no UI exists yet (task #15, unbuilt).
- Code Quality & Structure (10%): a structural/manual review criterion, not
  a per-case metric a dataset-driven harness can score.

Reporting a fabricated number for any of these would be worse than being
honest that they're out of scope for this harness — see `weighted_score`'s
`not_auto_scored` breakdown."""

from pydantic import BaseModel

from backend.models import Decision, DocType

CLASSIFICATION_ACCURACY_WEIGHT = 0.20
EXTRACTION_COMPLETENESS_WEIGHT = 0.20
POLICY_RETRIEVAL_QUALITY_WEIGHT = 0.15  # not auto-scored — manual spot-check per the assignment
DECISION_CORRECTNESS_WEIGHT = 0.25
UI_FUNCTIONALITY_WEIGHT = 0.10  # not auto-scored — no UI built yet
CODE_QUALITY_WEIGHT = 0.10  # not auto-scored — structural/manual review

REQUIRED_CLAIM_FIELDS = ("claim_amount", "icd10_codes", "cpt_codes", "provider_npi")

# Assignment's own passing bar (SS"Evaluation Criteria").
PASSING_OVERALL_MIN = 0.70
PASSING_DECISION_CORRECTNESS_MIN = 0.60


class CaseResult(BaseModel):
    doc_id: str
    category: str
    expected_doc_type: DocType
    actual_doc_type: DocType | None = None
    expected_decision: Decision
    actual_decision: Decision | None = None
    extracted_field_presence: dict[str, bool] | None = None  # claim_forms only
    error: str | None = None


def classification_accuracy(results: list[CaseResult]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.actual_doc_type == r.expected_doc_type)
    return correct / len(results)


def extraction_completeness(results: list[CaseResult]) -> float:
    claim_results = [r for r in results if r.category == "claim_forms"]
    if not claim_results:
        return 0.0
    per_case_scores = []
    for r in claim_results:
        presence = r.extracted_field_presence or {}
        found = sum(1 for field in REQUIRED_CLAIM_FIELDS if presence.get(field))
        per_case_scores.append(found / len(REQUIRED_CLAIM_FIELDS))
    return sum(per_case_scores) / len(per_case_scores)


def decision_correctness(results: list[CaseResult]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.actual_decision == r.expected_decision)
    return correct / len(results)


class WeightedScore(BaseModel):
    classification_accuracy: float
    extraction_completeness: float
    decision_correctness: float
    auto_scored_weight: float  # sum of the 3 weights actually contributing below
    auto_scored_subtotal: float  # weighted sum of the 3 metrics, out of auto_scored_weight
    not_auto_scored: dict[str, float]  # criterion name -> weight, needs manual/future work
    passes_decision_correctness_bar: bool
    passes_overall_bar_on_auto_scored_criteria_alone: bool


def weighted_score(results: list[CaseResult]) -> WeightedScore:
    ca = classification_accuracy(results)
    ec = extraction_completeness(results)
    dc = decision_correctness(results)

    auto_scored_weight = CLASSIFICATION_ACCURACY_WEIGHT + EXTRACTION_COMPLETENESS_WEIGHT + DECISION_CORRECTNESS_WEIGHT
    auto_scored_subtotal = (
        ca * CLASSIFICATION_ACCURACY_WEIGHT + ec * EXTRACTION_COMPLETENESS_WEIGHT + dc * DECISION_CORRECTNESS_WEIGHT
    )

    return WeightedScore(
        classification_accuracy=ca,
        extraction_completeness=ec,
        decision_correctness=dc,
        auto_scored_weight=auto_scored_weight,
        auto_scored_subtotal=auto_scored_subtotal,
        not_auto_scored={
            "policy_retrieval_quality": POLICY_RETRIEVAL_QUALITY_WEIGHT,
            "ui_functionality": UI_FUNCTIONALITY_WEIGHT,
            "code_quality_and_structure": CODE_QUALITY_WEIGHT,
        },
        passes_decision_correctness_bar=dc >= PASSING_DECISION_CORRECTNESS_MIN,
        # Can only ever be a lower bound until the 3 manual/UI criteria are
        # scored — true only if the auto-scored subtotal alone already
        # clears 70% of the FULL weight, which would mean passing is
        # guaranteed regardless of the other 3.
        passes_overall_bar_on_auto_scored_criteria_alone=auto_scored_subtotal >= PASSING_OVERALL_MIN,
    )
