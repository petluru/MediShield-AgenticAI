"""Component 8 (part 2): the LangGraph StateGraph wiring every agent built
so far into one pipeline over a single CaseState — RECEIVED -> CLASSIFIED ->
[routed specialists] -> FRAUD_CHECK -> AGGREGATED -> DECIDED, with an
optional AWAITING_REVIEW pause (Component 9's human-in-the-loop gate,
PROJECT_PLAN.md SS6) between the Orchestrator's decision and the graph
actually finishing.

Design note on "one CaseState = one document" vs. the assignment diagram's
"[PARALLEL: KYC + CLAIMS + POLICY]" box: this project models one case as one
uploaded document (Component 1: "assigns a unique case_id per submission"),
a decision made in backend/models/case_state.py and used consistently by
every agent built so far. Under that model, doc_type routing after
CLASSIFIED is mutually exclusive per document (a CLAIM_FORM is never also
an ID_DOCUMENT), so exactly one specialist path fires per case — the
"parallel" branches in the assignment's diagram describe the pipeline's
conceptual fan-out across the *different* document types a submission can
be, not concurrent execution within a single case. Real concurrency (many
cases in flight at once) belongs to the API layer (task #14), not this
graph. KYC/CLAIMS/POLICY still converge into a single Fraud Detection node
before the Orchestrator, matching the diagram's funnel point.

Routing after CLASSIFIED:
- CLAIM_FORM  -> Claims -> (if schema_valid and has CPT codes) -> Policy -> Fraud
- ID_DOCUMENT -> KYC -> Fraud
- everything else (DISCHARGE_SUMMARY, PRESCRIPTION, POLICY_AMENDMENT,
  UNKNOWN) -> straight to Fraud; none of the three specialists apply to
  those doc types as built (PROJECT_PLAN SS3's own example: "a PRESCRIPTION
  doesn't need the Claims agent").

Human-in-the-loop (PROJECT_PLAN.md SS6): the Orchestrator's own deterministic
`requires_human_review` rule (every ESCALATE, and any REJECT with an
elevated fraud score) decides whether the graph pauses. This is not another
LLM call — the same auditable-rule philosophy as the decision itself. A
paused case is resumed via `graph.invoke(Command(resume=review_payload),
config=...)`; the reviewer's outcome either confirms the computed decision
or overrides it, and the override is recorded on `CaseState.human_review`
for the audit trail, never silently discarded. Fast-path APPROVE never
pauses — same principle as the reference notebook's HITL middleware only
gating destructive actions, not read-only ones.

Every graph invocation now requires a `thread_id` in `config` (LangGraph's
checkpointer keys state by thread) — `case_id` is used as the thread id,
since it's already the pipeline's unique-per-submission identifier."""

from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from backend.agents.claims import extract_claim
from backend.agents.classifier import classify_document
from backend.agents.fraud_detection import assess_fraud_risk
from backend.agents.kyc import verify_kyc
from backend.agents.orchestrator import decide, requires_human_review
from backend.agents.policy_rag import determine_coverage
from backend.agents.vision_utils import IMAGE_PROCESSING_ERROR_FLAG, ImageProcessingError
from backend.config import Settings, get_settings
from backend.graph.checkpointer import default_checkpointer
from backend.models import (
    CaseState,
    CaseStatus,
    ClaimsOutput,
    ClassifierOutput,
    Decision,
    DocType,
    ExtractedClaimFields,
    HumanReviewResult,
    KYCOutput,
    ReviewOutcome,
)
from backend.security.prompt_injection import scan_case_text_fields


def _plan_from_policy_number(policy_number: str | None) -> str:
    """Maps this dataset's policy_number convention (e.g. "MED-GLD-...")
    to the Policy RAG Agent's plan identifier. The generated dataset only
    contains Gold-plan patients (verified against dataset/metadata.json),
    so "gold" is also the safe fallback for a missing/unrecognized number
    rather than failing the whole case."""
    if policy_number and "SLV" in policy_number:
        return "silver"
    return "gold"


def _route_after_classification(state: CaseState) -> Literal["claims", "kyc", "fraud"]:
    doc_type = state.classifier_result.doc_type if state.classifier_result else DocType.UNKNOWN
    if doc_type == DocType.CLAIM_FORM:
        return "claims"
    if doc_type == DocType.ID_DOCUMENT:
        return "kyc"
    return "fraud"


def _route_after_claims(state: CaseState) -> Literal["policy", "fraud"]:
    result = state.claims_result
    if result is not None and result.schema_valid and result.extracted_fields.cpt_codes:
        return "policy"
    return "fraud"


def _route_after_orchestrator(state: CaseState) -> Literal["human_review", "end"]:
    return "human_review" if state.status == CaseStatus.AWAITING_REVIEW else "end"


def build_case_graph(
    settings: Settings | None = None, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    settings = settings or get_settings()
    checkpointer = checkpointer or default_checkpointer(settings)

    def classify_node(state: CaseState) -> dict:
        try:
            result = classify_document(state.file_path, settings=settings)
        except ImageProcessingError as exc:
            # A document the Classifier can't even read is exactly the
            # "needs a human" case DocType.UNKNOWN already models — this
            # converges with the UNKNOWN-doc escalation rule in
            # orchestrator.py instead of needing its own special case.
            result = ClassifierOutput(doc_type=DocType.UNKNOWN, confidence=0.0, routing_tags=[IMAGE_PROCESSING_ERROR_FLAG])
            return {
                "classifier_result": result,
                "status": CaseStatus.CLASSIFIED,
                "errors": [*state.errors, f"classifier: {exc}"],
            }
        return {"classifier_result": result, "status": CaseStatus.CLASSIFIED}

    def kyc_node(state: CaseState) -> dict:
        try:
            result = verify_kyc(state.file_path, settings=settings)
        except ImageProcessingError as exc:
            result = KYCOutput(kyc_passed=False, flags=[IMAGE_PROCESSING_ERROR_FLAG], confidence=0.0)
            return {"kyc_result": result, "errors": [*state.errors, f"kyc: {exc}"]}
        return {"kyc_result": result}

    def claims_node(state: CaseState) -> dict:
        try:
            result = extract_claim(state.file_path, settings=settings)
        except ImageProcessingError as exc:
            result = ClaimsOutput(
                extracted_fields=ExtractedClaimFields(),
                schema_valid=False,
                validation_errors=[IMAGE_PROCESSING_ERROR_FLAG],
                confidence=0.0,
            )
            return {"claims_result": result, "errors": [*state.errors, f"claims: {exc}"]}
        return {"claims_result": result}

    def policy_node(state: CaseState) -> dict:
        assert state.claims_result is not None  # only reachable via _route_after_claims
        fields = state.claims_result.extracted_fields
        plan = _plan_from_policy_number(state.policy_number)
        result = determine_coverage(plan, fields.cpt_codes, fields.icd10_codes, settings=settings)
        return {"policy_result": result}

    def fraud_node(state: CaseState) -> dict:
        fields = state.claims_result.extracted_fields if state.claims_result else None
        result = assess_fraud_risk(
            patient_id=state.patient_id or state.case_id,
            claim_amount=fields.claim_amount if fields else None,
            cpt_codes=fields.cpt_codes if fields else None,
            service_date=fields.service_date if fields else None,
            provider_npi=fields.provider_npi if fields else None,
            settings=settings,
        )
        return {"fraud_result": result, "status": CaseStatus.FRAUD_CHECK}

    def aggregate_node(state: CaseState) -> dict:
        return {"status": CaseStatus.AGGREGATED}

    def orchestrator_node(state: CaseState) -> dict:
        result = decide(state, settings=settings)
        fraud_score = state.fraud_result.fraud_score if state.fraud_result else 0.0
        needs_review = requires_human_review(result.decision, fraud_score, settings)
        status = CaseStatus.AWAITING_REVIEW if needs_review else CaseStatus.DECIDED

        # PROJECT_PLAN.md SS7 category 1, output-side layer: the narrative
        # is the free text most likely to reach a human reviewer or the
        # future UI verbatim — flag, don't block (see
        # backend/security/prompt_injection.py's docstring for why).
        injection_flags = scan_case_text_fields(result.justification, *result.agent_summaries.values())
        errors = list(state.errors)
        errors.extend(injection_flags)

        return {"decision": result, "status": status, "errors": errors}

    def human_review_node(state: CaseState) -> dict:
        assert state.decision is not None  # only reachable after orchestrator_node
        review_request = {
            "case_id": state.case_id,
            "decision": state.decision.decision.value,
            "confidence": state.decision.confidence,
            "justification": state.decision.justification,
            "fraud_score": state.fraud_result.fraud_score if state.fraud_result else None,
        }
        # Pauses here until resumed with `Command(resume=<review payload>)`;
        # `response` is exactly the dict passed to `resume`.
        response = interrupt(review_request)

        outcome = ReviewOutcome(response["outcome"])
        overridden_raw = response.get("overridden_decision")
        review = HumanReviewResult(
            outcome=outcome,
            overridden_decision=Decision(overridden_raw) if overridden_raw else None,
            reviewer_notes=response.get("notes", ""),
        )

        final_decision = state.decision
        if outcome == ReviewOutcome.OVERRIDDEN and review.overridden_decision is not None:
            final_decision = state.decision.model_copy(update={"decision": review.overridden_decision})

        return {"human_review": review, "decision": final_decision, "status": CaseStatus.DECIDED}

    graph = StateGraph(CaseState)
    graph.add_node("classify", classify_node)
    graph.add_node("kyc", kyc_node)
    graph.add_node("claims", claims_node)
    graph.add_node("policy", policy_node)
    graph.add_node("fraud", fraud_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("human_review", human_review_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify", _route_after_classification, {"claims": "claims", "kyc": "kyc", "fraud": "fraud"}
    )
    graph.add_conditional_edges("claims", _route_after_claims, {"policy": "policy", "fraud": "fraud"})
    graph.add_edge("policy", "fraud")
    graph.add_edge("kyc", "fraud")
    graph.add_edge("fraud", "aggregate")
    graph.add_edge("aggregate", "orchestrator")
    graph.add_conditional_edges("orchestrator", _route_after_orchestrator, {"human_review": "human_review", "end": END})
    graph.add_edge("human_review", END)

    return graph.compile(checkpointer=checkpointer)
