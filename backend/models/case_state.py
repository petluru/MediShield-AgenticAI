"""Shared state that flows through the LangGraph pipeline. One CaseState
instance = one uploaded document (assignment Component 1: "unique case_id
per submission"). Each specialist agent writes to its own field, so parallel
branches (KYC + Claims + Policy) never conflict on the same key."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from backend.models.enums import CaseStatus, Decision, DocType, ReviewOutcome, RiskLevel


class ClassifierOutput(BaseModel):
    """Component 2: Classifier Agent."""

    doc_type: DocType
    confidence: float = Field(ge=0.0, le=1.0)
    routing_tags: list[str] = Field(default_factory=list)


class ExtractedClaimFields(BaseModel):
    """The specific fields the assignment requires the Claims Agent to pull
    (Component 5 / Extraction Completeness eval criterion)."""

    claim_amount: float | None = None
    icd10_codes: list[str] = Field(default_factory=list)
    cpt_codes: list[str] = Field(default_factory=list)
    provider_npi: str | None = None
    service_date: str | None = None


class KYCOutput(BaseModel):
    """Component 4: KYC Agent."""

    kyc_passed: bool
    flags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ClaimsOutput(BaseModel):
    """Component 5: Claims Agent."""

    extracted_fields: ExtractedClaimFields
    schema_valid: bool
    validation_errors: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class PolicyOutput(BaseModel):
    """Component 6: Policy Agent (RAG)."""

    covered: bool
    coverage_percentage: float = Field(ge=0.0, le=100.0)
    policy_clause: str
    exclusions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class FraudOutput(BaseModel):
    """Component 7: Fraud Detection Agent."""

    fraud_score: float = Field(ge=0.0, le=1.0)
    anomalies: list[str] = Field(default_factory=list)
    risk_level: RiskLevel
    escalated_to_opus: bool = False


class OrchestratorDecision(BaseModel):
    """Component 8: Orchestrator Agent."""

    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str
    agent_summaries: dict[str, str] = Field(default_factory=dict)
    escalated_to_opus: bool = False


class HumanReviewResult(BaseModel):
    """Component 9: the Case Management UI's human review queue records this
    when a reviewer resolves a paused (AWAITING_REVIEW) case — the audit
    trail for who approved/overrode what and why."""

    outcome: ReviewOutcome
    overridden_decision: Decision | None = None
    reviewer_notes: str = ""
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CaseState(BaseModel):
    """The LangGraph StateGraph's state schema. RECEIVED -> CLASSIFIED ->
    [PARALLEL: KYC + CLAIMS + POLICY] -> FRAUD_CHECK -> AGGREGATED -> DECIDED,
    with an optional AWAITING_REVIEW pause for the HITL gate (PROJECT_PLAN
    SS6) between AGGREGATED and DECIDED."""

    case_id: str
    file_path: str
    content_type: str
    status: CaseStatus = CaseStatus.RECEIVED

    patient_id: str | None = None
    policy_number: str | None = None

    classifier_result: ClassifierOutput | None = None
    kyc_result: KYCOutput | None = None
    claims_result: ClaimsOutput | None = None
    policy_result: PolicyOutput | None = None
    fraud_result: FraudOutput | None = None
    decision: OrchestratorDecision | None = None
    human_review: HumanReviewResult | None = None

    errors: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
