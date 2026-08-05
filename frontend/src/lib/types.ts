// Mirrors backend/models/case_state.py and backend/models/enums.py exactly
// — kept in the frontend rather than generated, since the backend has no
// OpenAPI-schema-export step in this project. If a backend field changes,
// this file needs a matching edit (same tradeoff already accepted
// elsewhere in this codebase for backend/evals' hand-written scoring
// schemas vs. the real Pydantic models).

export type DocType =
  | "CLAIM_FORM"
  | "ID_DOCUMENT"
  | "DISCHARGE_SUMMARY"
  | "PRESCRIPTION"
  | "POLICY_AMENDMENT"
  | "UNKNOWN";

export type CaseStatusValue =
  | "RECEIVED"
  | "CLASSIFIED"
  | "PROCESSING"
  | "FRAUD_CHECK"
  | "AGGREGATED"
  | "AWAITING_REVIEW"
  | "DECIDED";

export type Decision = "APPROVE" | "REJECT" | "ESCALATE";
export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";
export type ReviewOutcome = "APPROVED" | "OVERRIDDEN";

export interface ClassifierOutput {
  doc_type: DocType;
  confidence: number;
  routing_tags: string[];
}

export interface ExtractedClaimFields {
  claim_amount: number | null;
  icd10_codes: string[];
  cpt_codes: string[];
  provider_npi: string | null;
  service_date: string | null;
}

export interface KYCOutput {
  kyc_passed: boolean;
  flags: string[];
  confidence: number;
}

export interface ClaimsOutput {
  extracted_fields: ExtractedClaimFields;
  schema_valid: boolean;
  validation_errors: string[];
  confidence: number;
}

export interface PolicyOutput {
  covered: boolean;
  coverage_percentage: number;
  policy_clause: string;
  exclusions: string[];
  confidence: number;
}

export interface FraudOutput {
  fraud_score: number;
  anomalies: string[];
  risk_level: RiskLevel;
  escalated_to_opus: boolean;
}

export interface OrchestratorDecision {
  decision: Decision;
  confidence: number;
  justification: string;
  agent_summaries: Record<string, string>;
  escalated_to_opus: boolean;
}

export interface HumanReviewResult {
  outcome: ReviewOutcome;
  overridden_decision: Decision | null;
  reviewer_notes: string;
  reviewed_at: string;
}

export interface CaseState {
  case_id: string;
  file_path: string;
  content_type: string;
  status: CaseStatusValue;
  patient_id: string | null;
  policy_number: string | null;
  classifier_result: ClassifierOutput | null;
  kyc_result: KYCOutput | null;
  claims_result: ClaimsOutput | null;
  policy_result: PolicyOutput | null;
  fraud_result: FraudOutput | null;
  decision: OrchestratorDecision | null;
  human_review: HumanReviewResult | null;
  errors: string[];
  created_at: string;
  updated_at: string;
}

export interface PendingReview {
  case_id: string;
  decision: Decision;
  confidence: number;
  fraud_score: number | null;
  justification: string;
}

export interface CaseDetail {
  case: CaseState;
  pending_review: PendingReview | null;
}

export interface CaseListItem {
  case_id: string;
  status: CaseStatusValue | null;
  updated_at: string;
  decision: Decision | null;
}

export interface UploadResponse {
  case_id: string;
  status: string;
}

export interface ReviewRequest {
  outcome: ReviewOutcome;
  overridden_decision?: Decision;
  notes?: string;
}
