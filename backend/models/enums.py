from enum import StrEnum


class DocType(StrEnum):
    CLAIM_FORM = "CLAIM_FORM"
    ID_DOCUMENT = "ID_DOCUMENT"
    DISCHARGE_SUMMARY = "DISCHARGE_SUMMARY"
    PRESCRIPTION = "PRESCRIPTION"
    POLICY_AMENDMENT = "POLICY_AMENDMENT"
    UNKNOWN = "UNKNOWN"


class CaseStatus(StrEnum):
    RECEIVED = "RECEIVED"
    CLASSIFIED = "CLASSIFIED"
    PROCESSING = "PROCESSING"
    FRAUD_CHECK = "FRAUD_CHECK"
    AGGREGATED = "AGGREGATED"
    AWAITING_REVIEW = "AWAITING_REVIEW"
    DECIDED = "DECIDED"


class Decision(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ReviewOutcome(StrEnum):
    """A human reviewer's action on a paused case (Component 9's human
    review queue with override capability)."""

    APPROVED = "APPROVED"  # reviewer confirms the computed decision as-is
    OVERRIDDEN = "OVERRIDDEN"  # reviewer replaces it with a different decision
