"""API request/response schemas. Case *detail* responses reuse `CaseState`
directly (backend/models) rather than duplicating its structure — it's
already the well-typed, single source of truth for a case's data; only
genuinely new shapes (upload response, list item, review request) get
their own schema here."""

from typing import Any

from pydantic import BaseModel, Field

from backend.models import CaseState, Decision, ReviewOutcome


class UploadResponse(BaseModel):
    case_id: str
    status: str


class CaseListItem(BaseModel):
    case_id: str
    status: str | None
    updated_at: str
    decision: str | None


class CaseDetail(BaseModel):
    case: CaseState
    # The raw interrupt() payload (see backend/graph/pipeline.py's
    # human_review_node) if this case is currently AWAITING_REVIEW — not
    # part of CaseState itself, since it's LangGraph's own pause signal,
    # not domain data the pipeline computed.
    pending_review: dict[str, Any] | None = None


class ReviewRequest(BaseModel):
    outcome: ReviewOutcome
    overridden_decision: Decision | None = None
    notes: str = Field(default="")
