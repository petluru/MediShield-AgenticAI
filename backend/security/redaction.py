"""Sensitive Information Disclosure — redact before the prompt, not after
(PROJECT_PLAN.md SS7, category 2; reference/notebook_patterns.md SS2b).

Applied at the retrieval -> prompt-assembly boundary (backend/rag/retrieval.py,
right before a retrieved chunk is joined into what the Policy RAG Agent
sees), not at vector-store ingestion time — chunks are still embedded and
stored as-is; a chunk that's sensitive for one query might be fine to
surface differently elsewhere, so redaction is a per-use decision, not a
one-time ingestion-time mutation.

The current policy PDFs (Gold/Silver plan benefit schedules) don't contain
real patient PII/PHI — this is defense-in-depth for whatever gets chunked
into this pipeline next (e.g. if discharge summaries or claim forms ever
become retrievable context), not a fix for an observed leak in this
dataset."""

import re

# SSN (###-##-####), a "DOB:"/"born on"-prefixed date (PHI in this domain
# specifically, not any date), and a 16-digit card-number-shaped sequence.
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_DOB_PATTERN = re.compile(r"\b(DOB|date of birth|born on)\s*[:\-]?\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b", re.IGNORECASE)
_CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){13,16}\b")

_REDACTED_MARKER = "[REDACTED: sensitive record — not available to this agent]"


def is_sensitive(text: str) -> bool:
    if not text:
        return False
    return "CONFIDENTIAL" in text or bool(
        _SSN_PATTERN.search(text) or _DOB_PATTERN.search(text) or _CARD_PATTERN.search(text)
    )


def redact_if_sensitive(text: str) -> str:
    """Replace `text` wholesale with the redaction marker if it trips
    `is_sensitive` — a whole-chunk swap, not a partial in-place mask, so a
    reviewer never sees a half-redacted record that still leaks context
    around the masked span."""
    return _REDACTED_MARKER if is_sensitive(text) else text
