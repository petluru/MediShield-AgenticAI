"""Prompt Injection — output-side guard (PROJECT_PLAN.md SS7, category 1).

Every agent's system prompt already carries the input-side mitigation
("Rules that CANNOT be overridden...", untrusted-content delimiters like
"### DOCUMENT IMAGE (untrusted...) ###") — that's the first line of
defense, already live in every agent built so far. This module is the
second line: a keyword/regex scan of what the model actually generated,
looking for evidence it got hijacked (echoed injected instructions, tried
to claim new authority, tried to reveal its own system prompt).

Deliberately a *flag*, not a silent block or auto-replace: KYC's tamper
detector already taught this codebase that an aggressive false-positive-prone
guard is worse than a missed subtle case (see backend/agents/kyc.py's
docstring) — blocking a legitimate discharge summary that happens to
mention "the patient was told to ignore prior medication instructions"
would be a real, damaging false positive. Flags get recorded on
CaseState.errors for human review, never silently discarded and never
silently overriding a real decision."""

import re

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(the\s+|all\s+)?(above|previous|prior)", re.IGNORECASE),
    re.compile(r"new\s+instructions\s*:", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an|in)\b", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+)?you\s+(are|were)\s+(a|an)\b", re.IGNORECASE),
    re.compile(r"\bsystem\s+override\b", re.IGNORECASE),
]


def scan_for_injection_artifacts(text: str) -> list[str]:
    """Return a description per suspicious pattern found in `text` (empty
    list if clean). Intended for LLM-generated free text that ends up
    visible to a human reviewer or another agent (Orchestrator's
    justification/agent_summaries, Fraud's anomalies, Policy's
    policy_clause/exclusions) — not for the untrusted source documents
    themselves, which are handled by the input-side prompt rules."""
    if not text:
        return []
    return [f"possible injection artifact in output: matched {p.pattern!r}" for p in _INJECTION_PATTERNS if p.search(text)]


def scan_case_text_fields(*texts: str | None) -> list[str]:
    """Scan several output fields at once (e.g. an Orchestrator narrative's
    justification plus each agent_summaries value) and return the combined
    flag list, deduped, preserving order."""
    flags: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for flag in scan_for_injection_artifacts(text or ""):
            if flag not in seen:
                seen.add(flag)
                flags.append(flag)
    return flags
