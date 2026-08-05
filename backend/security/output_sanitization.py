"""Improper Output Handling — HTML-escape at the render boundary
(PROJECT_PLAN.md SS7, category 4; reference/notebook_patterns.md SS2d).

Stdlib only. Not wired anywhere yet — there is no UI (task #15) or API
response layer (task #14) to wire it into. This exists now so that work,
when it happens, has a ready, already-tested function to import rather than
each call site inventing its own escaping — and so this category isn't
silently forgotten once the UI actually renders agent-generated text
(justification, policy_clause, anomalies, reviewer_notes) as HTML.

Escape immediately before HTML interpolation, not earlier in the pipeline —
escaping at generation time would corrupt the text for every non-HTML
consumer (logs, the token-usage report, another agent's prompt)."""

import html


def sanitize_for_html(text: str) -> str:
    return html.escape(text or "")
