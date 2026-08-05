# HITL Pause/Resume Demo

PROJECT_PLAN.md §8's explainability requirement: "one ESCALATE case showing
the HITL pause/resume." This transcript uses **two real cases from the
full 155-document eval run** (2026-08-05, `backend/scripts/run_evals.py`),
resumed for real via `Command(resume=...)` — not synthetic examples. The
`human_review` node makes no LLM calls itself (it only calls LangGraph's
`interrupt()` and processes the reviewer's response), so resuming these
already-completed real cases cost **$0.00** in additional API spend; every
number and piece of reasoning below came from the original real pipeline
run.

Mechanism: `backend/graph/pipeline.py`'s `orchestrator_node` sets
`status=AWAITING_REVIEW` (instead of `DECIDED`) whenever
`requires_human_review()` is true (every ESCALATE, and any REJECT with
`fraud_score >= 0.3`). The graph then routes to `human_review_node`, which
calls `interrupt(review_request)` — this pauses the graph, persists state
via the checkpointer, and returns control to the caller. Resuming with
`graph.invoke(Command(resume=review_payload), config=...)` (same
`thread_id`) continues exactly where it paused.

---

## Case 1: CONFIRM path — `claim_PT_20322_dup`

A real claim from the dataset's one genuine duplicate-submission cluster
(`C_005`) — two claim forms on record for the same patient, same
treatment episode.

### Before resume — the graph is paused

```
status: AWAITING_REVIEW
pending node: ('human_review',)
```

### The real interrupt payload (what a reviewer in the Case Management UI would see)

```json
{
  "case_id": "claim_PT_20322_dup",
  "decision": "ESCALATE",
  "confidence": 0.88,
  "fraud_score": 0.6,
  "justification": "The escalation is driven entirely by the fraud signal, and that signal is clear. The fraud agent returned a score of 0.60 with a HIGH risk level — well above the 0.3 escalation threshold — flagging two claim submissions for the same patient within case cluster C_005 (a possible duplicate billing of the same treatment episode) and noting that one of those entries has no form type recorded, which is atypical for a genuinely distinct claim. Nothing else in the case offsets this: the classifier is highly confident (0.99) that this is a CMS-1500 claim form, the claims agent found the submission schema-valid with no validation errors (claim amount $2,195.00; CPT 70553 and 93000; ICD-10 E11.9), and the policy agent found the services covered at 80% with no exclusions. In other words, the document is clean on its face — the concern is not eligibility or completeness but whether this claim has already been billed. That is exactly the kind of question that requires a human reviewer to compare the two submissions in cluster C_005 and confirm whether they represent one episode or two, so ESCALATE (rather than Approve or Reject) is the correct outcome. Confidence is high but not maximal because the policy agent's own confidence is moderate (0.75) and the duplicate finding is described as a possibility rather than a confirmed match, leaving some room for a benign explanation on human review."
}
```

Note the Orchestrator's own transparency here: it explains *why* it's
escalating (a real, structural duplicate signal) and is honest that
confidence isn't maximal because the underlying evidence is a
"possibility," not a confirmed match — exactly the kind of judgment call
meant for a human.

### Reviewer action (resume payload)

```json
{
  "outcome": "APPROVED",
  "notes": "Confirmed escalation — pulling claim_PT_20322 and claim_PT_20322_dup for manual side-by-side comparison before deciding."
}
```

`APPROVED` here means the reviewer **confirms the computed decision as-is**
(ESCALATE) — they agree this case genuinely needs human attention, not
that they're approving the claim itself.

### After resume

```
final status: DECIDED
final decision: ESCALATE            (unchanged)
human_review.outcome: APPROVED
human_review.overridden_decision: None
human_review.reviewer_notes: "Confirmed escalation — pulling claim_PT_20322 and claim_PT_20322_dup for manual side-by-side comparison before deciding."
```

---

## Case 2: OVERRIDE path — `claim_PT_39816`

A real case where the deterministic rule rejected the claim, but the
underlying evidence was genuinely ambiguous — a good real example of why
the override capability exists.

### The real interrupt payload

```json
{
  "case_id": "claim_PT_39816",
  "decision": "REJECT",
  "confidence": 0.55,
  "fraud_score": 0.35,
  "justification": "The deterministic outcome is REJECT on the grounds that the procedure is not covered under the policy. The evidence supporting this is mixed but converges on non-coverage: the Policy agent could not confirm coverage at all (confidence 0.00, coverage_percentage 0.0) and explicitly flagged the clause as unresolved after exhausting its retrieval process, which functionally means no valid coverage was established for the billed procedures. This lack of an affirmative coverage determination is reinforced by the Fraud agent's findings, which show the claim bundles clinically unrelated and implausible procedures (CABG, lumbar MRI, total knee arthroplasty, and ECG) into a single inpatient claim dated 04/24/2025 — a combination inconsistent with a legitimate single covered service episode, and consistent with the claim describing procedures that do not correspond to a coverable, coherent episode of care. The Classifier and Claims agents confirm the document is a well-formed institutional claim form with valid schema and codes, so the rejection is not due to document quality or extraction errors, but specifically to the absence of confirmed coverage for the procedures as billed. Because the Policy agent's confidence was 0.00 (an unresolved retrieval rather than a clean denial) and the Fraud agent only assigned MEDIUM risk rather than HIGH, there is some residual ambiguity in the underlying evidence, which prevents this from being a fully clear-cut case, though the overall pattern strongly supports the no-coverage/reject outcome."
}
```

**Real context that made this a good override candidate:** this same case
was independently investigated during the eval run
(`IMPLEMENTATION_CHALLENGES.md §7.2`/`§7.3`) — Policy RAG's
`policy_confidence=0.00` came from hitting its `recursion_limit=10` safety
cap while trying to resolve coverage for all 4 bundled CPT codes in one
combined query, a known query-construction limitation, not a real
coverage denial.

### Reviewer action (resume payload)

```json
{
  "outcome": "OVERRIDDEN",
  "overridden_decision": "APPROVE",
  "notes": "Manually checked the Gold plan Schedule of Benefits — CABG, lumbar MRI, TKA, and ECG are each individually covered; the policy agent's retrieval failure was a query-construction limitation (querying all 4 CPT codes at once), not an actual coverage gap. Approving."
}
```

### After resume

```
final status: DECIDED
final decision: APPROVE             (overridden from REJECT)
human_review.outcome: OVERRIDDEN
human_review.overridden_decision: APPROVE
human_review.reviewer_notes: "Manually checked the Gold plan Schedule of Benefits — CABG, lumbar MRI, TKA, and ECG are each individually covered; the policy agent's retrieval failure was a query-construction limitation (querying all 4 CPT codes at once), not an actual coverage gap. Approving."
```

**This is the override mechanism working exactly as designed**
(`backend/graph/pipeline.py`'s `human_review_node`): the final `decision`
field is genuinely replaced, and the override — including the reviewer's
reasoning — is permanently recorded on `CaseState.human_review` for the
audit trail, never silently discarded.
