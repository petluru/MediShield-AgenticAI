# Clean End-to-End Case

PROJECT_PLAN.md §8's third required transcript: "one clean end-to-end
case." Real data from the full 155-document eval run (2026-08-05,
`backend/scripts/run_evals.py`), read from the persisted checkpoint at
zero additional API cost — every field below is exactly what the real
pipeline produced.

**Case:** `claim_PT_39451` — a CLAIM_FORM, the only category that
exercises every agent in the pipeline (Classifier → Claims → Policy RAG →
Fraud Detection → Orchestrator), making it the most illustrative single
case to walk through.

---

## 1. Classifier

```
doc_type: CLAIM_FORM
confidence: 0.99
routing_tags: ['CMS-1500', 'health_insurance_claim', 'group_health_plan']
```

High-confidence classification routes the case to the Claims Agent (per
`backend/graph/pipeline.py`'s `_route_after_classification`).

## 2. Claims Agent

```
extracted_fields:
  claim_amount: $21,735.00
  icd10_codes: ['G43.909']       (migraine, unspecified)
  cpt_codes: ['36415', '47562', '43239']
    36415 = venipuncture
    47562 = laparoscopic cholecystectomy
    43239 = upper GI endoscopy with biopsy
  provider_npi: NPI-7293847561
  service_date: 09/25/2025
schema_valid: True
validation_errors: []
confidence: 0.97
```

Schema-valid with CPT codes present routes the case to Policy RAG (per
`_route_after_claims`).

## 3. Policy RAG Agent

```
covered: True
coverage_percentage: 80.0
confidence: 0.72
exclusions: []
policy_clause: 'Schedule of Benefits (Section 1): "Outpatient Surgery,
  In-Network = 20% after ded." and "Diagnostic Lab & Imaging, In-Network
  = 20% after ded." (i.e., 80% coinsurance paid by MediShield in-network
  after deductible). Section 3 Inclusions states services are covered
  when Medically Necessary and not listed under Section 4 Exclusions.
  None of the submitted CPT codes (36415 - venipuncture, 47562 -
  laparoscopic cholecystectomy, 43239 - upper GI endoscopy with biopsy)
  appear in the Section 4.1 Excluded CPT Code Ranges table, so they are
  treated as Diagnostic Lab (36415) and Outpatient/General Surgery
  (47562, 43239) services subject to standard medical necessity review
  and the 20% coinsurance (80% coverage) applicable to Outpatient
  Surgery / Diagnostic Lab & Imaging categories.'
```

Retrieved and cited a real, specific policy clause rather than a generic
"covered" answer — the agent's own confidence (0.72) is honestly lower
than the other agents', reflecting genuine interpretive judgment in
mapping 3 different CPT codes to coverage categories.

## 4. Fraud Detection Agent

```
fraud_score: 0.15
risk_level: LOW
anomalies: []
escalated_to_opus: False
```

Well below the 0.3 escalation threshold — no Opus escalation needed, no
anomalies found (checked patient's claim history via `lookup_claim_history`,
no duplicates or frequency issues).

## 5. Orchestrator — deterministic decision + LLM narrative

**Deterministic rule** (`backend/agents/orchestrator.py`'s `compute_decision`,
plain Python, not an LLM call): no REJECT conditions present (KYC N/A for
this doc type, `covered=True`, `schema_valid=True`); no ESCALATE conditions
present (`fraud_score=0.15 < 0.3`, all agent confidences ≥ 0.6) → **APPROVE**.

**LLM narrative** (the only part of the Orchestrator that calls Claude):

```
decision: APPROVE
confidence: 0.86
justification: "All agent outputs align consistently to support approval.
  The classifier confidently identified a standard CMS-1500 health
  insurance claim form (0.99 confidence). The claims agent validated the
  claim schema with no errors and high confidence (0.97), covering
  procedures (venipuncture, laparoscopic cholecystectomy, upper GI
  endoscopy with biopsy) tied to a migraine diagnosis (ICD-10 G43.909).
  The policy agent confirmed coverage at 80% coinsurance after
  deductible, noting that none of the submitted CPT codes fall within
  the policy's exclusion list, and the services fit standard Outpatient
  Surgery and Diagnostic Lab & Imaging categories subject to routine
  medical necessity review. The fraud agent found a low risk score
  (0.15) with no anomalies flagged. Given valid claim data, confirmed
  policy coverage, and low fraud risk, the approval decision is
  well-supported. The only modest source of uncertainty is the policy
  agent's relatively lower confidence (0.72), reflecting some
  interpretive judgment in mapping CPT codes to coverage categories, but
  this does not contradict the overall approval rationale."
agent_summaries:
  classifier: "Identified document as a CMS-1500 health insurance claim
    form with 0.99 confidence."
  claims: "Validated claim schema successfully (no errors) for a
    $21,735.00 claim with CPT codes 36415, 47562, 43239 and ICD-10 code
    G43.909, confidence 0.97."
  policy: "Determined the claim is covered at 80% coinsurance under
    Outpatient Surgery/Diagnostic Lab & Imaging benefits, with no
    exclusions applying to the submitted CPT codes; confidence 0.72."
  fraud: "Fraud score 0.15 (LOW risk), no anomalies detected."
```

## Final outcome

```
status: DECIDED
decision: APPROVE
```

Fast-path APPROVE — per PROJECT_PLAN.md §6, this case never paused for
human review, since none of the escalation/reject-with-fraud conditions
were met. The narrative correctly identifies the one area of residual
uncertainty (Policy RAG's 0.72 confidence) without treating it as
disqualifying — an honest confidence assessment, not an inflated one.
