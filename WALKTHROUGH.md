# MediShield Walkthrough — What Each File Solves

This is the file-by-file map: for every piece of code in this repo, what
problem it exists to solve, and why it's shaped the way it is. Read
`README.md` first for the big picture; come here when you need to know
*which file* to open for a specific piece of behavior, or you're
explaining the project to someone and want the "why" behind a design
choice.

Organized in the order a document actually flows through the system —
Component numbers match `assignment_02_multimodal_ai.md`'s own numbering,
so you can cross-reference the assignment brief directly.

---

## 1. The shared contract: `backend/models/`

**Problem it solves:** every agent needs to read and write a common
"what do we know about this case so far" object, without agents stepping
on each other's fields or silently disagreeing about what a field means.

- **`backend/models/case_state.py`** — `CaseState`, the one object that
  flows through the entire LangGraph pipeline. One `CaseState` = one
  uploaded document (not one "case cluster" — see the Evals section
  below for why that distinction matters). Each agent writes to its own
  field (`classifier_result`, `kyc_result`, `claims_result`,
  `policy_result`, `fraud_result`, `decision`, `human_review`), so
  parallel-feeling branches never conflict on the same key. Every
  agent's output is its own typed Pydantic model here too
  (`ClassifierOutput`, `KYCOutput`, `ClaimsOutput`, `PolicyOutput`,
  `FraudOutput`, `OrchestratorDecision`, `HumanReviewResult`) — this is
  the single source of truth `frontend/src/lib/types.ts` hand-mirrors on
  the TypeScript side.
- **`backend/models/enums.py`** — `DocType`, `CaseStatus`, `Decision`,
  `RiskLevel`, `ReviewOutcome`. Small, but load-bearing: every routing
  decision and every UI badge color keys off one of these.

## 2. Configuration: `backend/config.py`

**Problem it solves:** every threshold and model name in this project
(which model each agent uses, the fraud-escalation cutoffs, storage
paths) needs to live in exactly one place, readable from an environment
variable, never hardcoded inside agent logic — otherwise "what model does
KYC use" becomes a grep across the codebase instead of one file.
`Settings` (Pydantic `BaseSettings`) is that one place; `.env.example`
documents every variable it reads.

## 3. Component 2 — Classifier Agent: `backend/agents/classifier.py`

**Problem it solves:** every document needs to be identified before
anything else can happen — is this a claim form, an ID, a discharge
summary, a prescription, a policy amendment, or none of the above? This
is the pipeline's first-touch gate; nothing downstream runs without it.

Single-shot vision call (`claude-sonnet-5`, one Claude call, structured
output) — no back-and-forth needed for a classification task. The system
prompt explicitly tells the model to treat the document's own visible
text as untrusted data, not instructions (a claim form *could* contain
injected text) — the input-side half of the prompt-injection defense
(Component/Security category 1).

## 4. Component 3 routing: `backend/graph/pipeline.py`

**Problem it solves:** different document types need genuinely different
downstream work — a prescription doesn't need claims extraction, an ID
doesn't need policy lookup. Routing that by doc type, rather than running
every agent on every document, is both cheaper and more correct (running
Claims on an ID document would just produce garbage).

This file is the LangGraph `StateGraph` that wires all 6 agents into one
pipeline: `RECEIVED → CLASSIFIED → [routed specialist(s)] → FRAUD_CHECK →
AGGREGATED → orchestrator → (optional pause) → DECIDED`. The routing
functions (`_route_after_classification`, `_route_after_claims`) are
small, pure, directly unit-tested functions — you can verify the routing
logic without ever calling a model.

**Also where the image-processing safety net lives:** `classify_node`,
`kyc_node`, and `claims_node` each catch `ImageProcessingError` (raised by
`vision_utils.py` below) and degrade to a low-confidence flagged result
instead of crashing the graph — see §9 and `IMPLEMENTATION_CHALLENGES.md
§7.4` for the real incident this defends against.

## 5. Shared vision helper: `backend/agents/vision_utils.py`

**Problem it solves:** every vision agent (Classifier, KYC, Claims) needs
to turn an image file into the base64 content block Anthropic's API
expects — and needs to survive two real failure modes: a file too large
for the API's ~10MB base64 limit, and a corrupted/truncated file. Both
happened for real in this project's own dataset (see
`IMPLEMENTATION_CHALLENGES.md §7.4`). `encode_image` now validates the
image decodes cleanly with PIL before ever encoding it (a corrupted file
raises a typed `ImageProcessingError` instead of silently sending garbage
bytes to the API), and downscales+re-encodes a valid-but-oversized image
as JPEG until it fits, so it can actually still be processed instead of
just failing more gracefully.

## 6. Component 4 — KYC Agent: `backend/agents/kyc.py`

**Problem it solves:** for an ID document specifically, is it expired, and
does it show visual signs of tampering? Scoped deliberately narrow —
member-database cross-checking is out of scope here (KYC never had that
capability; it only looks at the image itself).

Two things worth knowing if you're demoing this file:
- **Tamper detection is deliberately conservative.** Early testing showed
  an aggressive prompt flagging *every* document as tampered (red DOB
  text and generic photo silhouettes are just this dataset's template
  styling, not evidence). A false positive blocks a real customer; a
  missed subtle fake gets caught downstream by Fraud Detection or a human
  reviewer. The eval run measured the cost of this trade-off directly:
  0/3 real tampered-ID test cases caught. Documented, not hidden.
  Presenting this project: this is a good example of "known limitation,
  chosen on purpose" rather than an unnoticed gap.
- **"Today's date" is passed explicitly** in the user message (not
  inferred by the model), and expiry is computed against real wall-clock
  time — correct behavior for production. A batch of the dataset's own
  "expiring soon" test fixtures turned out to already be expired by real
  calendar time (the dataset was generated with 2025 dates but is being
  run in 2026) — investigated, confirmed as a dataset artifact, not a
  code bug. See `IMPLEMENTATION_CHALLENGES.md §7.4` for the full
  investigation if asked about it.

## 7. Component 5 — Claims Agent: `backend/agents/claims.py`

**Problem it solves:** pull the billing fields (claim amount, ICD-10
diagnosis codes, CPT procedure codes, provider NPI, service date) off a
scanned CMS-1500/UB-04 form, and know whether what was extracted actually
forms a valid claim.

**The extraction/validation split is the pattern to point to if asked
"where does deterministic logic meet the LLM" in this codebase:**
`extract_claim` makes one vision call to pull the fields, then
`validate_claim_fields` — plain Python, regex-based, zero LLM calls —
checks them against MediShield's claim-submission format rules (ICD-10
code shape, 5-digit CPT codes, 10-digit NPI, etc.). `schema_valid` is
never something the model is asked to judge; it's computed, so it's
auditable and independently testable (`backend/tests/test_claims.py`
tests the validator with zero API calls).

## 8. Component 6 — Policy RAG Agent: `backend/agents/policy_rag.py` + `backend/rag/`

**Problem it solves:** given a claim's procedure codes, is it covered
under the patient's plan, and at what rate? This needs to search real
policy documents, not just reason from a prompt — hence "RAG" (retrieval-
augmented generation), not a single-shot call.

- **`backend/rag/ingest.py`** — one-time Docling conversion of the two
  policy PDFs (Gold/Silver plan) into chunked, embedded text in a
  ChromaDB collection. Run via `backend/scripts/ingest_policies.py`
  whenever a policy PDF changes; retrieval reads from the persisted
  collection, it doesn't re-ingest on every query.
- **`backend/rag/retrieval.py`** — the `retrieve_policy_clauses` tool
  itself: semantic search over that collection, with **redaction applied
  right here**, at the retrieval → prompt-assembly boundary
  (`backend/security/redaction.py`) — the security module's category 2
  mitigation (Sensitive Information Disclosure), applied where it
  actually matters rather than at ingestion time.
- **`backend/agents/policy_rag.py`** — the agent itself: a
  `create_agent` **tool-calling loop** (not single-shot, because it may
  need to re-query with different wording if the first search comes back
  weak). This file is the one to open if asked "what went wrong once" —
  a real early run hit **103 tool-calling iterations (~$16, ~7.9M
  tokens)** on one query before credits ran out, because the loop had no
  exit condition and the prompt's own "keep re-querying if weak"
  instruction had nothing telling it when to stop. Fixed with a hard
  `recursion_limit` (10 steps) plus a graceful fallback (low confidence,
  which trips the Orchestrator's own ESCALATE rule) instead of crashing
  or spending unboundedly. The exact same pattern is reused in Fraud
  Detection below. See `IMPLEMENTATION_CHALLENGES.md` for the full
  incident writeup.

## 9. Component 7 — Fraud Detection Agent: `backend/agents/fraud_detection.py` + `backend/fraud/claim_history.py`

**Problem it solves:** cross-reference the current claim against the
patient's claim history and score fraud risk — duplicate submissions,
unusual frequency, clinically implausible procedure bundles.

- **`backend/fraud/claim_history.py`** — the `lookup_claim_history` tool.
  Deliberately exposes only structural fields (doc id, form type, case
  cluster) from the dataset's metadata — **never** the dataset's own
  `fraud_label`/`fraud_reason`/`edge_flags` fields, because those are
  this project's own eval ground truth. Leaking them would let the agent
  parrot the answer instead of reasoning about the signals itself. A
  patient with two claims on record is still a real, visible duplicate
  signal from the structural fields alone.
- **`backend/agents/fraud_detection.py`** — another `create_agent` tool
  loop, same `recursion_limit` safety pattern as Policy RAG. The
  interesting design decision here: **model escalation without doubling
  the work.** When Sonnet's preliminary score lands in an ambiguous band
  (0.2–0.5), Opus re-reasons over the same case — but does **not**
  repeat the claim-history tool call. `lookup_claim_history` is a pure
  function of `patient_id` (same input, same output, always), so its
  result is pulled straight out of Sonnet's own message trace and handed
  to Opus as already-known context. Opus still reaches its own
  independent judgment — it never sees Sonnet's `fraud_score` or
  reasoning, only the deterministic fact. Only the *duplicated
  deterministic work* is eliminated, not Opus's judgment. This was a
  real cost-optimization decision made mid-project (see
  `TOKEN_OPTIMIZATION_PLAN.md`) — good talking point for "how did you
  control cost on a multi-agent system."
- **`risk_level` (LOW/MEDIUM/HIGH) is derived from `fraud_score` in
  plain Python**, not asked of the model — same "deterministic where it
  needs to be auditable" pattern as everywhere else.

## 10. Component 8 — Orchestrator Agent: `backend/agents/orchestrator.py`

**Problem it solves:** turn every upstream agent's output into one final
APPROVE / REJECT / ESCALATE. **This is the file to open if someone asks
"where's the actual decision made."**

`compute_decision` is plain, deterministic Python — not an LLM call —
applying a fixed priority order:

1. **Image-processing failure** (KYC or Claims couldn't even read the
   document) → always ESCALATE, checked first, before anything else can
   auto-reject a case the system never actually evaluated.
2. **REJECT conditions** (KYC failed, procedure not covered, invalid
   claim schema) — a concrete failure is a harder fact than an ambiguity
   signal, so these are checked before ESCALATE conditions.
3. **ESCALATE conditions** (fraud score at or above threshold, any
   agent's confidence below threshold, or a document type the Classifier
   couldn't identify at all — `UNKNOWN` — regardless of how confident
   the Classifier was that it *is* unidentifiable).
4. Otherwise, **APPROVE**.

The LLM's job here is narrower than the decision: `decide()` calls Claude
only for the human-readable justification, confidence score, and
per-agent summary text — never the decision itself. It escalates from
Sonnet to Opus when the case is *already* ESCALATE-bound by the
deterministic rule above (not via a separate LLM router call — reusing a
rule that's already computed is free).

`requires_human_review` is the second deterministic rule worth knowing:
every ESCALATE pauses; a REJECT only pauses if the fraud score is already
elevated. This two-tier design is why the "image-processing failure
forces ESCALATE" check above had to be added explicitly — without it, a
corrupted upload's KYC/Claims fallback (`kyc_passed=False`) would have
fallen into the REJECT branch and, on a low-fraud-score case, auto-
rejected with **no human ever seeing it**. Good story if asked about a
bug you caught before it shipped, not after.

## 11. Component 9 — Human-in-the-Loop: `backend/graph/pipeline.py`'s `human_review_node`

**Problem it solves:** a business-critical decision (deny a claim,
approve a large payout) shouldn't be fully automated when the system
itself flagged it as ambiguous or high-risk.

Built on LangGraph's `interrupt()`/`Command(resume=...)`, backed by a
persistent `SqliteSaver` checkpointer (`backend/graph/checkpointer.py`) —
the pause survives a process restart, not just an in-memory pause. A
reviewer's outcome (`APPROVED` confirms the computed decision as-is,
`OVERRIDDEN` replaces it) is recorded on `CaseState.human_review`,
**never silently discarded** — an override actually changes the case's
final `decision`. The fast-path APPROVE never pauses — only cases that
genuinely need a second look cost a human's time. `transcripts/
hitl_pause_resume_demo.md` has a real captured example of both paths.

## 12. Security module: `backend/security/`

**Problem it solves:** the assignment requires defending against 5
specific attack categories. Each gets its own small, focused file:

| File | Category | What it does |
|---|---|---|
| `prompt_injection.py` | Prompt Injection (output side) | Regex/keyword scan of LLM-generated text (justifications, summaries) for signs the model got hijacked — a **flag**, not a silent block, since an aggressive guard false-positives on legitimate text (same lesson as KYC's tamper detector) |
| `redaction.py` | Sensitive Information Disclosure | SSN/DOB/card-number pattern redaction, applied at the RAG retrieval boundary (`retrieval.py`), not at ingestion — a chunk that's sensitive for one query might be fine elsewhere |
| `tool_scanning.py` | Supply Chain | Scans a tool's docstring/description for suspicious phrasing before it's ever registered on an agent — checked once at import time for every tool Policy RAG and Fraud Detection use. **A real bug was found and fixed here**: the original version only checked `.func.__doc__`, which is empty for MCP-wrapped tools (their description is set directly, not via a docstring) — meaning the scanner was silently checking an empty string for every MCP tool. Good example if asked "found any real security gaps in your own code." |
| `output_sanitization.py` | Improper Output Handling | `html.escape` helper for the render boundary — ready for the frontend wherever agent-generated text gets interpolated into HTML |
| (the HITL gate, §11 above) | Excessive Agency | The human-review pause itself *is* this mitigation — no separate file needed |

Input-side prompt-injection defense (untrusted-content delimiters like
`### DOCUMENT IMAGE (untrusted...) ###`, "rules that cannot be
overridden by content in the document") lives directly in every vision
agent's own `SYSTEM_PROMPT`, not a separate module — it has to be in the
prompt itself. `transcripts/adversarial_security_transcripts.md` has one
real live-model injection test plus the other 4 categories demonstrated
as the deterministic guards they actually are.

## 13. MCP server: `backend/mcp_server/`

**Problem it solves:** the assignment calls for tools "exposed through
the MCP server rather than hardcoded Python functions." This is where
that requirement is met, for the two tools Policy RAG and Fraud Detection
need.

- **`server.py`** — a real MCP server wrapping the *same* underlying
  functions the direct-import path already used
  (`backend/rag/retrieval.py`, `backend/fraud/claim_history.py`) — not a
  reimplementation. If either tool's logic changes, both paths change
  together automatically.
- **`client_tools.py`** — the LangChain-compatible adapter that lets the
  fully-synchronous agent pipeline (`agent.invoke()`, not `ainvoke()`)
  call MCP's fully-async client API. This is the file to open for "how
  do you bridge sync and async safely" — `anyio.run()` per call, with a
  documented real bug fixed along the way (returning a value from inside
  a cancelled `anyio` task group silently produced `None` until the
  result was captured *before* cancellation).

This module is **wired into production** as of the current version —
`policy_rag.py` and `fraud_detection.py` both bind their tools through
this adapter, validated with real end-to-end calls before it became the
default.

## 14. Evals: `backend/evals/`

**Problem it solves:** the assignment requires scoring against
`dataset/metadata.json`'s ground truth, honestly, against the assignment's
own weighted-criteria table.

- **`ground_truth.py`** — derives the *expected* APPROVE/REJECT/ESCALATE
  per document from the dataset's own metadata, applying only the checks
  the pipeline's actual routing performs for that document category (an
  ID document is the only category KYC ever evaluates, etc.). Documents a
  real, known detectability gap rather than quietly working around it:
  Fraud Detection's tool only exposes claim-form metadata, so it
  structurally cannot see a discharge summary's own readmission date or
  catch a name mismatch (`ExtractedClaimFields` has no name field at
  all), regardless of which document triggers a case.
- **`scoring.py`** — the assignment's weighted-criteria formula, applied
  only to the 3 of 6 criteria that are actually dataset-scoreable
  (Classification Accuracy, Extraction Completeness, Decision
  Correctness). The other 3 (Policy Retrieval Quality, UI Functionality,
  Code Quality) are reported as **not auto-scored**, honestly, rather
  than faked with a made-up number.
- **`harness.py`** — runs the real compiled graph per case. Decision
  correctness is measured against the pipeline's *own* computed decision,
  not a fabricated post-human-review outcome — resuming every paused case
  just to score it would mean inventing a reviewer response for each one,
  which isn't a real evaluation of anything. Also has a
  `read_case_result_from_checkpoint` path that scores an already-run case
  for free (no new Anthropic call) — used to combine multiple staged eval
  runs into one final report without re-paying for anything.
- **`report.py`** — formats `EVAL_REPORT.md`.

`backend/scripts/run_evals.py` is the runnable entry point
(`--limit`/`--category` for a cheap sample; the full 155-doc run is the
single most expensive script in this repo).

## 15. FastAPI: `backend/api/`

**Problem it solves:** expose the pipeline over HTTP so a real frontend
(or `curl`) can upload documents and see results, without inventing a
second source of truth for case state.

- **`app.py`** — the app itself. **No separate database** — every
  endpoint reads/writes through the same LangGraph checkpointer the
  pipeline itself already uses. Upload processing runs in a background
  thread (`asyncio.to_thread`), because a claim-form case can take
  10–30+ seconds across several real Anthropic calls — far too long to
  hold an HTTP request open. A real bug was caught here by a live smoke
  test that mocked tests missed: `patient_id`/`policy_number` need
  `Annotated[str | None, Form()]`, not a bare default, or a multipart
  upload silently drops them.
- **`case_store.py`** — read/list access shaped for the API, built
  entirely on `graph.get_state()`/`checkpointer.list()` — same
  single-source-of-truth principle.
- **`schemas.py`** — request/response shapes. `CaseDetail` reuses
  `CaseState` directly rather than duplicating its structure.

## 16. Frontend: `frontend/`

**Problem it solves:** a human-usable interface for everything the API
exposes — the assignment's Case Management UI requirement.

| File | What problem it solves |
|---|---|
| `src/lib/types.ts` | TypeScript mirror of the backend's Pydantic models — kept hand-written and in sync deliberately (no OpenAPI export step exists in this project) |
| `src/lib/api.ts` | The typed fetch client, bearer-token auth matching the backend's defaults |
| `src/components/StatusBadge.tsx` | Consistent status/decision color-coding everywhere it's shown |
| `src/components/ReviewActions.tsx` | The HITL confirm/override UI — the human half of Component 9 |
| `src/app/page.tsx` | Dashboard — every case, status, decision, at a glance |
| `src/app/cases/[caseId]/page.tsx` | Case detail — every agent's panel, the Orchestrator's justification, the audit trail, and the review controls when a case is paused |
| `src/app/review/page.tsx` | Review queue — the dashboard filtered to what actually needs a human right now |
| `src/app/upload/page.tsx` | The intake form itself |

## 17. Tests: `backend/tests/`

**Problem it solves:** verify code logic (routing, validation, decision
rules, error handling) without spending real API money on every check.
193 tests, every LLM call mocked — the suite validates that *this code*
behaves correctly given a model's output, not whether the model itself is
good at its job (that's what the eval suite, §14, is for). Runs in under
a minute, costs $0, and is safe to run as often as you like — including
live, during a presentation, as proof the codebase is in a known-good
state.

---

For the story of what actually broke while building each of these — and
how it got fixed — read `IMPLEMENTATION_CHALLENGES.md`, organized in the
same rough order as this document.
