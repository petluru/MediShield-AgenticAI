# MediShield — Implementation Challenges & Learnings

This is a running log of every real technical obstacle hit while building
this system, why it happened, how it was fixed, and what it teaches about
the tools involved. It's meant to be read, not just referenced — if you
want to explain this project (or defend a design decision) in an
interview, this is the "how it was actually built, warts and all" version
that `PROJECT_PLAN.md` (the forward-looking plan) doesn't capture.

Updated as new tasks surface new obstacles — this is not a one-time
snapshot.

---

## 1. Environment & Dependency Setup

### 1.1 Python 3.10 → 3.11 (a transitive dependency had no 3.10 wheels)

**What happened:** `uv sync` failed immediately:
```
error: Distribution `onnxruntime==1.24.3` can't be installed because it
doesn't have a source distribution or wheel for the current platform
hint: You're using CPython 3.10, but onnxruntime (v1.24.3) only has wheels
with the following Python implementation tags: cp311, cp312, cp313, cp314
```

**Why:** `onnxruntime` isn't a direct dependency of this project — it's
pulled in transitively by `docling` (used for the Policy RAG PDF
ingestion, task #6) for its layout/OCR models. `onnxruntime` simply
stopped publishing Windows wheels for Python 3.10 as of that version. The
original plan assumed 3.10 because that's what `.python-version` was set
to before anyone had actually tried installing the full dependency tree.

**Fix:** Bumped `.python-version` and `requires-python` in
`pyproject.toml` to `>=3.11`. `uv` downloaded and cached `cpython-3.11.14`
automatically — no manual Python install needed.

**Takeaway:** Pin a Python version early, but don't trust it until you've
actually run `uv sync` (or `pip install`) against the *real* dependency
tree — transitive dependencies can have narrower platform support than
your direct ones, and you won't know until you try.

### 1.2 OneDrive-synced project directory breaks `uv`'s default install mode

**What happened:** After fixing the Python version, `uv sync` failed
differently, mid-install:
```
error: Failed to install: jiter-0.16.0-cp311-cp311-win_amd64.whl
Caused by: failed to hardlink file ... The cloud operation cannot be
performed on a file with incompatible hardlinks. (os error 396)
```

**Why:** This project's working directory lives under
`OneDrive - EPAM\Documents\...`. `uv`'s default install strategy
hardlinks packages from its global cache into `.venv` for speed. OneDrive
represents synced files as cloud placeholders, and Windows refuses to
hardlink those — hardlinks require both link and target to be regular,
fully-resident files on the same volume, and OneDrive's virtualization
layer breaks that assumption.

**Fix:** Set `UV_LINK_MODE=copy` before every `uv` invocation in this
repo, which makes `uv` copy files into `.venv` instead of hardlinking
them. Slightly slower installs, otherwise invisible.

**Takeaway:** Cloud-synced folders (OneDrive, Dropbox, iCloud Drive) and
tools that assume a "normal" local filesystem (hardlinks, file locks,
inotify-style watchers) don't mix well. If you hit a mysterious
filesystem error on Windows in a project under `Documents`, check whether
OneDrive sync is the actual culprit before assuming it's a tool bug.

---

## 2. LLM API Integration Quirks (claude-sonnet-5 / langchain 1.x)

### 2.1 `claude-sonnet-5` rejects `temperature` outright

**What happened:** The very first live API call from the Classifier Agent
failed with a real HTTP 400, not a silent ignore:
```
anthropic.BadRequestError: Error code: 400 - {'type': 'error', 'error':
{'type': 'invalid_request_error', 'message': '`temperature` is deprecated
for this model.'}}
```

**Why:** Earlier Claude model families accepted (and mostly ignored,
functionally) a `temperature` parameter for controlling response
randomness. The claude-5 family removed it as a request parameter
entirely — passing it, even `temperature=0`, is now a hard error rather
than a no-op.

**Fix:** Stopped passing `temperature` anywhere. All `ChatAnthropic(...)`
construction goes through one shared factory
(`backend/agents/llm_factory.py:build_chat_anthropic`) specifically so
this had to be fixed in exactly one place instead of six (once per
agent).

**Takeaway:** When a new model family ships, don't assume old
"harmless" parameters still are — some get promoted from
ignored-but-accepted to a hard validation error. Centralizing model
construction behind one factory function paid for itself the moment this
was discovered.

### 2.2 `ChatAnthropic(...)`'s constructor is stricter than it looks

**What happened:** `mypy` flagged three separate issues on what looked
like a normal constructor call:
```
error: Unexpected keyword argument "model" for "ChatAnthropic"
error: Argument "api_key" has incompatible type "str"; expected "SecretStr"
error: Missing named argument "timeout" for "ChatAnthropic"
error: Missing named argument "stop" for "ChatAnthropic"
```

**Why:** `ChatAnthropic` is a Pydantic v2 model. Pydantic v2 classes are
`@dataclass_transform`-decorated, which means type checkers (mypy, not
just IDEs) synthesize a strict `__init__` signature directly from the
declared fields — using each field's **alias**, not its Python attribute
name, when one is set. The field is named `model` internally with alias
`model_name`; at *runtime* `populate_by_name=True` means either spelling
works, but mypy's synthesized signature only recognizes the alias. Same
story for `api_key` (aliases `anthropic_api_key`, typed as `SecretStr`,
not plain `str`), and `timeout`/`stop` are fields with no default, so the
synthesized signature makes them mandatory even though most real usage
doesn't think about them.

**Fix:** `build_chat_anthropic()` passes `model_name=`,
`api_key=SecretStr(...)`, `timeout=60`, `stop=None` — the exact shape
mypy (and, it turns out, correctness) wants.

**Takeaway:** For any Pydantic-v2-based SDK class, don't guess the
constructor from docs or old examples — inspect `ClassName.model_fields`
to see actual field names, aliases, and required-ness. mypy's synthesized
signature is often the most accurate documentation available, precisely
because it's derived from the real field metadata, not written by hand.

### 2.3 A mypy false-positive tied to an *unrelated* type annotation

**What happened:** This exact code type-checked fine at module level (as
in `test_config.py`):
```python
settings = Settings(_env_file=None, ANTHROPIC_API_KEY="x")
```
But the same call, wrapped in a helper with an explicit return
annotation, did not:
```python
def make_settings() -> Settings:
    return Settings(_env_file=None, ANTHROPIC_API_KEY="x")
    # error: Unexpected keyword argument "_env_file" for "Settings"
```

**Why:** `_env_file` is a special `pydantic-settings` constructor
parameter (it overrides which `.env` file to load for just this
instantiation), not a declared model field. When mypy has to unify a
call's inferred type against an explicit `-> Settings` return annotation,
it appears to walk a different, stricter overload-resolution path for
the dataclass-transform-synthesized constructor — one that loses track
of `_env_file` even though the bare, unannotated call resolves it fine.
This is a narrow, somewhat obscure mypy/pydantic-settings interaction,
confirmed by direct A/B testing (identical call, only the surrounding
function's return annotation changed).

**Fix:** Dropped the explicit return type annotation on the test helper
and let mypy infer it. Trivial fix, but only findable by bisecting what
actually differed between the passing and failing case.

**Takeaway:** Not every mypy error is a real bug — some are the type
checker's own resolution quirks. Before "fixing" code to satisfy a type
checker, reproduce the error in isolation and vary one thing at a time.
Here, that took four small experiments (module-level vs. function-wrapped,
in-repo vs. `/tmp`, different field names) to find the actual variable
that mattered (the return annotation), rather than guessing.

---

## 3. Agent Prompt-Engineering Tradeoffs

These aren't bugs — they're judgment calls made visible by testing
against real (if synthetic) data, which is exactly why the plan called
for smoke-testing every agent against the dataset before moving on.

### 3.1 KYC tamper detection: false positives vs. false negatives

**What happened:** A first version of the KYC Agent's system prompt
("flag font inconsistencies, pixel artifacts, signs of digital editing")
flagged **5 of 6** smoke-test ID images as tampered — including clean
ones with no tampering at all.

**Why:** The synthetic dataset generator renders every ID with a red DOB
field and a generic gray silhouette in the photo box — as a fixed,
uniform template choice, not per-document variation. The model correctly
noticed both were visually different from "expected" ID styling, but
had no way to know they were *intentional and universal* rather than
signs of tampering, so it flagged nearly everything.

**Fix:** Rewrote the prompt to explicitly name these as non-evidence
("many legitimate documents render one field in a different color for
emphasis... this alone is NOT evidence of tampering") and raised the bar
for what counts as real tamper evidence (font *typeface* mismatch,
irregular kerning, localized pixel/compression artifacts — not just a
color or a placeholder photo). This fixed the false positives (2 clean
IDs now correctly pass), but as a direct consequence, it also stopped
catching one genuinely tampered ID in the dataset — a ~4px expiry-date
font-size shift on a deliberately skewed scan, which is subtle enough
that even a careful human glance misses it (verified by looking at the
image directly).

**The tradeoff, made explicit:** A KYC gate that's too aggressive blocks
real customers (high false-positive cost, at scale, is a real business
and trust problem). A KYC gate that's too lenient misses fraud, but this
system doesn't rely on KYC as the only fraud signal — the Fraud Detection
Agent (task #7) cross-references patient/claim history, and anything the
Orchestrator isn't confident about gets a human-in-the-loop review before
a final decision. Given that safety net, erring toward fewer false
positives at the KYC layer specifically was the right call, and it's
documented in code (`backend/agents/kyc.py`'s module docstring) so it
doesn't look like an oversight later.

**Takeaway:** Precision/recall tradeoffs in prompts are real engineering
decisions, not bugs to "eventually fix" — they should be made
deliberately, tested against real examples (not just imagined ones), and
written down with the reasoning, because the "obviously correct" fix
(catch more tampering!) has a real cost (false-positive rate) that's easy
to miss if you only look at the one case you're trying to fix.

### 3.2 Claims Agent: NPI format didn't match the real-world convention I assumed

**What happened:** The Claims Agent extracted provider NPIs correctly in
every smoke test, but flagged **3 of 4** as schema-invalid:
```
provider NPI 'NPI-5610293847' doesn't match the expected 10-digit format
```

**Why:** I wrote the validation regex (`^\d{10}$`) against the
real-world NPI standard (always exactly 10 bare digits). But this
dataset's generator (`generate_docs.py`) formats every provider's NPI
with a literal `"NPI-"` prefix baked into the fixture data
(`"NPI-1029384756"`), consistently, across every claim form. The
extraction was correct — my validation assumption about the format was
wrong for *this* dataset.

**Fix:** Widened the regex to `^(NPI-)?\d{10}$`, accepting both the
dataset's prefixed convention and the bare real-world format.

**Takeaway:** When validating extracted data, verify format assumptions
against the actual source data (or its generator, if synthetic) before
writing the regex — don't assume a "well-known standard" format applies
without checking. This is also a good example of why smoke-testing
against real dataset images (not just trusting the code compiles) caught
a real, silent-failure-shaped bug: without it, every single claim in this
dataset would have been incorrectly marked schema-invalid.

---

## 4. RAG Infrastructure (Docling + Chroma)

### 4.1 Chroma's default distance metric produced meaningless "relevance" scores

**What happened:** Retrieval worked (the right policy clauses came back
ranked correctly), but every result showed a near-zero relevance score
regardless of how good the match actually was:
```
[plan=gold | section=4. Exclusions | relevance=0.04]   <- this was a great match
```

**Why:** Chroma's default vector index metric is squared L2 (Euclidean)
distance, which is unbounded — its range depends on the embedding
vectors' norms, not a fixed `[0, 1]` or `[0, 2]` interval. My formatting
code assumed `relevance = 1 - distance`, which only makes sense for a
distance metric bounded near `[0, 1]`. Squared L2 distances between
MiniLM embeddings are routinely > 1, so `1 - distance` went negative and
got clamped to 0 almost every time — the underlying retrieval was fine,
the display math was wrong.

**Fix:** Recreated the collection with `metadata={"hnsw:space":
"cosine"}` (cosine distance is bounded `[0, 2]`, 0 = identical) and
changed the formula to `relevance = 1 - distance / 2`. Had to delete and
re-ingest the collection since Chroma's distance metric is fixed at
collection-creation time, not query time.

**Takeaway:** A distance/relevance score is only meaningful relative to
the metric that produced it — always check what metric a vector index
actually uses by default before writing code that interprets its output
as a normalized score. This kind of bug is dangerous specifically because
it fails *silently*: nothing crashes, the numbers just quietly mean
nothing, and it would have fed directly into the Policy Agent's own
"is this retrieval weak? re-query if so" judgment call in the next task.

### 4.2 A Chroma client race condition surfaced only through the agent's tool-calling loop

**What happened:** Direct calls to the retrieval tool worked fine in
isolation. Running the same tool through the Policy RAG Agent's
`create_agent` tool-calling loop crashed with:
```
ValueError: Could not connect to tenant default_tenant. Are you sure it
exists?
AttributeError: 'RustBindingsAPI' object has no attribute 'bindings'
```

**Why:** The retrieval tool called `chromadb.PersistentClient(path=...)`
fresh on every invocation. That's harmless when calls happen one at a
time. But LangGraph's tool-execution node (`ToolNode`) runs multiple tool
calls from a single LLM turn **concurrently**, on a thread pool — and
when Claude's response includes more than one tool call in one turn
(which it does when refining a weak initial retrieval, exactly the
"re-query if results are weak" behavior this agent is designed to do),
two threads ended up constructing a brand-new `PersistentClient` against
the same on-disk SQLite-backed path at the same instant. That raced on
Chroma's internal tenant/database bootstrap check and corrupted the
client's internal state.

**Fix:** Replaced per-call client construction with a single
lazily-created, lock-guarded client shared for the process's lifetime
(`backend/rag/ingest.py:_get_client`), following the same "one client,
reused" pattern Chroma's own docs assume for multi-threaded apps.

**Takeaway:** Some bugs only exist at the intersection of two components
that are each individually correct — the retrieval tool was fine, and
`create_agent`'s parallel tool execution is fine (and desirable — it's
faster). The bug was an implicit assumption ("this function is called
once at a time") that held during unit testing (mocked, so no real
concurrency) and during a manual single-call test, but broke under the
actual agent's real usage pattern. This is a good argument for a genuine
end-to-end smoke test, not just mocked unit tests — the mocks in
`test_retrieval.py` and `test_policy_rag.py` never called the real
`chromadb.PersistentClient` constructor, so they couldn't have caught
this.

---

## 5. Security / Credential Handling

### 5.1 A real API key ended up in the wrong file (and in this chat transcript)

**What happened:** While setting up billing, the real Anthropic API key
got pasted into `.env.example` instead of `.env`. This matters because
`.gitignore` explicitly does **not** ignore `.env.example`
(`!.env.example`) — it's meant to be the committed template with
placeholder values that other developers (or a grader) copy from. Had a
`git init` + first commit happened before this was caught, the real key
would have landed in git history on the very first commit.

**Fix:** Copied the real values into `.env` (which *is* gitignored),
restored `.env.example`'s `ANTHROPIC_API_KEY` to a placeholder
(`sk-ant-...`), and flagged that the real key had also been visible in
the chat conversation itself — recommended rotating it (revoke +
regenerate in the Anthropic Console) since a conversation transcript
isn't a secrets vault, independent of whether it ever reached git.

**Takeaway:** The `.env` / `.env.example` split only works as a safety
mechanism if real secrets consistently go in the *un*tracked file — one
copy-paste into the wrong one defeats it silently, with no error or
warning at write time. The failure mode isn't visible until you check
`git status` (or, worse, only after a push). Two habits make this safer:
double-check which of the two files you're editing before pasting a
credential, and treat "did a secret briefly touch a chat transcript or
the wrong file" as reason enough to rotate it, even if it never actually
reached a public place — free/cheap to rotate, expensive to be wrong
about how exposed it was.

---

## 6. Cost Control Tooling

### 6.1 A token-usage logger that silently double-counted every cache hit

**What happened:** After adding `SQLiteCache` (exact-match response caching
— a request identical to a prior one is served from disk for free, no API
call) and a callback that logs `usage_metadata` after every LLM call, a
quick verification showed something wrong: calling the same agent on the
same image three times in a row (1 real call + 2 cache hits) produced
**three** identical log entries, all reporting the same token counts —
even though only the first call actually touched the API. Left alone, this
would have made the token-usage log systematically overstate cost every
time the cache did its job, which defeats the entire purpose of tracking
it accurately.

**Why:** LangChain's caching layer (`_generate_with_cache`) intercepts
*before* the API call, and on a hit, replays the previously-stored
response through the exact same callback path a fresh generation would
use — `on_llm_end` fires either way, with no separate "this was a cache
hit" signal exposed at the callback layer. Confirmed empirically by
dumping `response_metadata` for a fresh call vs. a cached replay of the
same request: byte-identical output, **including the same Anthropic
message `id`** — a real second API call always gets a new unique `id`
from Anthropic, so a repeated `id` is the one reliable tell that no
tokens were actually spent.

**Fix:** The logger now dedupes against response `id`s already present in
the log file before writing a new row — reading from the file itself
(`backend/agents/llm_factory.py:_TokenUsageLogger._already_logged`),
not an in-memory set, since separate smoke-test script runs are separate
processes and still need to recognize a cache hit that was originally
written to disk by an earlier run of a *different* script.

**Takeaway:** A callback firing is not proof that work happened — caching
layers are specifically designed to make a cache hit indistinguishable
from a fresh call at the level most instrumentation hooks into, so cost
telemetry built on "count every callback" without checking for a fresh
vs. replayed signal will double-count exactly the savings it's supposed
to be measuring. When in doubt, verify empirically (dump the response
metadata for both cases side by side) rather than assuming a callback
means "billed."

### 6.2 Policy RAG's tool-calling loop had no exit condition — 103 iterations, ~7.9M tokens, one query

**What happened:** During the first live end-to-end run of the full
LangGraph pipeline (task #8), regenerating the token usage summary showed
something alarming: a `policy_rag` row with **103 calls and ~7.9 million
input tokens** (~$16 of the ~$16.15 total logged that day) — from a single
case. Reading the raw log confirmed it wasn't corrupted data: 103 real,
successful calls in one continuous ~7-minute stretch, with input tokens
climbing steadily and almost linearly from 1,672 to 141,265 across the
run. The run only stopped because the account ran out of credits — nothing
in the code itself ever called a halt.

**Why:** `create_agent` has no built-in cap on tool-calling iterations, and
the Policy RAG Agent's own system prompt explicitly told it to "re-query
with different wording" whenever retrieval results looked weak — with no
instruction on when to stop trying. Each iteration appends the previous
tool call and its result to the conversation, so token cost per call grows
with iteration count, not just call count: 103 linearly-growing calls cost
far more than 103 calls at a flat size would. The specific query that
triggered it (`gold` plan, CPT `59400` a maternity code, paired with
diagnosis `K35.80` appendicitis) is a clinically nonsensical combination —
exactly the kind of query where the policy documents will never contain a
clause that "clearly answers" the question, so the model's own "keep
re-querying until you get a clear answer" instruction never had a natural
exit.

**Fix:** Two layers, matching the security notebook's "no single fix
stops it, you layer defenses" pattern used elsewhere in this project:
1. A hard cap via LangGraph's `config={"recursion_limit": N}` passed to
   `agent.invoke(...)` (`backend/agents/policy_rag.py`, and applied to
   Fraud Detection's identical `create_agent` pattern too, even though
   its simpler single-lookup tool is far less likely to spiral).
2. `GraphRecursionError` is caught and turned into a safe, low-confidence
   fallback result — `PolicyOutput(covered=False, confidence=0.0, ...)`
   for Policy RAG, a fraud_score forced above the escalation threshold for
   Fraud Detection — rather than letting the exception crash the whole
   case pipeline. This isn't just error handling for its own sake: a
   confidence of 0.0 automatically trips the Orchestrator's own ESCALATE
   rule (`confidence < confidence_escalate_max`), so a case that hits this
   limit correctly falls back to human review instead of silently
   approving, silently rejecting, or crashing outright.
3. Tightened the system prompt itself to explicitly cap re-querying at
   twice, as defense-in-depth alongside the hard limit — reduces how often
   the cap actually gets hit, rather than relying on the cap alone.

**Takeaway:** An LLM instruction like "keep trying until you get a clear
answer" has no implicit exit condition — if the underlying data genuinely
can't answer the question, "keep trying" can mean "keep trying forever."
Any prompt that tells a tool-calling agent to retry or re-query needs an
explicit stopping condition in the prompt *and* a hard structural cap in
code — the prompt-level instruction is necessary but not sufficient, since
it's exactly the kind of soft constraint an LLM can rationalize past when
every individual re-query still seems reasonable in isolation. This is
also a good example of why the token-usage logging built earlier this
session (SS6.1) wasn't just a cost-reporting nicety — it's what surfaced
this bug at all. Nothing crashed, nothing errored (until the credit limit
hit), and the pipeline's own final output for that case would have looked
completely normal; the only signal that something was badly wrong was the
token count in a log file. Instrumentation built for one purpose
(answering "how many tokens have we used") ended up catching an
unrelated, more serious problem (a runaway agent loop) — a reminder that
observability tends to pay for itself in ways beyond its original reason
for existing.

---

## 7. Evals & Multi-Agent Interaction Edge Cases

### 7.1 A "clean" claim form got REJECTed — not a bug, a real multi-CPT-code retrieval limitation

**What happened:** The first real eval run (task #9's harness, 10-case
sample against `dataset/metadata.json`'s ground truth) landed 70% Decision
Correctness — passing the assignment's 60% bar — with 3 mismatches. Two
were the expected detectability gap (Fraud Detection can't see a discharge
summary's or prescription's own fields, documented when the harness was
built). The third, `claim_PT_99733`, was unexpected: ground truth said
APPROVE (this cluster's injected fraud signal, `date_conflict`, lives on
the *prescription* document, not this claim — so the claim form itself
carries no fraud/edge label), but the pipeline returned REJECT.

Investigated for free by reading the case's already-persisted LangGraph
checkpoint (`graph.get_state(config)` against the real `SqliteSaver` the
eval run used) rather than spending another API call — the full state,
including every agent's reasoning, was already on disk.

**Why:** the claim form's extracted CPT codes were `80053, 90837, 27447,
99291` — a metabolic panel, psychotherapy, total knee arthroplasty, and
critical care, all billed the same service date. This combination is
clinically incoherent, almost certainly an artifact of the synthetic
dataset generator assigning CPT codes without enforcing clinical
coherence for "clean" (non-fraud-labeled) clusters, not a deliberately
injected test signal. Two independent, correct agent behaviors then
compounded:
1. **Fraud Detection correctly flagged the bundle** as a "clinically
   implausible procedure bundle" (possible upcoding), scoring 0.35 —
   just over the 0.3 escalation threshold, entirely reasonable given what
   it was actually handed.
2. **Policy RAG's `_build_query` joins every CPT code from a claim into
   one combined query string** (`backend/agents/policy_rag.py`). Asked to
   resolve coverage for all four codes at once, it never converged and
   hit the `recursion_limit=10` safety cap (see SS6.2 — the same guard
   that fixed the original runaway-loop incident did its job here: capped
   at 10 iterations, not 103, and degraded to a safe `covered=False,
   confidence=0.0` fallback instead of crashing or guessing). Confirmed
   this is specifically a *multi-code* problem, not a general Policy RAG
   weakness: CPT 27447 queried alone, in this same session's prompt-caching
   validation work, resolved cleanly (`covered=True`, 80%, confidence 0.68).

The Orchestrator's deterministic rule then correctly REJECTed on
`covered=False`, and — notably — its LLM-written narrative was honest
about the uncertainty: it explicitly called out that Policy RAG's own
confidence was 0.00 and that it "did not converge," rather than presenting
the rejection as more certain than the evidence supported.

**Resolution:** no code changed. This is a real, defensible system
behavior given a synthetic-data artifact, not a defect — expected ground
truth reflects the *intended* per-cluster test signal, and this document
just happened to also trip an unrelated, genuine limitation. Two real,
non-urgent findings worth keeping:
- Policy RAG has no per-CPT-code query strategy — a multi-procedure claim
  spanning unrelated clinical domains can fail to converge even when each
  individual code would resolve cleanly. Worth revisiting if real claim
  volume shows this isn't rare (e.g. query once per CPT code and
  aggregate, instead of one combined query) — not worth doing speculatively
  now.
- The synthetic dataset can produce clinically-incoherent CPT bundles on
  clusters that weren't meant to test anything suspicious. Worth keeping
  in mind when interpreting future eval mismatches on `claim_forms` cases
  that "shouldn't" have anything wrong with them — check the actual
  extracted codes before assuming a mismatch is a pipeline bug.

**Takeaway:** an eval mismatch against ground truth isn't automatically a
bug — the checkpointer persisting full per-case state (built for the HITL
gate, not for debugging) turned out to make this kind of investigation
free and fast, the same "instrumentation built for one purpose catches an
unrelated problem" pattern as SS6.1/6.2. Reading real agent reasoning
before touching any code confirmed this was two individually-correct
agent behaviors compounding on an unintentionally messy input, not
something worth reflexively "fixing."

### 7.2 Full `claim_forms` eval run (31 real cases, $2.9062): 51.6% Decision Correctness, FAILS the 60% bar — three real, distinct causes, none are bugs

**What happened:** with a hard $14 API budget in mind, the full 155-doc
eval was staged (`claim_forms` first — highest cost and variance —
then the rest). The real 31-case run cost $2.9062 (near the low end of a
$2.80–$6.20 estimate built from the 10-case pilot). Decision Correctness
came in at 51.6%, below the assignment's 60% minimum, with 15/31
mismatches. Every case was investigated for free by reading its persisted
LangGraph checkpoint (same technique as SS7.1) rather than spending
anything further.

**Why (three separate causes, ranked by impact):**
1. **~9 of 15 mismatches:** the synthetic dataset assigns clinically
   incoherent CPT-code bundles to claim forms on clusters that were never
   meant to test fraud. Real examples: `claim_PT_67125` (endoscopy +
   psychotherapy + total knee arthroplasty, $45,650, same date),
   `claim_PT_81810` (psychotherapy + knee arthroplasty + CT chest +
   critical care, $48,350). Fraud Detection correctly flags these as
   implausible bundles (fraud_score 0.35–0.6) — this is the system working
   as designed on an accidentally-suspicious input, the same pattern
   SS7.1 found on one case, now confirmed systematic across ~29% of this
   category.
2. **~4 of 15:** Policy RAG hits `recursion_limit=10` and returns
   `covered=False, confidence=0.0` (the SS6.2 safety-cap fallback,
   confirmed by its exact signature). On 2 of these the case was a real
   fraud cluster (`C_003`, `C_004`) — REJECT fired from the coverage
   fallback before Fraud Detection's ESCALATE could even be evaluated
   (REJECT is checked first in the decision rule), accidentally masking
   the intended test signal. ~13% hit rate across this real run — not
   rare, not the 50% the 2-case pilot suggested, but a real recurring cost
   (SS7.1 already flagged Policy RAG's single-combined-query design as a
   contributor).
3. **2 of 15, real and narrower:** the `uncovered_procedure` cluster pair
   using CPT `17000` (cosmetic lesion removal) was incorrectly approved as
   covered, while the 3 clusters using CPT `21120` were correctly
   rejected — a genuine, CPT-specific Policy RAG retrieval gap, not
   explained by either cause above.

One additional honest caveat: `claim_PT_20322` (the duplicate-claim
cluster's *original* claim, not the `_dup` copy) got flagged for the
patient-level duplicate signal, which Fraud Detection's tool can see
regardless of which document in the cluster triggers the case. Ground
truth (SS[[ground_truth.py]]) only labels the specific `_dup` document as
expected-ESCALATE, by deliberate methodological choice — this "miss" may
be under-crediting genuinely correct behavior, not a real error.

**Resolution:** no code changed. All three causes are either dataset
artifacts (cause 1) or already-known, already-guarded limitations (cause
2, SS7.1/SS6.2) or a narrow, real gap worth a future look but not urgent
given only 2 cases (cause 3). Re-running with different ground truth
wouldn't be honest either — the point of the eval is to surface exactly
this kind of gap between "intended" and "actual," and it did.

**Takeaway:** a headline eval number (51.6%, below bar) can be
simultaneously true and misleading if reported without the per-case
breakdown. The dataset's own quality (CPT-code coherence on "clean"
clusters) turned out to be a bigger driver of the score than any pipeline
defect — worth remembering before trusting any single aggregate metric on
a synthetic dataset without checking what's actually driving it.

### 7.3 Full 155-doc eval (stage 2, 124 remaining cases, $2.4555): 78.1% Decision Correctness overall — PASSES the bar, plus two genuinely new bugs

**What happened:** stage 2 (id_documents, discharge_summaries,
prescriptions, policy_amendments, unknown — 124 cases) completed for
$2.4555, combined with stage 1 (SS7.2) via free checkpoint reads (no
re-run) into a full 155-case report: **95.5% Classification Accuracy,
96.0% Extraction Completeness, 78.1% Decision Correctness — passes the
assignment's 60% bar**, even though `claim_forms` alone (SS7.2) had
failed it. `claim_forms` is only 20% of the dataset by count, and the
other 80% scored well enough to pull the blended number comfortably over
the bar.

**Why (34 total mismatches across the full run, grouped by cause):**

1. **Not a bug — ground truth under-credits a real patient-level signal
   (4 cases):** `id_PT_20322`, `discharge_PT_20322`, `rx_PT_20322`,
   `amend_PT_20322` (all non-`_dup` documents of the C_005 duplicate-claim
   patient) all correctly ESCALATE. Fraud Detection's tool returns every
   claim on record for a patient regardless of which document triggered
   the case, so the duplicate signal is visible — and correctly acted
   on — from any of that patient's documents, not just the `_dup` claim
   itself. Ground truth (`backend/evals/ground_truth.py`) only labels the
   specific `_dup` document as expected-ESCALATE by deliberate
   methodological choice (documented in that module already); this
   confirms the system is *more* consistent than that choice gives it
   credit for.
2. **Already-known, deliberate tradeoff, now quantified (3 cases):**
   `id_PT_57795`, `id_PT_62383`, `id_PT_74454` — all 3 real `tampered_id`
   test cases missed. This is the KYC tamper-detection calibration
   documented from early in the project (SS3.1 in this file / see
   `backend/agents/kyc.py`'s own docstring): deliberately conservative to
   avoid false-positiving on clean documents. 0/3 catch rate on the real
   test cases is the concrete cost of that tradeoff, now measured rather
   than assumed.
3. **NEW real bug: KYC rejects "expiring soon" IDs, contradicting its own
   system prompt (4 of 5 cases):** `id_PT_50538`, `id_PT_67125`,
   `id_PT_54236`, `id_PT_88139` all REJECT when they should APPROVE.
   `backend/agents/kyc.py`'s system prompt explicitly states an expiring-soon
   document "still passes... add an 'expiring_soon' flag" — the model is
   not reliably following its own stated policy. Only `id_PT_99687` (the
   5th real expiring-soon case) got this right. Not yet root-caused
   further (would need to inspect the actual model responses/flags
   returned) — flagged as a real, open issue, not fixed here.
4. **NEW real gap: a confidently-misclassified "wrong document type" never
   escalates (2 of 4 `unknown` cases):** `unknown_bank_statement_001` and
   `unknown_utility_bill_001` both APPROVE when they should ESCALATE. The
   Classifier correctly and *confidently* labels these as UNKNOWN (high
   confidence), so the Orchestrator's only escalation trigger for this
   category — low agent confidence — never fires; nothing else about a
   bank statement looks fraudulent to Fraud Detection, so it sails
   through as APPROVE. The other 2 `unknown` cases (illegible/blank scans,
   genuinely low classifier confidence) correctly escalated. This is a
   real design gap: `doc_type == UNKNOWN` isn't itself a deterministic
   escalation trigger, only low confidence is, and those are two different
   things. A real fix candidate: add `doc_type == DocType.UNKNOWN` as its
   own deterministic condition in `requires_human_review`/
   `compute_decision`, the same auditable-rule pattern already used for
   everything else in the Orchestrator — not implemented here, flagged for
   a future session.
5. **NEW real infrastructure bug: 3 `policy_amendments` images failed at
   the Anthropic API level, with no graceful fallback (2 distinct causes):**
   - `amend_PT_39451`: 7.70MB PNG -> ~10.27MB base64, over Anthropic's
     10MB base64 image limit (`BadRequestError`, "image exceeds 10 MB
     maximum"). Confirmed by checking the real file size — genuinely too
     large, not a fluke.
   - `amend_PT_69470` (1.69MB) and `amend_PT_50538` (5.20MB) both got a
     generic "Could not process image" error, well under the size limit —
     a different, file-specific issue (likely a malformed/corrupted PNG
     or an unusual encoding), not diagnosed further here.
   - **The real gap:** `backend/agents/vision_utils.py`'s `encode_image`
     and every vision-agent node have no error handling for either case —
     unlike the tool-loop agents' `GraphRecursionError` fallback
     (SS6.2), a case whose image can't be processed at all currently has
     no safe-degradation path in the real pipeline (only the eval
     harness's own try/except, added for eval robustness, caught these
     three). A real production case hitting this today would crash the
     graph node with an unhandled exception. Not fixed here — flagged as
     a real, concrete gap worth closing before this pipeline sees
     real documents outside the curated dataset.

Also confirmed (not new): `discharge_PT_19116`, `rx_PT_99733`, `id_PT_39451`
(name_mismatch) — the same structural detectability gap documented when
ground truth was built (Fraud Detection's tool only exposes claim_forms
metadata; no agent anywhere extracts or compares names). `claim_forms`'
15 mismatches are SS7.2's already-documented findings, unchanged on
re-read.

**Resolution:** no code changed this session for any of these. Two are
confirmed non-bugs (ground-truth methodology, already-known tradeoff);
three are real, actionable findings recorded for a future session: the
`expiring_soon` prompt-adherence issue, the `UNKNOWN`-doc-type escalation
gap (with a concrete, low-risk deterministic-rule fix candidate already
identified), and the oversized/malformed-image handling gap (needs a
resize-before-encode path for oversized files and a safe fallback for
unprocessable ones, mirroring the `GraphRecursionError` pattern).

**Takeaway:** the full-dataset run did what staging was supposed to do —
it answered the real open question (does the *overall* score clear 60%
despite `claim_forms` failing on its own? yes, 78.1%) cheaply, and it
surfaced two genuinely new, non-obvious bugs (a prompt-adherence gap and
an unhandled image-processing failure mode) that neither the unit tests
nor the `claim_forms`-only run could have found, because neither exercises
KYC's expiring-soon path at volume or touches large/malformed
`policy_amendments` images at all. Running the cheap, low-variance
categories was not wasted effort just because the outcome was "mostly
as predicted."

### 7.4 Closing out the 3 known bugs (2026-08-05): 1 was already fixed, 1 turned out not to be a code bug at all, 1 got a real fix

Per PROJECT_PLAN.md SS10, these were deliberately deferred until the very
end (explicit user choice: "frontend now, bugs at the very end", after the
frontend — SS13 below — was done). Revisiting all three with fresh eyes:

**Bug #4 in the original SS7.3 list (`UNKNOWN`-doc-type escalation gap):
already fixed and validated earlier the same session** (see the
`orchestrator.py` docstring dated 2026-08-05 and
`test_compute_decision_escalates_on_unknown_doc_type_even_with_high_confidence`
in `backend/tests/test_orchestrator.py`) — real live-API re-runs of both
failing cases confirmed. Only noted here because [[medishield-known-bugs]]
had gone briefly stale claiming it was still open.

**Bug #3 in the original list (`expiring_soon` KYC rejections): NOT a code
bug — root-caused to dataset staleness, no code changed.** Read the 5
actual ID images directly (Claude's own vision, not a project Anthropic
API call — zero cost against the project's budget) instead of guessing
from the model's flags:

| Case | Printed expiry | Real "today" when re-examined |
|---|---|---|
| `id_PT_50538` | 07/10/2025 | 2026-08-05 |
| `id_PT_67125` | 02/07/2025 | 2026-08-05 |
| `id_PT_54236` | 12/02/2025 (passport) | 2026-08-05 |
| `id_PT_99687` | 11/09/2025 (member card) | 2026-08-05 |

Every one of these is over a year past its printed expiry relative to
real wall-clock time. `dataset_summary.md` states the dataset was
`**Generated:** 2026-08-01` — so these dates were *already* in the past
even at generation time, not just stale by the time the eval ran a few
days later. This is a bug in the (external, not-in-this-repo) dataset
generator's date logic, not in `backend/agents/kyc.py`. `kyc.py`'s
`verify_kyc` compares the document's printed expiry against
`datetime.now(timezone.utc).date()` — genuinely, correctly "expired" by
that comparison. Changing this comparison to make these specific test
fixtures pass would make production KYC *wrong* for any real expired ID.
**No code changed.** Documented here and in [[medishield-known-bugs]] as
a permanent dataset limitation: any future eval re-run will keep
reporting these 5 `expiring_soon_id`-labeled cases as REJECT-not-APPROVE
mismatches, forever, regardless of when it's run — that's expected, not a
regression, and not worth chasing further without regenerating the
dataset with dynamically-computed expiry dates (out of scope; the
generator itself was never part of this repo).

**Bug #5 in the original list (image-handling crash): real fix shipped.**
Confirmed all 3 failure causes locally first, zero API cost:
`Image.open(...).verify()`/`.load()` on the actual dataset files showed
`amend_PT_39451` is genuinely oversized (8.07MB raw / 10.76MB base64, over
Anthropic's ~10MB base64 limit) while `amend_PT_69470` and `amend_PT_50538`
are genuinely **corrupted PNG files** — PIL errors "broken PNG file
(incomplete checksum in b'IDAT')" and "Truncated File Read" respectively.
Also dataset artifacts, not something this code caused — but the *absence*
of any error handling around them was a real, worth-fixing gap regardless
of whose fault the specific files were, since the FastAPI upload endpoint
(SS9 below) now means a real user really can upload an oversized or
corrupted image.

Fix, in `backend/agents/vision_utils.py`: `encode_image` now (1) forces a
full PIL decode before ever base64-encoding, so a corrupted/truncated file
raises a typed `ImageProcessingError` instead of silently producing
garbage bytes the API would reject with an opaque error; (2) for a
valid-but-oversized image, progressively downscales and re-encodes as
JPEG until it fits under the base64 limit, so the document can actually
still be processed instead of just failing more gracefully. Every
vision-agent graph node (`classify_node`, `kyc_node`, `claims_node` in
`backend/graph/pipeline.py`) now catches `ImageProcessingError` and
degrades to a low-confidence flagged result instead of crashing the graph.

**A second, subtler bug found while wiring the fallback in, fixed before
it ever shipped:** the naive version of this (KYC returns
`kyc_passed=False`, Claims returns `schema_valid=False`) would have fallen
straight into `compute_decision`'s REJECT branch, and
`requires_human_review` only pauses a REJECT when `fraud_score` is already
elevated — so a corrupted upload on an otherwise-low-fraud-score case
would have silently auto-REJECTed a real claim with **no human ever
reviewing it**, exactly the "excessive agency" failure mode SS7 (security
module) exists to prevent. Fixed by adding an explicit
`IMAGE_PROCESSING_ERROR_FLAG` check in `compute_decision`, checked *before*
the REJECT block, that forces `ESCALATE` unconditionally whenever KYC or
Claims couldn't actually evaluate the document — the same "a case we
can't evaluate needs a human, regardless of what any other rule concludes"
principle as the `UNKNOWN`-doc-type rule.

Validated three ways, all zero real API cost: `encode_image` run directly
against the actual 3 problem files (confirmed `amend_PT_39451` now encodes
successfully as a 617KB JPEG instead of raising; the 2 corrupted files
raise a clean `ImageProcessingError` instead of an opaque API failure), 5
new unit tests in `backend/tests/test_vision_utils.py`, 2 new
`compute_decision` tests proving the REJECT-vs-ESCALATE gap above is
closed, and 3 new `backend/tests/test_pipeline.py` graph-level tests
proving `graph.invoke` completes cleanly end-to-end on all three node
types instead of raising (the actual regression the bug was about). 193
tests passing (up from 190).

## 8. MCP Server (task #11/#13)

### 8.1 `mcp>=1.1.2` resolved to 2.0.0 — a fully-restructured, fully-async API, and three real bugs found wiring it up

**What happened:** `mcp` was declared in `pyproject.toml` (`mcp>=1.1.2`)
but had never actually been installed — `uv sync` fixed that, and
resolved to **`mcp` 2.0.0**, a much newer major version than the pinned
floor. The commonly-documented `mcp.server.fastmcp.FastMCP` class doesn't
exist in this version; the equivalent is `mcp.server.mcpserver.MCPServer`
(same `.tool()` decorator shape, different module path, different
internal attribute name for the low-level server —
`_lowlevel_server`, not `_mcp_server`, discovered by reading
`run_stdio_async`'s own source rather than guessing).

Three more real, concrete bugs surfaced building and wiring the server
(`backend/mcp_server/`), all found by writing and running real
protocol-level tests (in-process MCP client/server round-trips —
zero Anthropic cost, MCP tool discovery/calling never touches an LLM)
before trusting any of this:

1. **`CallToolResult.isError` doesn't exist in mcp 2.0 — it's `is_error`**
   (snake_case). Found immediately by a real test failure
   (`AttributeError`), not by reading changelogs.
2. **Returning a value out of an `anyio` task group you're in the middle
   of cancelling is unreliable.** `backend/mcp_server/client_tools.py`'s
   first draft called `tg.cancel_scope.cancel()` then tried to `return
   result.content[0].text` from inside the (now-cancelling) scope — this
   silently produced `None` instead of the real result, no exception, no
   warning. Fixed by capturing the result into a variable *before*
   cancelling, and returning it only after both `async with` blocks have
   fully unwound. A real, non-obvious concurrency gotcha, not a typo.
3. **The Supply Chain security guard (`backend/security/tool_scanning.py`,
   see SS§7 category 3) was silently checking an empty string for every
   MCP-transport tool.** `StructuredTool.from_function(func=X,
   description=Y)` — what the MCP client adapter uses, since the wrapped
   function has no docstring of its own — sets `.description` directly;
   the original scanner only checked `.func.__doc__`, which was empty for
   these tools. `is_tool_safe()` returned `True` for every MCP tool
   regardless of its actual description text — a real security-control
   gap, not just a missing test. Fixed by checking `tool.description`
   (what's actually sent to Claude) in addition to the unwrapped
   function's docstring.

**Resolution:** `backend/mcp_server/server.py` (the real MCP server,
wrapping the *same* underlying callables `retrieve_policy_clauses`/
`lookup_claim_history` already use via `.func` — single source of truth,
nothing duplicated) and `backend/mcp_server/client_tools.py` (the
LangChain-compatible adapter, fresh in-process connection per call rather
than a persistent background session — simpler to get right without live
iteration, and the reconnect overhead is negligible since there's no real
network involved). Both `backend/agents/policy_rag.py` and
`backend/agents/fraud_detection.py` now bind their tools through this MCP
transport by default (PROJECT_PLAN.md §5's original intent), validated
with real end-to-end `create_agent` calls before switching the default —
not just protocol-level tests — and a full-pipeline smoke test
(`claim_PT_39451`) confirming the swap doesn't change real behavior:
identical decision, identical fraud score, identical coverage result as
the pre-swap baseline. 12 new tests, all passing, full suite 169/169 at
the time of this entry.

**Takeaway:** a fast-moving dependency (a `>=` floor resolving to a major
version bump nobody explicitly chose) is exactly the situation where
"read the docs and assume" fails quietly — every one of these three bugs
was found by writing a real test and watching it fail, not by reasoning
about the API from memory. The security-guard gap (#3) is the sharpest
lesson: a control that "doesn't crash" isn't the same as a control that
works, and the only way to know the difference is to check what it's
actually looking at.

## 9. FastAPI Ingestion API (task #12)

### 9.1 A real live-server smoke test caught a bug the mocked tests couldn't

**What happened:** `backend/api/app.py` was built with a full mocked test
suite (12 tests, `backend/tests/test_api.py`, zero API cost via httpx's
`ASGITransport` — no real network, no real server process). All 12 passed
first try. But a genuine live smoke test — actually starting `uvicorn` and
posting a real file upload with `curl` — showed `patient_id` silently
coming back `null` even though it was submitted correctly as multipart
form data.

**Why:** `upload_case`'s endpoint signature declared `patient_id: str |
None = None` as a bare parameter alongside a `File(...)` parameter. For a
multipart request, FastAPI requires every non-file field to be explicitly
declared `Form(...)` — an undecorated parameter is treated as a *query*
parameter instead, so a value posted as a form field is silently ignored
rather than erroring. The mocked test suite never caught this because
`ASGITransport`'s in-process request construction (via `httpx`'s own
multipart encoder, called the same way in the test as in the real curl
request) *should* have hit the identical code path — and did; the gap
wasn't the test transport, it was that the original test asserted the
upload endpoint's *immediate response* (202, case_id present) and never
asserted the resulting `patient_id` actually landed on the persisted case
state. Fixed by adding exactly that assertion
(`test_upload_captures_patient_id_and_policy_number_from_form_data`) once
the bug was known, but it wasn't written *before* the bug was found —
worth noting honestly rather than implying the test suite would have
caught it on its own.

**Fix:** `Annotated[str | None, Form()] = None` for both `patient_id` and
`policy_number`.

**A second, non-code lesson from the same debugging session:** re-verifying
the fix required restarting the live server, and `pkill -f "uvicorn
backend.api.app"` silently failed to kill the running Windows process
(spawned via `uv run`, whose process tree/name matching doesn't line up
with what `pkill -f` expects on this platform) — the "restarted" server
actually failed to bind (port already in use) and exited immediately,
while the *original, unfixed* process kept serving requests underneath,
making the fix look like it hadn't worked. Diagnosed by checking
`netstat -ano` for the actual PID holding the port and killing it directly
with `taskkill //F //PID <pid>`. Worth remembering for this environment
specifically: don't trust `pkill`/similar Unix process-matching tools
against `uv run`-spawned Windows processes — verify by port ownership
instead.

**Takeaway:** a fully-mocked test suite proves the code *handles* the
inputs it's given correctly; it doesn't prove the *inputs it's given* are
what a real client would actually send. The gap here (query param vs. form
field) is a classic FastAPI multipart footgun that only a real multipart
request — real `curl`/`httpx` wire encoding, not a hand-constructed test
payload — reliably exercises. Same principle as §7's staged eval runs and
§8's real MCP protocol tests: real I/O finds a different class of bug
than logic tests do, and neither replaces the other.

## How this list will grow

Remaining tasks (Fraud Detection's MCP tool integration, the LangGraph
state machine wiring with parallel branches, HITL interrupts, the
security module's adversarial tests, the FastAPI/Next.js layers) are all
likely to surface their own version of "the docs say X, the real
behavior is Y" — that's normal for a project pulling together this many
moving pieces (LangGraph, LangChain 1.x, Docling, Chroma, MCP, FastAPI,
Next.js) for the first time in combination. Each new one gets added here
as it's found, not batched up at the end.
