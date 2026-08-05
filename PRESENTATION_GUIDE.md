# Presentation Guide — Presenting MediShield Live, In Person

A script for demoing this project from your own laptop, in front of
people, with no rehearsal needed beyond reading this once. Pick the
timing variant that matches your slot; the sections build on each other,
so a shorter talk just stops earlier.

---

## Before you walk in the room

**The night before / an hour before, not while people are watching:**

1. Confirm `.env` has a real `ANTHROPIC_API_KEY` and check your remaining
   budget — `uv run python -m backend.scripts.token_usage_report` shows
   real spend to date. **Decide up front whether you're doing a live
   upload or not** (see "If you don't want to spend API budget live"
   below) — don't decide this on stage.
2. Start both servers and leave them running:
   ```bash
   uv run uvicorn backend.api.app:app --host 127.0.0.1 --port 8000
   ```
   ```bash
   cd frontend && npm run dev
   ```
3. Open `http://localhost:3000` and confirm the dashboard loads real
   cases. If it's empty, something's wrong — fix it now, not live.
4. **Open these tabs/windows in advance**, in this order, so alt-tabbing
   during the talk is muscle memory, not a search:
   - Tab 1: the dashboard (`localhost:3000`)
   - Tab 2: a terminal with the test suite ready to run (don't run it yet)
   - Tab 3: `EVAL_REPORT.md` open in your editor
   - Tab 4: `transcripts/adversarial_security_transcripts.md` open in your editor
   - Tab 5 (optional, deep-dive only): `backend/agents/orchestrator.py`
5. Pick your **one hero case** for the live walkthrough. Recommended:
   `claim_PT_99733` — it exercises every agent, lands on ESCALATE for a
   genuinely interesting reason (a clinically implausible procedure
   bundle, not a boring missing-field rejection), and is already sitting
   in the dashboard from the real eval run. Know its case ID before you
   start; don't hunt for it live.
6. Close anything else that might notification-pop mid-demo.

**If you don't want to spend API budget live:** don't upload a new
document. Everything in this guide's demo script also works by clicking
into *already-processed* cases sitting in the dashboard from the eval
run — real data, zero new cost. Only the "upload a fresh document" beat
(§4 below) costs anything, and it's clearly marked optional.

---

## The 2-minute opener (say this first, before touching the laptop)

> "This is MediShield — a document intake pipeline for a health insurer.
> A claim, an ID, a discharge summary, whatever comes in gets looked at
> by six specialist AI agents, each doing one narrow job, and a
> deterministic rule — not another AI guess — turns their findings into
> approve, reject, or escalate to a human. The interesting part isn't
> that it uses AI; it's *where* it deliberately doesn't: every decision
> that needs to be auditable is plain code, not a model call. I'll show
> you a real case go through the whole pipeline, then how it handles
> someone trying to prompt-inject it, then what happens when it's
> genuinely unsure."

---

## 1. The architecture, at a glance (2–3 min)

Open `README.md`'s architecture diagrams (or draw the pipeline flow on a
whiteboard if you don't want to show markdown). Say the six agents out
loud in order and *why* each is separate — you don't need notes for this
if you've internalized the table in `README.md`'s "Why multi-agent"
section: Classifier gates everything, KYC/Claims/Policy only run for the
document types that need them, everything converges on Fraud Detection,
and the Orchestrator is the one place it all comes together.

**The one sentence to land here:** *"The Orchestrator's actual
APPROVE/REJECT/ESCALATE decision is deterministic Python, not an LLM
call — I'll show you exactly why in a minute."*

## 2. Live walkthrough of one real case (5–7 min) — the core of the demo

1. On the dashboard, find your hero case (`claim_PT_99733`) and click
   **View**.
2. Walk the page top to bottom, narrating each panel:
   - **Classifier**: "99% confident this is a CMS-1500 claim form."
   - **Claims**: "Extracted a schema-valid claim — $16,825, four CPT
     codes, no validation errors."
   - **Policy RAG**: "Covered at 80%, but notice the confidence is only
     72% — it had to map four different benefit categories to one
     service date. That's the RAG agent being honest about ambiguity,
     not just returning a number."
   - **Fraud Detection**: "This is where it gets interesting. Score
     0.35, MEDIUM risk. It didn't flag a duplicate or a frequency
     problem — it flagged that a total knee replacement, psychotherapy,
     and critical care were all billed on the *same day*, which doesn't
     make clinical sense together. That's a judgment call, not a
     lookup."
   - **Final Decision**: "ESCALATE — read the justification out loud, or
     let people read it themselves for a few seconds. Point out it
     explains *why* confidence isn't higher: the evidence is a
     possibility, not a confirmed problem."
3. Point at the **review panel** at the top: "This is paused, waiting
   for a human. I can confirm the computed decision, or override it —
   either way, that action gets recorded, never silently dropped." —
   click **Override**, pick a decision, type a one-line note, and submit
   it live. Refresh and show the "Human Review" panel now populated.

**This one case alone demonstrates:** multi-agent orchestration, a real
tool-calling RAG loop, fraud reasoning that isn't a simple lookup, and
the human-in-the-loop gate — the whole system in one example.

## 3. Security, without spending anything (3–4 min)

Switch to `transcripts/adversarial_security_transcripts.md` (already
open). You don't need to re-run anything live — read from it directly:

- Show **Test 1** (prompt injection paired with a valid ID): the
  document image contained text trying to get the model to say the ID
  passes regardless of what it sees. The model still returned a correct
  result and the flag is visible in the transcript. Say: *"The document
  itself is untrusted input — every system prompt in this codebase says
  so explicitly, and this transcript proves it holds under a real
  model call, not just in theory."*
- Mention the other four categories are enforced as **deterministic
  Python guards** (redaction, tool-docstring scanning, output escaping,
  the human-review gate) — point out this was a deliberate choice: code
  you can point to and reason about, not "trust the prompt."

**If asked "did you find any real security gaps in your own code":**
yes — tell the tool-docstring-scanning story from `WALKTHROUGH.md` §12
(the scanner was silently checking an empty string for MCP-wrapped
tools until fixed). Good, honest answer; don't dodge it.

## 4. Optional: a live upload (2–3 min, costs real API money)

Only do this if you decided in advance to spend the budget. Go to the
Upload page, pick any file from `dataset/id_documents/` or
`dataset/claim_forms/`, submit it, and narrate while it processes:
*"This is running for real right now — vision calls to Claude,
classifying, extracting, checking policy and fraud, all live."* Refresh
the dashboard after ~15–30 seconds and click into the new case. This is
the single most convincing "it's not a demo, it's real" beat — use it if
you can afford the ~$0.01–0.05 it costs for one case, skip it if budget
is tight (§2 already proved everything works with zero new spend).

## 5. The engineering discipline behind it (3–5 min, for a technical audience)

This is the section to expand or cut depending on who's in the room —
skip entirely for a non-technical audience, spend the most time here for
engineers.

- **Cost control was a first-class concern, not an afterthought.** Open
  `TOKEN_BUDGET.md`'s bottom line: *"A single runaway tool-calling loop
  once cost $16 — 99.5% of everything spent on this project at that
  point — because the RAG agent's own prompt told it to keep re-querying
  when results looked weak, with nothing telling it when to stop."* Show
  the fix in `backend/agents/policy_rag.py`: a hard `recursion_limit`
  plus a graceful fallback. This is a great "real bug, real fix, real
  measured improvement" story.
- **The escalation-without-doubling-work optimization**
  (`WALKTHROUGH.md` §9): Fraud Detection reuses a deterministic tool
  result across a Sonnet→Opus escalation instead of re-running the whole
  loop — a real, measured cost reduction, and a good example of
  "know exactly which part of your pipeline is deterministic and which
  part needs independent judgment."
- **Run the test suite live** if you have a terminal ready: `uv run
  pytest backend/tests/ -q` — 193 tests, all mocked, finishes in under a
  minute, costs nothing. Say: *"This validates the code's logic — did we
  route correctly, did we handle a corrupted image correctly — separate
  from whether the model itself is good at its job, which is what the
  eval suite measures instead."*
- **Show `EVAL_REPORT.md`'s honesty, not just its score.** 78.1%
  Decision Correctness against a 60% bar is the headline, but the more
  interesting thing to point at is the "Not Auto-Scored" table — three
  of the assignment's six weighted criteria are reported as *not*
  measured by this harness, with a stated reason, rather than a
  fabricated number. That's a stronger signal of engineering judgment
  than a single score.

## 6. Closing (1 min)

> "Everything I showed is real — real API calls, real evaluation data,
> real bugs that got found and fixed, documented honestly including the
> ones I chose not to fix and why. The code, the docs, and every
> transcript are all in the repo."

---

## Timing variants

- **5-minute lightning talk:** §Opener + §2 only (walk the hero case).
  Skip everything else.
- **15-minute standard demo:** §Opener + §1 + §2 + §3. Skip §4 and §5
  unless someone asks.
- **30-minute deep dive / technical audience:** all six sections, in
  order.

---

## Anticipated questions, and honest answers

**"Why isn't the eval score higher?"** 78.1% clears the 60% bar, and the
34 mismatches out of 155 cases break down into distinct, understood
causes — not one systemic failure. Some are ground-truth methodology
choices (documented in `backend/evals/ground_truth.py`), one is a
deliberate trade-off (KYC's conservative tamper detection), and the rest
were investigated and closed. Point to `IMPLEMENTATION_CHALLENGES.md
§7.3`/`§7.4` — the honest breakdown is more defensible than a higher
number with no explanation behind it.

**"How much did this cost to build?"** Real, measured total is in
`TOKEN_BUDGET.md` — roughly $0.11 in steady-state calls, plus one
$16 runaway-loop incident that got fixed and is the reason the
recursion-limit pattern exists everywhere else in the codebase now. The
full 155-document eval run separately cost about $5.79. Total project
spend stayed under the ~$10–14 budget across the whole build.

**"What would you do differently / what's not done?"** Be direct:
WebSocket live status streaming wasn't built (it's an explicit bonus
challenge in the assignment, not required), Fraud Detection can't see
cross-document signals because its tool only exposes claim-form metadata
(a structural gap, documented, not hidden), and GitHub packaging (this
repo isn't `git init`'d yet as of this writing) is the last remaining
step.

**"Why deterministic decision logic instead of just asking the LLM?"**
Auditability and reproducibility. A business-critical decision that
could produce a different answer from the same inputs on a different day
is a liability in an insurance context specifically — you need to be
able to explain to a regulator or a customer exactly why a claim was
rejected, and "the model said so" isn't an answer you can stand behind.

**"How do you know the model isn't just getting lucky on this
dataset?"** The 155-document eval is the whole point of that concern —
it's not one cherry-picked case, it's the full synthetic dataset,
including edge cases specifically designed to be hard (duplicate claims,
tampered IDs, ambiguous coverage). The mismatches are documented, not
swept under the rug.

**"Is this production-ready?"** No, and the docs say so explicitly —
`README.md`'s "Known limitations" section lists exactly what's not
covered (WebSocket streaming, cross-document fraud signals) and what's a
deliberate trade-off versus an oversight. It's a capstone-scoped system
that demonstrates the architecture and engineering discipline a
production version would need, not a finished product.

---

## If something breaks live

- **Servers won't start / API key issue:** fall back to `EVAL_REPORT.md`
  and the `transcripts/` folder — every number in them is real, captured
  data, and none of it requires a live server. You can present the
  entire "what it found" story from static files if you have to.
- **Live upload hangs or errors:** don't troubleshoot on stage — say
  "let's not burn time on this live, here's a case that already ran" and
  pivot straight to §2's hero case.
- **Someone asks something you don't know:** it's fine to say "good
  question, that's not something I've dug into — let me check
  `IMPLEMENTATION_CHALLENGES.md` and get back to you" rather than
  guessing. The whole point of that document is that it's the honest
  record.
