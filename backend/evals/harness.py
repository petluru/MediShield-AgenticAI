"""Runs the compiled case graph against a list of `EvalCase`s and collects
`CaseResult`s for scoring.py. Takes an already-compiled graph rather than
building one itself, so tests can inject a graph wired to fully mocked
agent functions (matching backend/tests/test_pipeline.py's pattern) — this
module makes zero Anthropic calls itself; every real call happens inside
the agent functions the graph was built with.

Decision correctness is measured against the pipeline's OWN computed
decision (`result["decision"].decision`), not a post-human-review outcome —
`orchestrator_node` sets both `decision` and `status` in the same return
even when `status` ends up `AWAITING_REVIEW`, so a paused case's decision is
already available without resuming it. This is deliberate: the eval scores
the AI pipeline's correctness, not a human reviewer's later override
choice — resuming every ESCALATE/REJECT-with-fraud case just to read a
decision that's already sitting in the state would also mean fabricating a
reviewer response for every paused case, which isn't a real evaluation of
anything."""

from langgraph.graph.state import CompiledStateGraph

from backend.evals.ground_truth import EvalCase
from backend.evals.scoring import REQUIRED_CLAIM_FIELDS, CaseResult
from backend.models import CaseState


def _extracted_field_presence(claims_result: object) -> dict[str, bool] | None:
    if claims_result is None:
        return None
    fields = claims_result.extracted_fields  # type: ignore[attr-defined]
    presence = {}
    for name in REQUIRED_CLAIM_FIELDS:
        value = getattr(fields, name, None)
        presence[name] = bool(value)  # non-empty list, non-None scalar, non-empty string
    return presence


def _case_result_from_state(case: EvalCase, state: dict) -> CaseResult:
    classifier_result = state.get("classifier_result")
    decision_result = state.get("decision")
    claims_result = state.get("claims_result")

    return CaseResult(
        doc_id=case.doc_id,
        category=case.category,
        expected_doc_type=case.expected_doc_type,
        actual_doc_type=classifier_result.doc_type if classifier_result else None,
        expected_decision=case.expected_decision,
        actual_decision=decision_result.decision if decision_result else None,
        extracted_field_presence=_extracted_field_presence(claims_result) if case.category == "claim_forms" else None,
    )


def _thread_config(case: EvalCase) -> dict:
    return {"configurable": {"thread_id": f"eval-{case.doc_id}"}}


def run_case(graph: CompiledStateGraph, case: EvalCase) -> CaseResult:
    state = CaseState(
        case_id=case.doc_id,
        file_path=case.file_path,
        content_type="image/png",
        patient_id=case.patient_id,
        policy_number=case.policy_number,
    )
    config = _thread_config(case)

    try:
        result = graph.invoke(state, config=config)
    except Exception as exc:  # noqa: BLE001 — an eval run must never crash on one bad case
        return CaseResult(
            doc_id=case.doc_id,
            category=case.category,
            expected_doc_type=case.expected_doc_type,
            expected_decision=case.expected_decision,
            error=f"{type(exc).__name__}: {exc}",
        )

    return _case_result_from_state(case, result)


def run_eval_suite(graph: CompiledStateGraph, cases: list[EvalCase]) -> list[CaseResult]:
    return [run_case(graph, case) for case in cases]


def read_case_result_from_checkpoint(graph: CompiledStateGraph, case: EvalCase) -> CaseResult:
    """Read-only — reconstructs a CaseResult from a case's already-persisted
    checkpoint state (`graph.get_state`), no `invoke()` call and therefore
    no Anthropic spend. Used to combine multiple already-run eval stages
    (e.g. separate `--category` runs) into one final report without paying
    to re-run anything that already completed."""
    config = _thread_config(case)
    state = graph.get_state(config).values
    if not state:
        return CaseResult(
            doc_id=case.doc_id,
            category=case.category,
            expected_doc_type=case.expected_doc_type,
            expected_decision=case.expected_decision,
            error="no checkpoint found for this case — it was never run",
        )
    return _case_result_from_state(case, state)
