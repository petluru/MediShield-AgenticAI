"""Manual smoke test for the full LangGraph pipeline (task #8) against real
dataset cases — the first end-to-end run through everything built so far,
including the human-in-the-loop pause/resume gate (PROJECT_PLAN.md SS6).
Not part of the pytest suite (hits the live Anthropic API, several calls
per case, costs money).

If a case pauses for human review (ESCALATE, or REJECT with an elevated
fraud score), this script auto-approves it to keep the demo non-interactive
— the actual Case Management UI (task #14/#15) is where a real reviewer
approves/overrides. Auto-approving here just resumes the graph, it doesn't
change what the pipeline itself does.

Usage:
    uv run python -m backend.scripts.smoke_pipeline
"""

from langgraph.types import Command

from backend.config import get_settings
from backend.graph.pipeline import build_case_graph
from backend.models import CaseState, CaseStatus

# (file_path, patient_id, policy_number, note)
SAMPLES = [
    ("dataset/claim_forms/claim_PT_19116.png", "PT_19116", "MED-GLD-6770619", "clean claim form, expect APPROVE"),
    (
        "dataset/claim_forms/claim_PT_62350.png",
        "PT_62350",
        "MED-GLD-4567281",
        "proc/diag mismatch fraud case, expect REJECT or ESCALATE",
    ),
    ("dataset/id_documents/id_PT_19116.png", "PT_19116", "MED-GLD-6770619", "clean ID document, expect APPROVE"),
]


def main() -> None:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set — copy .env.example to .env and fill it in first.")

    graph = build_case_graph(settings=settings)

    for i, (file_path, patient_id, policy_number, note) in enumerate(SAMPLES):
        case = CaseState(
            case_id=f"smoke-{i}",
            file_path=file_path,
            content_type="image/png",
            patient_id=patient_id,
            policy_number=policy_number,
        )
        thread_config = {"configurable": {"thread_id": case.case_id}}
        result = graph.invoke(case, config=thread_config)

        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            print(
                f"[{file_path}] PAUSED for human review: decision={payload['decision']} "
                f"confidence={payload['confidence']:.2f} justification={payload['justification']}"
            )
            result = graph.invoke(
                Command(resume={"outcome": "APPROVED", "notes": "auto-approved by smoke script"}),
                config=thread_config,
            )
            print(f"       resumed: human_review={result['human_review']}")

        decision = result["decision"]
        print(
            f"[{file_path}]\n"
            f"       note={note}\n"
            f"       doc_type={result['classifier_result'].doc_type.value} status={result['status'].value}\n"
            f"       decision={decision.decision.value} confidence={decision.confidence:.2f} "
            f"escalated_to_opus={decision.escalated_to_opus}\n"
            f"       justification={decision.justification}\n"
            f"       agent_summaries={decision.agent_summaries}"
        )
        assert result["status"] == CaseStatus.DECIDED


if __name__ == "__main__":
    main()
