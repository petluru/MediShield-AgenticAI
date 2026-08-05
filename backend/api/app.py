"""FastAPI ingestion API (assignment brief's "Ingestion API" component;
PROJECT_PLAN.md §2 item 12): file upload, case management endpoints, and
the trigger point for the LangGraph pipeline built earlier.

No separate database — every endpoint reads/writes through the same
LangGraph checkpointer (backend/graph/checkpointer.py) the pipeline itself
uses (backend/api/case_store.py), so there's exactly one source of truth
for a case's status.

Pipeline processing runs in a background thread
(`asyncio.to_thread(graph.invoke, ...)`) rather than blocking the request —
the graph/agents are fully synchronous (`agent.invoke()`, not `ainvoke()`,
by design elsewhere in this codebase — see backend/mcp_server/client_tools.py's
docstring for why that boundary matters), and a claim_form case can take
10-30+ seconds across several real Anthropic calls, far too long to hold
an HTTP request open for. The client polls `GET /cases/{case_id}` (or a
future WebSocket, not built yet — PROJECT_PLAN.md §5's streaming updates
is explicitly a bonus feature per assignment_02_multimodal_ai.md, not a
passing-bar requirement) for status.

Auth: a simple bearer-token check against `Settings.api_auth_tokens_list`
(already declared, previously unused) — not full user/session management,
appropriately scoped for this project's actual requirement ("no hardcoded
credentials"), not invented complexity."""

import asyncio
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command

from backend.api.case_store import get_case_snapshot, list_cases
from backend.api.schemas import CaseDetail, CaseListItem, ReviewRequest, UploadResponse
from backend.config import Settings, get_settings
from backend.graph.checkpointer import default_checkpointer
from backend.graph.pipeline import build_case_graph
from backend.models import CaseState, CaseStatus
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

_ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/tiff"}


def create_app(
    settings: Settings | None = None,
    graph: CompiledStateGraph | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> FastAPI:
    """Factory, not a bare module-level `app = FastAPI()` — lets tests
    build an app against an isolated graph/checkpointer (e.g. an
    `InMemorySaver`-backed graph with mocked agents, matching
    backend/tests/test_pipeline.py's pattern) instead of the real one.
    `graph`/`checkpointer` should always be passed together in tests —
    they must share the same checkpointer instance to see consistent
    state, same requirement `backend/graph/pipeline.py:build_case_graph`
    already has internally."""
    settings = settings or get_settings()
    checkpointer = checkpointer or default_checkpointer(settings)
    graph = graph or build_case_graph(settings=settings, checkpointer=checkpointer)

    app = FastAPI(title="MediShield Document Intake API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
        token = (authorization or "").removeprefix("Bearer ").strip()
        if token not in settings.api_auth_tokens_list:
            raise HTTPException(401, "Missing or invalid bearer token")

    def _thread_config(case_id: str) -> dict:
        return {"configurable": {"thread_id": case_id}}

    async def _process_case_in_background(case: CaseState) -> None:
        await asyncio.to_thread(graph.invoke, case, config=_thread_config(case.case_id))

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/cases", response_model=UploadResponse, status_code=202, dependencies=[Depends(_require_auth)])
    async def upload_case(
        file: Annotated[UploadFile, File()],
        # Must be `Form(...)`, not a bare `str | None` default — a
        # multipart request (required for the file upload) makes FastAPI
        # treat unbannotated non-file params as query parameters, not form
        # fields, so a real value posted as multipart form data silently
        # came through as None. Found by a real end-to-end smoke test
        # against the live server, not by reading the request in isolation
        # — see IMPLEMENTATION_CHALLENGES.md.
        patient_id: Annotated[str | None, Form()] = None,
        policy_number: Annotated[str | None, Form()] = None,
    ) -> UploadResponse:
        if file.content_type not in _ALLOWED_CONTENT_TYPES:
            raise HTTPException(400, f"Unsupported content type: {file.content_type!r}")

        case_id = str(uuid.uuid4())
        upload_dir = settings.resolved_path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename or "").suffix or ".png"
        dest_path = upload_dir / f"{case_id}{suffix}"
        dest_path.write_bytes(await file.read())

        case = CaseState(
            case_id=case_id,
            file_path=str(dest_path),
            content_type=file.content_type,
            patient_id=patient_id,
            policy_number=policy_number,
        )
        asyncio.create_task(_process_case_in_background(case))
        return UploadResponse(case_id=case_id, status="RECEIVED")

    @app.get("/cases", response_model=list[CaseListItem], dependencies=[Depends(_require_auth)])
    async def get_cases(limit: int = 100) -> list[dict]:
        return list_cases(checkpointer, limit=limit)

    @app.get("/cases/{case_id}", response_model=CaseDetail, dependencies=[Depends(_require_auth)])
    async def get_case(case_id: str) -> dict:
        snapshot = get_case_snapshot(graph, case_id)
        if snapshot is None:
            raise HTTPException(404, f"Case {case_id!r} not found")
        return {"case": snapshot["values"], "pending_review": snapshot["pending_review"]}

    @app.post("/cases/{case_id}/review", dependencies=[Depends(_require_auth)])
    async def review_case(case_id: str, review: ReviewRequest) -> dict:
        snapshot = get_case_snapshot(graph, case_id)
        if snapshot is None:
            raise HTTPException(404, f"Case {case_id!r} not found")
        if snapshot["values"].get("status") != CaseStatus.AWAITING_REVIEW:
            raise HTTPException(409, f"Case {case_id!r} is not awaiting review")

        resume_payload: dict = {"outcome": review.outcome.value, "notes": review.notes}
        if review.overridden_decision is not None:
            resume_payload["overridden_decision"] = review.overridden_decision.value

        result = await asyncio.to_thread(
            graph.invoke, Command(resume=resume_payload), config=_thread_config(case_id)
        )
        return {
            "case_id": case_id,
            "status": result["status"].value,
            "decision": result["decision"].decision.value,
        }

    return app


app = create_app()
