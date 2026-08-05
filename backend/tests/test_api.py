"""FastAPI endpoint tests. Fully mocked agents (matching
backend/tests/test_pipeline.py's pattern) + an isolated `InMemorySaver` —
zero Anthropic API cost. Upload tests only check the immediate response
(case_id, RECEIVED status) — background-processing correctness is already
covered by test_pipeline.py/test_eval_harness.py; GET/review tests
pre-populate case state via a direct `graph.invoke()` rather than relying
on racing the upload endpoint's background task."""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver
from unittest.mock import patch

from backend.api.app import create_app
from backend.config import Settings
from backend.graph.pipeline import build_case_graph
from backend.models import (
    CaseState,
    ClassifierOutput,
    Decision,
    DocType,
    FraudOutput,
    OrchestratorDecision,
    RiskLevel,
)

AUTH_HEADERS = {"Authorization": "Bearer test-token"}


@pytest.fixture
def mocked_pipeline():
    with (
        patch("backend.graph.pipeline.classify_document") as mock_classify,
        patch("backend.graph.pipeline.verify_kyc") as mock_kyc,
        patch("backend.graph.pipeline.extract_claim") as mock_claims,
        patch("backend.graph.pipeline.determine_coverage") as mock_policy,
        patch("backend.graph.pipeline.assess_fraud_risk") as mock_fraud,
        patch("backend.graph.pipeline.decide") as mock_decide,
    ):
        yield {
            "classify": mock_classify,
            "kyc": mock_kyc,
            "claims": mock_claims,
            "policy": mock_policy,
            "fraud": mock_fraud,
            "decide": mock_decide,
        }


def _set_clean_defaults(mocked_pipeline, decision=Decision.APPROVE, confidence=0.9, fraud_score=0.05):
    mocked_pipeline["classify"].return_value = ClassifierOutput(doc_type=DocType.PRESCRIPTION, confidence=0.9)
    mocked_pipeline["fraud"].return_value = FraudOutput(
        fraud_score=fraud_score, anomalies=[], risk_level=RiskLevel.LOW if fraud_score < 0.3 else RiskLevel.MEDIUM
    )
    mocked_pipeline["decide"].return_value = OrchestratorDecision(
        decision=decision, confidence=confidence, justification="test case"
    )


def build_app(tmp_path):
    settings = Settings(
        _env_file=None,
        ANTHROPIC_API_KEY="sk-ant-test",
        API_AUTH_TOKENS="test-token",
        UPLOAD_DIR=str(tmp_path / "uploads"),
    )
    checkpointer = InMemorySaver()
    graph = build_case_graph(settings=settings, checkpointer=checkpointer)
    app = create_app(settings=settings, graph=graph, checkpointer=checkpointer)
    return app, graph


async def test_health_check_no_auth_required(mocked_pipeline, tmp_path):
    app, _ = build_app(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


async def test_upload_without_auth_returns_401(mocked_pipeline, tmp_path):
    app, _ = build_app(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("test.png", b"fake-png-bytes", "image/png")}
        r = await client.post("/cases", files=files)
        assert r.status_code == 401


async def test_upload_rejects_unsupported_content_type(mocked_pipeline, tmp_path):
    app, _ = build_app(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        r = await client.post("/cases", files=files, headers=AUTH_HEADERS)
        assert r.status_code == 400


async def test_upload_accepts_valid_image_and_returns_case_id(mocked_pipeline, tmp_path):
    _set_clean_defaults(mocked_pipeline)
    app, _ = build_app(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("test.png", b"fake-png-bytes", "image/png")}
        r = await client.post("/cases", files=files, headers=AUTH_HEADERS)
        assert r.status_code == 202
        body = r.json()
        assert body["case_id"]
        assert body["status"] == "RECEIVED"
        await asyncio.sleep(0.3)  # let the background task finish


async def test_upload_captures_patient_id_and_policy_number_from_form_data(mocked_pipeline, tmp_path):
    # Real bug found by a live-server smoke test: patient_id/policy_number
    # must be declared as Form() fields, not bare params, or a multipart
    # request silently drops them to None.
    _set_clean_defaults(mocked_pipeline)
    app, graph = build_app(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("test.png", b"fake-png-bytes", "image/png")}
        data = {"patient_id": "PT_TEST_1", "policy_number": "MED-GLD-1"}
        r = await client.post("/cases", files=files, data=data, headers=AUTH_HEADERS)
        assert r.status_code == 202
        case_id = r.json()["case_id"]
        await asyncio.sleep(0.3)

    snapshot = graph.get_state({"configurable": {"thread_id": case_id}}).values
    assert snapshot["patient_id"] == "PT_TEST_1"
    assert snapshot["policy_number"] == "MED-GLD-1"


async def test_get_case_not_found_returns_404(mocked_pipeline, tmp_path):
    app, _ = build_app(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/cases/does-not-exist", headers=AUTH_HEADERS)
        assert r.status_code == 404


async def test_get_case_returns_full_case_detail(mocked_pipeline, tmp_path):
    _set_clean_defaults(mocked_pipeline)
    app, graph = build_app(tmp_path)
    case = CaseState(case_id="case-1", file_path="x.png", content_type="image/png")
    graph.invoke(case, config={"configurable": {"thread_id": "case-1"}})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/cases/case-1", headers=AUTH_HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["case"]["case_id"] == "case-1"
        assert body["case"]["status"] == "DECIDED"
        assert body["case"]["decision"]["decision"] == "APPROVE"
        assert body["pending_review"] is None


async def test_list_cases_returns_known_cases(mocked_pipeline, tmp_path):
    _set_clean_defaults(mocked_pipeline)
    app, graph = build_app(tmp_path)
    for cid in ("case-a", "case-b"):
        case = CaseState(case_id=cid, file_path="x.png", content_type="image/png")
        graph.invoke(case, config={"configurable": {"thread_id": cid}})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/cases", headers=AUTH_HEADERS)
        assert r.status_code == 200
        ids = {c["case_id"] for c in r.json()}
        assert {"case-a", "case-b"} <= ids


async def test_review_case_not_awaiting_review_returns_409(mocked_pipeline, tmp_path):
    _set_clean_defaults(mocked_pipeline)  # APPROVE, never pauses
    app, graph = build_app(tmp_path)
    case = CaseState(case_id="case-2", file_path="x.png", content_type="image/png")
    graph.invoke(case, config={"configurable": {"thread_id": "case-2"}})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/cases/case-2/review", json={"outcome": "APPROVED"}, headers=AUTH_HEADERS)
        assert r.status_code == 409


async def test_review_case_confirms_escalate_decision(mocked_pipeline, tmp_path):
    _set_clean_defaults(mocked_pipeline, decision=Decision.ESCALATE, confidence=0.5, fraud_score=0.4)
    app, graph = build_app(tmp_path)
    case = CaseState(case_id="case-3", file_path="x.png", content_type="image/png")
    graph.invoke(case, config={"configurable": {"thread_id": "case-3"}})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/cases/case-3/review", json={"outcome": "APPROVED", "notes": "confirmed"}, headers=AUTH_HEADERS
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "DECIDED"
        assert body["decision"] == "ESCALATE"


async def test_review_case_override_changes_decision(mocked_pipeline, tmp_path):
    _set_clean_defaults(mocked_pipeline, decision=Decision.ESCALATE, confidence=0.5, fraud_score=0.4)
    app, graph = build_app(tmp_path)
    case = CaseState(case_id="case-4", file_path="x.png", content_type="image/png")
    graph.invoke(case, config={"configurable": {"thread_id": "case-4"}})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/cases/case-4/review",
            json={"outcome": "OVERRIDDEN", "overridden_decision": "APPROVE", "notes": "reviewed"},
            headers=AUTH_HEADERS,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["decision"] == "APPROVE"


async def test_review_case_not_found_returns_404(mocked_pipeline, tmp_path):
    app, _ = build_app(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/cases/does-not-exist/review", json={"outcome": "APPROVED"}, headers=AUTH_HEADERS)
        assert r.status_code == 404
