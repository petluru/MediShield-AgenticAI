from unittest.mock import MagicMock, patch

from backend.rag.retrieval import _format_chunk, retrieve_policy_clauses


def test_format_chunk_converts_cosine_distance_to_relevance():
    text = _format_chunk("some clause text", {"plan": "gold", "headings": "4. Exclusions"}, distance=0.4)
    assert "relevance=0.80" in text
    assert "plan=gold" in text
    assert "section=4. Exclusions" in text
    assert "some clause text" in text


def test_format_chunk_clamps_relevance_at_zero_for_distant_matches():
    text = _format_chunk("x", {}, distance=3.0)
    assert "relevance=0.00" in text
    assert "section=(no heading)" in text
    assert "plan=unknown" in text


def test_format_chunk_redacts_sensitive_document_text():
    # PROJECT_PLAN.md SS7 category 2 — redaction applied at the
    # retrieval -> prompt-assembly boundary, per backend/security/redaction.py.
    text = _format_chunk("Member SSN: 123-45-6789 on file.", {"plan": "gold"}, distance=0.1)
    assert "123-45-6789" not in text
    assert "REDACTED" in text


@patch("backend.rag.retrieval.get_collection")
def test_retrieve_policy_clauses_returns_message_when_no_matches(mock_get_collection):
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    mock_get_collection.return_value = mock_collection

    result = retrieve_policy_clauses.invoke({"query": "anything"})

    assert result == "No matching policy clauses found."


@patch("backend.rag.retrieval.get_collection")
def test_retrieve_policy_clauses_filters_by_plan(mock_get_collection):
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["clause text"]],
        "metadatas": [[{"plan": "silver", "headings": "1. Schedule"}]],
        "distances": [[0.2]],
    }
    mock_get_collection.return_value = mock_collection

    result = retrieve_policy_clauses.invoke({"query": "deductible", "plan": "silver", "n_results": 3})

    mock_collection.query.assert_called_once_with(query_texts=["deductible"], n_results=3, where={"plan": "silver"})
    assert "clause text" in result
    assert "plan=silver" in result
