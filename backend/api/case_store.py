"""Reuses the LangGraph checkpointer (backend/graph/checkpointer.py) as the
case store — no separate database. A case's full state (every agent
result, decision, human_review) is already exactly what's persisted per
thread_id by the pipeline itself; this module only adds read/list access
shaped for the API. Same single-source-of-truth principle already used
throughout this project (the eval harness and every transcripts/ file read
case state this same way, via `graph.get_state`)."""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph


def get_case_snapshot(graph: CompiledStateGraph, case_id: str) -> dict[str, Any] | None:
    """Returns `{"values": <CaseState dict>, "pending_review": <interrupt
    payload dict, or None if not paused>}`, or None if this case_id has no
    checkpoint at all (never uploaded)."""
    config = {"configurable": {"thread_id": case_id}}
    snapshot = graph.get_state(config)
    if not snapshot.values:
        return None

    pending_review = None
    if snapshot.tasks and snapshot.tasks[0].interrupts:
        pending_review = snapshot.tasks[0].interrupts[0].value

    return {"values": snapshot.values, "pending_review": pending_review}


def list_cases(checkpointer: BaseCheckpointSaver, limit: int = 100) -> list[dict[str, Any]]:
    """Latest known state per case (deduped across each case's many
    intermediate checkpoints — one per graph step), newest first."""
    latest_by_thread: dict[str, tuple[str, dict[str, Any]]] = {}
    for tup in checkpointer.list(None):
        thread_id = tup.config["configurable"]["thread_id"]
        ts = tup.checkpoint.get("ts", "")
        if thread_id not in latest_by_thread or ts > latest_by_thread[thread_id][0]:
            latest_by_thread[thread_id] = (ts, tup.checkpoint.get("channel_values", {}))

    cases = [
        {
            "case_id": thread_id,
            # `status`/`decision` deserialize back to the real enum objects
            # (CaseStatus/Decision) via the checkpointer's own type
            # registry — `.value` extracts the plain string for the JSON
            # response.
            "status": values["status"].value if values.get("status") else None,
            "updated_at": ts,
            "decision": values.get("decision").decision.value if values.get("decision") else None,
        }
        for thread_id, (ts, values) in latest_by_thread.items()
    ]
    cases.sort(key=lambda c: c["updated_at"], reverse=True)
    return cases[:limit]
