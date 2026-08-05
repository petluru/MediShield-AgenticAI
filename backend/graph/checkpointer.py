"""Default checkpointer for the case graph — required for `interrupt()`
(PROJECT_PLAN.md SS6's human-in-the-loop gate) to work at all. One
process-global `SqliteSaver` per `checkpoint_db_path`, same pattern as
`llm_factory._ensure_llm_cache`: built from a raw `sqlite3.Connection`
rather than `SqliteSaver.from_conn_string`, since that classmethod is a
`@contextmanager` generator meant for a single `with` block, not a
long-lived saver reused across many graph invocations."""

import sqlite3
import threading

from langgraph.checkpoint.sqlite import SqliteSaver

from backend.config import Settings

_checkpointer: SqliteSaver | None = None
_checkpointer_lock = threading.Lock()


def default_checkpointer(settings: Settings) -> SqliteSaver:
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    with _checkpointer_lock:
        if _checkpointer is not None:
            return _checkpointer
        db_path = settings.resolved_path(settings.checkpoint_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        _checkpointer = SqliteSaver(conn)
        return _checkpointer
