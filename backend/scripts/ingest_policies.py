"""One-time (or re-run-on-change) ingestion of the MediShield policy PDFs
into the persistent Chroma collection used by the Policy RAG Agent.

Usage:
    uv run python -m backend.scripts.ingest_policies
"""

from backend.rag.ingest import ingest_all_policies


def main() -> None:
    counts = ingest_all_policies()
    for plan, n_chunks in counts.items():
        print(f"{plan}: {n_chunks} chunks ingested")


if __name__ == "__main__":
    main()
