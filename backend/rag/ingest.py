"""Docling ingestion of the MediShield policy PDFs into the shared Chroma
collection (Component 6: Policy Agent / RAG). Run once (or whenever a policy
PDF changes) via `backend/scripts/ingest_policies.py` — retrieval reads from
the persisted collection, it doesn't re-ingest on every query."""

import threading
from pathlib import Path
from typing import cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from docling.chunking import DocMeta, HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from backend.config import Settings, get_settings

COLLECTION_NAME = "medishield_policies"

DEFAULT_POLICY_PDFS = {
    "gold": "dataset/policies/medishield_gold_plan.pdf",
    "silver": "dataset/policies/medishield_silver_plan.pdf",
}

# create_agent's tool node runs parallel tool calls on a thread pool; two
# threads constructing a PersistentClient against the same path at once
# race on tenant/database bootstrap ("Could not connect to tenant
# default_tenant"). One client per process, built once, avoids that.
_client: ClientAPI | None = None
_client_lock = threading.Lock()


def _get_client(settings: Settings) -> ClientAPI:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = chromadb.PersistentClient(path=str(settings.resolved_path(settings.chroma_persist_dir)))
    return _client


def _converter() -> DocumentConverter:
    # These policy PDFs are real vector-text PDFs (verified with pdftotext),
    # so OCR is unnecessary; table structure detection stays on since the
    # Schedule of Benefits section is table-heavy.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})


def get_collection(settings: Settings | None = None) -> Collection:
    settings = settings or get_settings()
    client = _get_client(settings)
    # Cosine distance gives a bounded [0, 2] range we can turn into an
    # interpretable relevance score; the HNSW default (squared L2) doesn't.
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def ingest_policy_pdf(pdf_path: str, plan_name: str, settings: Settings | None = None) -> int:
    """Convert, chunk, and upsert one policy PDF into Chroma under
    metadata `plan=plan_name`. Idempotent: prior chunks for this plan are
    deleted before the new ones are added, so re-running doesn't duplicate."""
    settings = settings or get_settings()
    path = Path(pdf_path)
    if not path.is_absolute():
        path = settings.resolved_path(pdf_path)

    result = _converter().convert(str(path))
    chunker = HybridChunker()
    chunks = list(chunker.chunk(result.document))

    collection = get_collection(settings)
    collection.delete(where={"plan": plan_name})

    collection.add(
        ids=[f"{plan_name}::{i}" for i in range(len(chunks))],
        documents=[chunker.contextualize(c) for c in chunks],
        metadatas=[
            {
                "plan": plan_name,
                "headings": " > ".join(cast(DocMeta, c.meta).headings or []),
                "chunk_index": i,
            }
            for i, c in enumerate(chunks)
        ],
    )
    return len(chunks)


def ingest_all_policies(
    plan_pdfs: dict[str, str] | None = None, settings: Settings | None = None
) -> dict[str, int]:
    settings = settings or get_settings()
    plan_pdfs = plan_pdfs or DEFAULT_POLICY_PDFS
    return {name: ingest_policy_pdf(path, name, settings) for name, path in plan_pdfs.items()}
