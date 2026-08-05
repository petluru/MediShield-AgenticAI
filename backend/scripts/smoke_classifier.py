"""Manual smoke test for the Classifier Agent against real dataset images.
Not part of the pytest suite (hits the live Anthropic API, costs money).

Usage:
    uv run python -m backend.scripts.smoke_classifier
"""

from backend.agents.classifier import classify_document
from backend.config import get_settings
from backend.models import DocType

SAMPLES: list[tuple[str, DocType]] = [
    ("dataset/claim_forms/claim_PT_15075.png", DocType.CLAIM_FORM),
    ("dataset/id_documents/id_PT_15075.png", DocType.ID_DOCUMENT),
    ("dataset/prescriptions/rx_PT_15075.png", DocType.PRESCRIPTION),
    ("dataset/discharge_summaries/discharge_PT_15075.png", DocType.DISCHARGE_SUMMARY),
    ("dataset/policy_amendments/amend_PT_15075.png", DocType.POLICY_AMENDMENT),
    ("dataset/unknown/unknown_blurry_scan_001.png", DocType.UNKNOWN),
]


def main() -> None:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set — copy .env.example to .env and fill it in first.")

    correct = 0
    for file_path, expected in SAMPLES:
        result = classify_document(file_path, settings=settings)
        status = "OK " if result.doc_type == expected else "MISS"
        correct += result.doc_type == expected
        print(
            f"[{status}] {file_path}\n"
            f"       expected={expected.value} got={result.doc_type.value} "
            f"confidence={result.confidence:.2f} tags={result.routing_tags}"
        )

    print(f"\n{correct}/{len(SAMPLES)} correct")


if __name__ == "__main__":
    main()
