"""Manual smoke test for the KYC Agent against real dataset ID images.
Not part of the pytest suite (hits the live Anthropic API, costs money).

Note on dates: the dataset's synthetic treatment dates fall somewhere in
2025, and expiry offsets in generate_docs.py (_exp_dates) are computed
relative to each case's own treatment_date, not real wall-clock time. We
pass a fixed `as_of` in that same window (rather than real "today") so the
expired / expiring-soon / valid cases actually differ from each other —
the eval harness (task #11) will need the same treatment for a stable
comparison across the whole dataset.

Usage:
    uv run python -m backend.scripts.smoke_kyc
"""

from datetime import date

from backend.agents.kyc import verify_kyc
from backend.config import get_settings

AS_OF = date(2025, 8, 15)

# (file_path, edge_flag or None, fraud_reason or None)
SAMPLES: list[tuple[str, str | None]] = [
    ("dataset/id_documents/id_PT_19116.png", None),  # clean
    ("dataset/id_documents/id_PT_99733.png", None),  # clean
    ("dataset/id_documents/id_PT_15075.png", "expired_id"),
    ("dataset/id_documents/id_PT_57795.png", "tampered_id"),
    ("dataset/id_documents/id_PT_50538.png", "expiring_soon_id"),
    ("dataset/id_documents/id_PT_39451.png", "name_mismatch (fraud, NOT visually detectable from the ID alone)"),
]


def main() -> None:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set — copy .env.example to .env and fill it in first.")

    for file_path, edge_case in SAMPLES:
        result = verify_kyc(file_path, as_of=AS_OF, settings=settings)
        print(
            f"[{file_path}]\n"
            f"       edge_case={edge_case}\n"
            f"       kyc_passed={result.kyc_passed} confidence={result.confidence:.2f} "
            f"flags={result.flags}"
        )


if __name__ == "__main__":
    main()
