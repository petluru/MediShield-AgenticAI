"""Manual smoke test for the Claims Agent against real dataset claim forms.
Not part of the pytest suite (hits the live Anthropic API, costs money).

Usage:
    uv run python -m backend.scripts.smoke_claims
"""

from backend.agents.claims import extract_claim
from backend.config import get_settings

# (file_path, edge_case note)
SAMPLES: list[tuple[str, str]] = [
    ("dataset/claim_forms/claim_PT_19116.png", "clean"),
    ("dataset/claim_forms/claim_PT_17665.png", "missing_fields (NPI + signature blanked on the form)"),
    ("dataset/claim_forms/claim_PT_62350.png", "fraud: proc_diag_mismatch (schema should still be valid)"),
    ("dataset/claim_forms/claim_PT_69470.png", "fraud: amount_under_10k (schema should still be valid)"),
]


def main() -> None:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set — copy .env.example to .env and fill it in first.")

    for file_path, edge_case in SAMPLES:
        result = extract_claim(file_path, settings=settings)
        fields = result.extracted_fields
        print(
            f"[{file_path}]\n"
            f"       edge_case={edge_case}\n"
            f"       claim_amount={fields.claim_amount} icd10={fields.icd10_codes} "
            f"cpt={fields.cpt_codes} npi={fields.provider_npi} service_date={fields.service_date}\n"
            f"       schema_valid={result.schema_valid} confidence={result.confidence:.2f} "
            f"errors={result.validation_errors}"
        )


if __name__ == "__main__":
    main()
