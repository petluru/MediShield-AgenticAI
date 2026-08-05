"""Manual smoke test for the Fraud Detection Agent against real dataset
patient identities. Not part of the pytest suite (hits the live Anthropic
API, costs money). Claim details below are illustrative (not re-extracted
via the Claims Agent for this smoke test) except patient_id, which is real
and drives the lookup_claim_history tool against the actual dataset.

Usage:
    uv run python -m backend.scripts.smoke_fraud_detection
"""

from backend.agents.fraud_detection import assess_fraud_risk
from backend.config import get_settings

# (patient_id, claim_amount, cpt_codes, icd10_codes note, service_date, note)
SAMPLES = [
    ("PT_20322", 5200.0, ["99213"], "11/15/2025", "has a real duplicate claim on record (claim_PT_20322 + _dup)"),
    ("PT_19116", 4095.0, ["90837", "93000"], "11/30/2025", "clean, single claim on record"),
    ("PT_62350", 18900.0, ["59400", "27447"], "11/15/2025", "maternity CPT 59400 billed with an appendicitis-only claim history"),
]


def main() -> None:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set — copy .env.example to .env and fill it in first.")

    for patient_id, claim_amount, cpt_codes, service_date, note in SAMPLES:
        result = assess_fraud_risk(
            patient_id=patient_id,
            claim_amount=claim_amount,
            cpt_codes=cpt_codes,
            service_date=service_date,
            settings=settings,
        )
        print(
            f"[{patient_id}] {note}\n"
            f"       fraud_score={result.fraud_score:.2f} risk_level={result.risk_level.value} "
            f"escalated_to_opus={result.escalated_to_opus}\n"
            f"       anomalies={result.anomalies}"
        )


if __name__ == "__main__":
    main()
