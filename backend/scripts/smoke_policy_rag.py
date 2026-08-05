"""Manual smoke test for the Policy RAG Agent against the real ingested
policy PDFs. Not part of the pytest suite (hits the live Anthropic API,
costs money). Run `uv run python -m backend.scripts.ingest_policies` first.

Usage:
    uv run python -m backend.scripts.smoke_policy_rag
"""

from backend.agents.policy_rag import determine_coverage
from backend.config import get_settings

# (plan, cpt_codes, icd10_codes, expectation note)
SAMPLES: list[tuple[str, list[str], list[str] | None, str]] = [
    ("gold", ["21920"], None, "liposuction — expect NOT covered (cosmetic exclusion)"),
    ("gold", ["99213"], ["J01.90"], "routine office visit — expect covered"),
    ("silver", ["59400"], ["K35.80"], "maternity procedure billed for appendicitis diagnosis — proc/diag mismatch"),
]


def main() -> None:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set — copy .env.example to .env and fill it in first.")

    for plan, cpt_codes, icd10_codes, note in SAMPLES:
        result = determine_coverage(plan, cpt_codes, icd10_codes, settings=settings)
        print(
            f"[plan={plan} cpt={cpt_codes} icd10={icd10_codes}]\n"
            f"       note={note}\n"
            f"       covered={result.covered} coverage_pct={result.coverage_percentage} "
            f"confidence={result.confidence:.2f}\n"
            f"       clause={result.policy_clause[:200]}\n"
            f"       exclusions={result.exclusions}"
        )


if __name__ == "__main__":
    main()
