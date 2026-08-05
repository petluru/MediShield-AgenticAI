from unittest.mock import MagicMock, patch

from backend.agents.claims import _ClaimsExtraction, extract_claim, validate_claim_fields
from backend.config import Settings
from backend.models import ExtractedClaimFields


def make_settings():
    return Settings(_env_file=None, ANTHROPIC_API_KEY="sk-ant-test")


def make_valid_fields(
    claim_amount: float | None = 40750.0,
    icd10_codes: list[str] | None = None,
    cpt_codes: list[str] | None = None,
    provider_npi: str | None = "1234567890",
    service_date: str | None = "05/26/2025",
) -> ExtractedClaimFields:
    return ExtractedClaimFields(
        claim_amount=claim_amount,
        icd10_codes=icd10_codes if icd10_codes is not None else ["K35.80"],
        cpt_codes=cpt_codes if cpt_codes is not None else ["29827", "90837", "47562"],
        provider_npi=provider_npi,
        service_date=service_date,
    )


def test_validate_claim_fields_accepts_well_formed_claim():
    assert validate_claim_fields(make_valid_fields()) == []


def test_validate_claim_fields_flags_missing_npi():
    errors = validate_claim_fields(make_valid_fields(provider_npi=None))
    assert any("NPI" in e for e in errors)


def test_validate_claim_fields_flags_malformed_npi():
    errors = validate_claim_fields(make_valid_fields(provider_npi="12345"))
    assert any("NPI" in e and "10-digit" in e for e in errors)


def test_validate_claim_fields_flags_zero_or_missing_amount():
    assert any("claim_amount" in e for e in validate_claim_fields(make_valid_fields(claim_amount=0)))
    assert any("claim_amount" in e for e in validate_claim_fields(make_valid_fields(claim_amount=None)))


def test_validate_claim_fields_flags_malformed_icd10():
    errors = validate_claim_fields(make_valid_fields(icd10_codes=["not-a-code"]))
    assert any("ICD-10" in e for e in errors)


def test_validate_claim_fields_flags_malformed_cpt():
    errors = validate_claim_fields(make_valid_fields(cpt_codes=["123"]))
    assert any("CPT" in e for e in errors)


def test_validate_claim_fields_flags_missing_service_date():
    errors = validate_claim_fields(make_valid_fields(service_date=None))
    assert any("service_date" in e for e in errors)


@patch("backend.agents.claims.build_chat_anthropic")
@patch("backend.agents.claims.encode_image")
def test_extract_claim_marks_schema_invalid_when_fields_fail_validation(mock_encode, mock_build):
    mock_encode.return_value = {"type": "image", "source": {}}
    fake_llm = MagicMock()
    fake_structured = MagicMock()
    fake_structured.invoke.return_value = _ClaimsExtraction(
        extracted_fields=make_valid_fields(provider_npi=None),
        confidence=0.9,
    )
    fake_llm.with_structured_output.return_value = fake_structured
    mock_build.return_value = fake_llm

    result = extract_claim("dataset/claim_forms/claim_PT_17665.png", settings=make_settings())

    assert result.schema_valid is False
    assert any("NPI" in e for e in result.validation_errors)
    assert result.confidence == 0.9


@patch("backend.agents.claims.build_chat_anthropic")
@patch("backend.agents.claims.encode_image")
def test_extract_claim_marks_schema_valid_for_clean_claim(mock_encode, mock_build):
    mock_encode.return_value = {"type": "image", "source": {}}
    fake_llm = MagicMock()
    fake_structured = MagicMock()
    fake_structured.invoke.return_value = _ClaimsExtraction(extracted_fields=make_valid_fields(), confidence=0.95)
    fake_llm.with_structured_output.return_value = fake_structured
    mock_build.return_value = fake_llm

    result = extract_claim("dataset/claim_forms/claim_PT_19116.png", settings=make_settings())

    assert result.schema_valid is True
    assert result.validation_errors == []
