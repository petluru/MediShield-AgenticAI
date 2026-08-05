from backend.security.redaction import is_sensitive, redact_if_sensitive


def test_is_sensitive_detects_ssn():
    assert is_sensitive("Member SSN: 123-45-6789") is True


def test_is_sensitive_detects_dob():
    assert is_sensitive("Patient DOB: 04/12/1985 confirmed.") is True


def test_is_sensitive_detects_card_like_number():
    assert is_sensitive("Card on file: 4111 1111 1111 1111") is True


def test_is_sensitive_detects_confidential_marker():
    assert is_sensitive("CONFIDENTIAL: internal review notes") is True


def test_is_sensitive_false_for_ordinary_policy_text():
    text = "Section 4.1: Excluded CPT Code Ranges for cosmetic procedures."
    assert is_sensitive(text) is False


def test_is_sensitive_false_for_empty_text():
    assert is_sensitive("") is False


def test_redact_if_sensitive_replaces_whole_chunk():
    text = "Patient SSN 123-45-6789, procedure covered at 80%."
    redacted = redact_if_sensitive(text)
    assert redacted != text
    assert "123-45-6789" not in redacted
    assert "REDACTED" in redacted


def test_redact_if_sensitive_leaves_clean_text_unchanged():
    text = "Section 3: Inclusions cover medically necessary services."
    assert redact_if_sensitive(text) == text
