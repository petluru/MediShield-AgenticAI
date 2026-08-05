from backend.security.output_sanitization import sanitize_for_html


def test_sanitize_for_html_escapes_script_tags():
    result = sanitize_for_html("<script>alert('xss')</script>")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_sanitize_for_html_leaves_plain_text_unchanged():
    assert sanitize_for_html("routine office visit, approved") == "routine office visit, approved"


def test_sanitize_for_html_handles_empty_and_none():
    assert sanitize_for_html("") == ""
    assert sanitize_for_html(None) == ""
