from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.agents.kyc import verify_kyc
from backend.config import Settings
from backend.models import KYCOutput


def make_settings():
    return Settings(_env_file=None, ANTHROPIC_API_KEY="sk-ant-test")


@patch("backend.agents.kyc.build_chat_anthropic")
@patch("backend.agents.kyc.encode_image")
def test_verify_kyc_passes_as_of_date_into_prompt(mock_encode, mock_build):
    mock_encode.return_value = {"type": "image", "source": {}}
    fake_llm = MagicMock()
    fake_structured = MagicMock()
    fake_structured.invoke.return_value = KYCOutput(kyc_passed=True, flags=[], confidence=0.9)
    fake_llm.with_structured_output.return_value = fake_structured
    mock_build.return_value = fake_llm

    result = verify_kyc(
        "dataset/id_documents/id_PT_19116.png",
        as_of=date(2025, 8, 15),
        settings=make_settings(),
    )

    assert result.kyc_passed is True
    messages = fake_structured.invoke.call_args[0][0]
    human_text = messages[1].content[0]["text"]
    assert "2025-08-15" in human_text
    assert "untrusted" in human_text


@patch("backend.agents.kyc.build_chat_anthropic")
@patch("backend.agents.kyc.encode_image")
def test_verify_kyc_resolves_relative_path_against_repo_root(mock_encode, mock_build):
    mock_encode.return_value = {"type": "image", "source": {}}
    fake_llm = MagicMock()
    fake_structured = MagicMock()
    fake_structured.invoke.return_value = KYCOutput(kyc_passed=True, flags=[], confidence=0.9)
    fake_llm.with_structured_output.return_value = fake_structured
    mock_build.return_value = fake_llm

    verify_kyc("dataset/id_documents/id_PT_19116.png", as_of=date(2025, 8, 15), settings=make_settings())

    encoded_path = mock_encode.call_args[0][0]
    assert isinstance(encoded_path, Path)
    assert encoded_path.is_absolute()
    assert encoded_path.name == "id_PT_19116.png"
