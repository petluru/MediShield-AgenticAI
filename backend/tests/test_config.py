from backend.config import Settings, get_settings


def test_settings_defaults_no_env_file():
    settings = Settings(_env_file=None)
    assert settings.classifier_model == "claude-sonnet-5"
    assert settings.escalation_model == "claude-opus-5"
    assert settings.fraud_escalate_min == 0.3


def test_cors_and_auth_token_lists_split_on_comma():
    settings = Settings(_env_file=None, CORS_ORIGINS="http://a.com, http://b.com", API_AUTH_TOKENS="t1,t2")
    assert settings.cors_origins_list == ["http://a.com", "http://b.com"]
    assert settings.api_auth_tokens_list == ["t1", "t2"]


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
