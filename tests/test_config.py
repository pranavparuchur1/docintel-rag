"""Config must fail fast, loudly, and name the offending variable."""

import pytest
from pydantic import ValidationError

from docintel.config import ConfigError, Settings, get_settings

VALID_UA = "Jane Doe jane@example.com"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Isolate tests from the developer's real environment and .env file."""
    for var in (
        "EDGAR_USER_AGENT",
        "DATABASE_URL",
        "VECTOR_BACKEND",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "OPENAI_API_KEY",
        "DATA_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()


def test_missing_user_agent_fails():
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_reports_variable_name(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # ensure no repo-root .env file is picked up
    with pytest.raises(ConfigError, match="EDGAR_USER_AGENT"):
        get_settings()


def test_anonymous_user_agent_rejected():
    with pytest.raises(ValidationError, match="Your Name email@domain.com"):
        Settings(_env_file=None, edgar_user_agent="mozilla/5.0")


def test_valid_settings_have_local_free_defaults():
    s = Settings(_env_file=None, edgar_user_agent=VALID_UA)
    assert s.vector_backend == "pgvector"
    assert s.embedding_provider == "local"
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.edgar_max_requests_per_sec <= 10.0  # EDGAR fair-access ceiling


def test_openai_provider_requires_key():
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(_env_file=None, edgar_user_agent=VALID_UA, embedding_provider="openai")


def test_openai_provider_with_key_accepted():
    s = Settings(
        _env_file=None,
        edgar_user_agent=VALID_UA,
        embedding_provider="openai",
        openai_api_key="sk-test",
    )
    assert s.embedding_provider == "openai"
