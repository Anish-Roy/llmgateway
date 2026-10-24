from app.config import Settings


def test_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/x")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert settings.database_url == "postgresql://u:p@db:5432/x"
    assert settings.groq_api_key.get_secret_value() == "test-key"


def test_api_keys_are_hidden_in_repr(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert "test-key" not in repr(settings)


def test_defaults_without_env(monkeypatch):
    for var in ("APP_ENV", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    settings = Settings(_env_file=None)

    assert settings.app_env == "dev"
    assert settings.groq_api_key is None
