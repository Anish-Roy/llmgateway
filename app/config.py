from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Gateway configuration, read from environment variables or a `.env` file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"
    # Any case: "debug" works too. An unknown level stops the app at startup.
    log_level: Annotated[
        Literal["DEBUG", "INFO", "WARNING", "ERROR"], BeforeValidator(lambda v: str(v).upper())
    ] = "INFO"

    database_url: str = "postgresql://gateway:gateway@localhost:5433/gateway"

    # SecretStr keeps keys out of logs and reprs; read with .get_secret_value().
    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    # Per provider call (the HTTP client's timeout).
    request_timeout_s: float = 30.0
    # For the whole request, across retries and fallbacks. Without it, 2 providers x 2 tries
    # x 30 s could keep a client waiting two minutes.
    request_deadline_s: float = 45.0

    # How long a key lookup is cached. Also how long a revoked key keeps working, and how long
    # a changed limit takes to apply, on a running gateway.
    key_cache_ttl_s: float = 30.0

    # Tokens reserved for the reply when a request sets no max_tokens, until the real usage
    # is known. Higher is stricter on keys that send many such requests at once.
    rate_limit_default_completion_tokens: int = 1024

    # The exact-match cache answers only requests whose temperature is set and at most this.
    # Above it, answers are meant to vary. 0 caches only fully greedy requests.
    cache_max_temperature: float = Field(default=0.2, ge=0, le=2)

    # Model aliases and the providers behind them. See config/models.yaml.
    models_file: Path = Path(__file__).resolve().parent.parent / "config" / "models.yaml"


@lru_cache
def get_settings() -> Settings:
    return Settings()
