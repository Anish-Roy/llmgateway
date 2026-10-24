from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Gateway configuration, read from environment variables or a `.env` file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    database_url: str = "postgresql://gateway:gateway@localhost:5433/gateway"

    # SecretStr keeps keys out of logs and reprs; read with .get_secret_value().
    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    request_timeout_s: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
