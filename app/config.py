"""Application settings loaded from the environment and .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the gateway, read from environment variables and .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str
    default_model_name: str = "big"
    request_timeout_seconds: float = 30.0
    max_output_tokens: int = 512
    temperature: float = 0.0
    log_level: str = "INFO"
    models_config_path: Path = Path("config/models.toml")


@lru_cache
def get_settings() -> Settings:
    """Return the settings, built once and cached."""
    return Settings()
