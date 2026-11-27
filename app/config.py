"""Application settings loaded from the environment and .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the gateway, read from environment variables and .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str = ""
    default_model_name: str = "big"
    request_timeout_seconds: float = 30.0
    max_output_tokens: int = 512
    temperature: float = 0.0
    log_level: str = "INFO"
    models_config_path: Path = Path("config/models.toml")
    database_path: str = "data/gateway.db"
    provider_mode: Literal["real", "fake"] = "real"
    fake_config_path: str = "config/fake.toml"
    fake_seed: int = 42
    routing_config_path: str = "config/routing.toml"
    dashboard_data_dir: str = "data"
    results_dir: str = "results"
    load_questions_dir: str = "config"


@lru_cache
def get_settings() -> Settings:
    """Return the settings, built once and cached."""
    return Settings()
