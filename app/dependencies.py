"""Wiring for FastAPI Depends: builds the registry, provider, repository, and chat service."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings
from app.providers.base import LLMProvider
from app.providers.groq_provider import GroqProvider
from app.repositories.base import RequestRepository
from app.repositories.sqlite_repository import SqliteRequestRepository
from app.services.chat_service import ChatService
from app.services.model_registry import ModelRegistry


@lru_cache
def get_model_registry() -> ModelRegistry:
    """Return the model registry, built once from the configured TOML file."""
    return ModelRegistry(get_settings().models_config_path)


@lru_cache
def get_llm_provider() -> LLMProvider:
    """Return the Groq provider, built once from settings."""
    settings = get_settings()
    return GroqProvider(
        api_key=settings.groq_api_key,
        timeout_seconds=settings.request_timeout_seconds,
        max_output_tokens=settings.max_output_tokens,
        temperature=settings.temperature,
    )


@lru_cache
def get_request_repository() -> RequestRepository:
    """Return the request repository, built once from the configured database path."""
    return SqliteRequestRepository(Path(get_settings().database_path))


def get_chat_service(
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    repository: Annotated[RequestRepository, Depends(get_request_repository)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatService:
    """Return a ChatService wired with the provider, registry, repository, and default model."""
    return ChatService(
        provider=provider,
        registry=registry,
        repository=repository,
        default_model_name=settings.default_model_name,
    )
