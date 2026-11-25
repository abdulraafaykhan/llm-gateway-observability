"""Wiring for FastAPI Depends: builds the registry, provider, and chat service."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings
from app.providers.base import LLMProvider
from app.providers.groq_provider import GroqProvider
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


def get_chat_service(
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatService:
    """Return a ChatService wired with the provider, registry, and default model name."""
    return ChatService(
        provider=provider,
        registry=registry,
        default_model_name=settings.default_model_name,
    )
