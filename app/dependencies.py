"""Wiring for FastAPI Depends: builds the registry, provider, repository, and chat service."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings
from app.models.routing import RoutingConfig
from app.providers.base import LLMProvider
from app.providers.fake_provider import FakeProvider
from app.providers.groq_provider import GroqProvider
from app.repositories.base import RequestRepository
from app.repositories.sqlite_repository import SqliteRequestRepository
from app.routing.base import Router
from app.routing.factory import build_rule_router
from app.services.chat_service import ChatService
from app.services.fake_profiles import load_fake_profiles
from app.services.model_registry import ModelRegistry
from app.services.routing_config import load_routing_config


@lru_cache
def get_model_registry() -> ModelRegistry:
    """Return the model registry, built once from the configured TOML file."""
    return ModelRegistry(get_settings().models_config_path)


@lru_cache
def get_llm_provider() -> LLMProvider:
    """Return the provider for PROVIDER_MODE (fake or Groq), built once from settings."""
    settings = get_settings()
    if settings.provider_mode == "fake":
        return FakeProvider(
            profiles=load_fake_profiles(settings.fake_config_path),
            seed=settings.fake_seed,
        )
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is required when PROVIDER_MODE=real")
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


@lru_cache
def get_routing_config() -> RoutingConfig:
    """Return the routing settings, loaded once from the configured TOML file."""
    return load_routing_config(get_settings().routing_config_path)


@lru_cache
def get_router() -> Router:
    """Return the rule router, built once from the routing settings."""
    return build_rule_router(get_routing_config())


def get_chat_service(
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    repository: Annotated[RequestRepository, Depends(get_request_repository)],
    router: Annotated[Router, Depends(get_router)],
    routing_config: Annotated[RoutingConfig, Depends(get_routing_config)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatService:
    """Return a ChatService wired with its provider, registry, repository, and router."""
    return ChatService(
        provider=provider,
        registry=registry,
        repository=repository,
        router=router,
        routing_config=routing_config,
        default_model_name=settings.default_model_name,
    )
