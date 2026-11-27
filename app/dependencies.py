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
from app.services.dashboard_service import DashboardService
from app.services.demo_service import DemoService
from app.services.experiments_service import ExperimentsService
from app.services.fake_profiles import load_fake_profiles
from app.services.gateway_info_service import GatewayInfoService
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


def sqlite_repository_factory(path: Path) -> RequestRepository:
    """Build a SQLite repository for a dashboard source; the dashboard only reads from it."""
    return SqliteRequestRepository(path)


@lru_cache
def get_dashboard_service() -> DashboardService:
    """Return the dashboard service, built once from settings."""
    settings = get_settings()
    return DashboardService(
        data_dir=settings.dashboard_data_dir,
        default_database_path=settings.database_path,
        repository_factory=sqlite_repository_factory,
    )


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


@lru_cache
def get_gateway_info_service() -> GatewayInfoService:
    """Return the Gateway page service, built once (it reads the gateway's own database)."""
    return GatewayInfoService(
        settings=get_settings(),
        registry=get_model_registry(),
        routing_config=get_routing_config(),
        router=get_router(),
        repository=get_request_repository(),
    )


@lru_cache
def get_experiments_service() -> ExperimentsService:
    """Return the Experiments page service, built once from the results folder."""
    return ExperimentsService(get_settings().results_dir, get_routing_config())


@lru_cache
def get_demo_service() -> DemoService:
    """Return the demo service; its requests go through the same ChatService path as real ones."""
    settings = get_settings()
    registry = get_model_registry()
    chat_service = ChatService(
        provider=get_llm_provider(),
        registry=registry,
        repository=get_request_repository(),
        router=get_router(),
        routing_config=get_routing_config(),
        default_model_name=settings.default_model_name,
    )
    return DemoService(
        chat_service=chat_service,
        model_names=registry.names(),
        provider_mode=settings.provider_mode,
        database_file=Path(settings.database_path).name,
        questions_dir=settings.load_questions_dir,
    )


async def stop_running_demo() -> None:
    """Cancel a running demo at shutdown, if the demo service was ever built."""
    if get_demo_service.cache_info().currsize:
        await get_demo_service().stop()
