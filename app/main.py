"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.controllers import (
    chat_controller,
    dashboard_controller,
    demo_controller,
    experiments_controller,
    gateway_controller,
    health_controller,
    request_log_controller,
)
from app.dependencies import (
    get_llm_provider,
    get_model_registry,
    get_request_repository,
    get_router,
    get_routing_config,
    stop_running_demo,
)
from app.providers.base import ProviderError
from app.services.dashboard_service import NoDataError, UnknownSourceError
from app.services.demo_service import (
    DemoBusyError,
    DemoUnavailableError,
    TooManyDemoRequestsError,
)
from app.services.experiments_service import RunsNotComparableError, UnknownFileError
from app.services.model_registry import UnknownModelError
from app.services.routing_config import validate_routing_models
from app.views.error_view import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging, build the registry and provider, and initialize storage at startup."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    registry = get_model_registry()
    logger.info("Loaded models: %s", ", ".join(registry.names()))
    get_llm_provider()
    routing_config = get_routing_config()
    validate_routing_models(routing_config, registry)
    get_router()
    logger.info(
        "Router ready: small=%s big=%s threshold=%d fallback=%s",
        routing_config.small_model,
        routing_config.big_model,
        routing_config.threshold,
        routing_config.fallback_model if routing_config.fallback_enabled else "off",
    )
    if settings.provider_mode == "fake":
        logger.warning("FAKE MODE: answers and timings are simulated")
    await get_request_repository().initialize()
    logger.info("Request log database ready: %s", settings.database_path)
    yield
    await stop_running_demo()


def error_response(status_code: int, error_type: str, message: str) -> JSONResponse:
    """Build a JSON error response in the ErrorResponse shape."""
    body = ErrorResponse(error=ErrorDetail(type=error_type, message=message))
    return JSONResponse(status_code=status_code, content=body.model_dump())


async def handle_unknown_model(request: Request, error: UnknownModelError) -> JSONResponse:
    """Return 400 when the requested model name is not configured."""
    return error_response(400, "unknown_model", str(error))


async def handle_provider_error(request: Request, error: ProviderError) -> JSONResponse:
    """Return 502 when the provider call fails."""
    return error_response(502, "provider_error", str(error))


async def handle_unknown_source(request: Request, error: UnknownSourceError) -> JSONResponse:
    """Return 400 when the dashboard is asked for a database file that is not listed."""
    return error_response(400, "unknown_source", str(error))


async def handle_no_data(request: Request, error: NoDataError) -> JSONResponse:
    """Return 404 when the dashboard's data folder has no database files."""
    return error_response(404, "no_data", str(error))


# Errors from the Gateway, Experiments, and demo services: (class, HTTP status, error type).
ERROR_TYPES: tuple[tuple[type[Exception], int, str], ...] = (
    (UnknownFileError, 400, "unknown_file"),
    (RunsNotComparableError, 400, "runs_not_comparable"),
    (TooManyDemoRequestsError, 400, "too_many_requests"),
    (DemoUnavailableError, 409, "demo_unavailable"),
    (DemoBusyError, 409, "demo_busy"),
)


def simple_handler(
    status_code: int, error_type: str
) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
    """Build a handler that returns the error's message in the ErrorResponse shape."""

    async def handle(request: Request, error: Exception) -> JSONResponse:
        """Return the fixed status and error type with the error's message."""
        return error_response(status_code, error_type, str(error))

    return handle


def create_app() -> FastAPI:
    """Create the FastAPI app with routers and exception handlers."""
    app = FastAPI(title="LLM Gateway", lifespan=lifespan)
    app.include_router(chat_controller.router)
    app.include_router(health_controller.router)
    app.include_router(request_log_controller.router)
    app.include_router(dashboard_controller.router)
    app.include_router(gateway_controller.router)
    app.include_router(experiments_controller.router)
    app.include_router(demo_controller.router)
    app.add_exception_handler(UnknownModelError, handle_unknown_model)
    app.add_exception_handler(ProviderError, handle_provider_error)
    app.add_exception_handler(UnknownSourceError, handle_unknown_source)
    app.add_exception_handler(NoDataError, handle_no_data)
    for error_class, status_code, error_type in ERROR_TYPES:
        app.add_exception_handler(error_class, simple_handler(status_code, error_type))
    return app


app = create_app()
