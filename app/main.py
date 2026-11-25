"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.controllers import chat_controller, health_controller, request_log_controller
from app.dependencies import get_model_registry, get_request_repository
from app.providers.base import ProviderError
from app.services.model_registry import UnknownModelError
from app.views.error_view import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging, load the model registry, and initialize storage once at startup."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    registry = get_model_registry()
    logger.info("Loaded models: %s", ", ".join(registry.names()))
    await get_request_repository().initialize()
    logger.info("Request log database ready: %s", settings.database_path)
    yield


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


def create_app() -> FastAPI:
    """Create the FastAPI app with routers and exception handlers."""
    app = FastAPI(title="LLM Gateway", lifespan=lifespan)
    app.include_router(chat_controller.router)
    app.include_router(health_controller.router)
    app.include_router(request_log_controller.router)
    app.add_exception_handler(UnknownModelError, handle_unknown_model)
    app.add_exception_handler(ProviderError, handle_provider_error)
    return app


app = create_app()
