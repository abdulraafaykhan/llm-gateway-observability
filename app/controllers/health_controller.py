"""Router for the health check endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_model_registry
from app.services.model_registry import ModelRegistry

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, str | list[str] | bool]:
    """Report the server status, provider mode, model names, and that "auto" routing is on."""
    return {
        "status": "ok",
        "mode": settings.provider_mode,
        "models": registry.names(),
        "auto": True,
    }
