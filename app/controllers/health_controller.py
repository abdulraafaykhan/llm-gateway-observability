"""Router for the health check endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import get_model_registry
from app.services.model_registry import ModelRegistry

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
) -> dict[str, str | list[str]]:
    """Report that the server is up and list the configured model names."""
    return {"status": "ok", "models": registry.names()}
