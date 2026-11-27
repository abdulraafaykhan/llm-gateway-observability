"""Routes for the Gateway page and its read-only JSON API."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from app.controllers.dashboard_controller import STATIC_ROOT, serve_asset
from app.dependencies import get_gateway_info_service
from app.services.gateway_info_service import GatewayInfoService
from app.views.gateway_view import (
    GatewayConfigResponse,
    ModelsHealthResponse,
    RoutePreviewRequest,
    RoutePreviewResponse,
)

router = APIRouter(tags=["gateway"])

GATEWAY_DIR = STATIC_ROOT / "gateway"
GATEWAY_ASSETS: dict[str, str] = {"gateway.js": "text/javascript", "gateway.css": "text/css"}

Service = Annotated[GatewayInfoService, Depends(get_gateway_info_service)]


@router.get("/gateway", include_in_schema=False)
async def gateway_page() -> FileResponse:
    """Serve the Gateway page."""
    return FileResponse(GATEWAY_DIR / "index.html", media_type="text/html")


@router.get("/gateway/assets/{file_name}", include_in_schema=False)
async def gateway_asset(file_name: str) -> FileResponse:
    """Serve the Gateway page's script or style sheet."""
    return serve_asset(GATEWAY_DIR, file_name, GATEWAY_ASSETS)


@router.get("/api/gateway/config", response_model=GatewayConfigResponse)
async def gateway_config(service: Service) -> GatewayConfigResponse:
    """Return the non-secret gateway settings, models, and routing settings."""
    return service.config()


@router.get("/api/gateway/models", response_model=ModelsHealthResponse)
async def model_health(
    service: Service, minutes: Annotated[int, Query(ge=1, le=525600)] = 60
) -> ModelsHealthResponse:
    """Return live numbers and a health word for each configured model."""
    return ModelsHealthResponse(minutes=minutes, models=await service.model_health(minutes))


@router.post("/api/gateway/route-preview", response_model=RoutePreviewResponse)
async def route_preview(request: RoutePreviewRequest, service: Service) -> RoutePreviewResponse:
    """Show what the router would choose; no model is called and nothing is recorded."""
    return service.preview(request.question)
