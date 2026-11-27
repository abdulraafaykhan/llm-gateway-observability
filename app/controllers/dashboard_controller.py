"""Routes for the web pages (chat and dashboard), their static files, and the dashboard API."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse

from app.dependencies import get_dashboard_service
from app.services.dashboard_service import DashboardService
from app.views.dashboard_view import (
    InsightsResponse,
    RequestsResponse,
    SourceListResponse,
    SummaryResponse,
    TimeseriesResponse,
)
from app.views.error_view import ErrorResponse

router = APIRouter(tags=["dashboard"])

STATIC_ROOT = Path(__file__).resolve().parent.parent / "static"
DASHBOARD_DIR = STATIC_ROOT / "dashboard"
CHAT_DIR = STATIC_ROOT / "chat"
SHARED_DIR = STATIC_ROOT / "shared"
DASHBOARD_ASSETS: dict[str, str] = {
    "dashboard.js": "text/javascript",
    "dashboard.css": "text/css",
}
CHAT_ASSETS: dict[str, str] = {"chat.js": "text/javascript", "chat.css": "text/css"}
SHARED_ASSETS: dict[str, str] = {"nav.js": "text/javascript", "nav.css": "text/css"}
ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
}

Service = Annotated[DashboardService, Depends(get_dashboard_service)]
Source = Annotated[str | None, Query(description="Database file name in the data folder")]
Minutes = Annotated[int | None, Query(ge=1, le=525600, description="Only the last N minutes")]


def serve_asset(folder: Path, file_name: str, allowed: dict[str, str]) -> FileResponse:
    """Serve one listed static file from a folder, or 404 for anything else."""
    media_type = allowed.get(file_name)
    if media_type is None:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(folder / file_name, media_type=media_type)


@router.get("/", include_in_schema=False)
async def home() -> RedirectResponse:
    """Send visitors of the site root to the chat page."""
    return RedirectResponse(url="/chat", status_code=307)


@router.get("/chat", include_in_schema=False)
async def chat_page() -> FileResponse:
    """Serve the chat page."""
    return FileResponse(CHAT_DIR / "index.html", media_type="text/html")


@router.get("/chat/assets/{file_name}", include_in_schema=False)
async def chat_asset(file_name: str) -> FileResponse:
    """Serve the chat page's script or style sheet."""
    return serve_asset(CHAT_DIR, file_name, CHAT_ASSETS)


@router.get("/shared/{file_name}", include_in_schema=False)
async def shared_asset(file_name: str) -> FileResponse:
    """Serve the navigation script or style sheet shared by all pages."""
    return serve_asset(SHARED_DIR, file_name, SHARED_ASSETS)


@router.get("/dashboard", include_in_schema=False)
async def dashboard_page() -> FileResponse:
    """Serve the dashboard page."""
    return FileResponse(DASHBOARD_DIR / "index.html", media_type="text/html")


@router.get("/dashboard/assets/{file_name}", include_in_schema=False)
async def dashboard_asset(file_name: str) -> FileResponse:
    """Serve the dashboard's script or style sheet (only those two files)."""
    return serve_asset(DASHBOARD_DIR, file_name, DASHBOARD_ASSETS)


@router.get("/api/dashboard/sources", response_model=SourceListResponse)
async def list_sources(service: Service) -> SourceListResponse:
    """List the database files the dashboard can read."""
    return SourceListResponse(sources=service.list_sources(), default=service.default_source())


@router.get("/api/dashboard/summary", response_model=SummaryResponse, responses=ERROR_RESPONSES)
async def summary(
    service: Service, source: Source = None, minutes: Minutes = None
) -> SummaryResponse:
    """Return the stats report for one source and range."""
    name = service.resolve_source(source)
    report = await service.summary(name, minutes)
    return SummaryResponse.from_report(name, minutes, report)


@router.get(
    "/api/dashboard/timeseries", response_model=TimeseriesResponse, responses=ERROR_RESPONSES
)
async def timeseries(
    service: Service, source: Source = None, minutes: Minutes = None
) -> TimeseriesResponse:
    """Return time buckets for one source and range."""
    name = service.resolve_source(source)
    result = await service.timeseries(name, minutes)
    return TimeseriesResponse.from_result(name, minutes, result)


@router.get("/api/dashboard/insights", response_model=InsightsResponse, responses=ERROR_RESPONSES)
async def insights(
    service: Service, source: Source = None, minutes: Minutes = None
) -> InsightsResponse:
    """Return failures, routing, latency spread, and slowest requests for one source and range."""
    name = service.resolve_source(source)
    result = await service.insights(name, minutes)
    return InsightsResponse.from_result(name, minutes, result)


@router.get("/api/dashboard/requests", response_model=RequestsResponse, responses=ERROR_RESPONSES)
async def recent_requests(
    service: Service,
    source: Source = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    minutes: Minutes = None,
) -> RequestsResponse:
    """Return the newest recorded requests for one source and range."""
    name = service.resolve_source(source)
    records = await service.recent(name, limit, minutes)
    return RequestsResponse.from_records(name, records)
