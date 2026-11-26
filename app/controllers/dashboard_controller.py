"""Routes for the read-only dashboard page and its JSON API."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from app.dependencies import get_dashboard_service
from app.services.dashboard_service import DashboardService
from app.views.dashboard_view import (
    RequestsResponse,
    SourceListResponse,
    SummaryResponse,
    TimeseriesResponse,
)
from app.views.error_view import ErrorResponse

router = APIRouter(tags=["dashboard"])

STATIC_DIR = Path(__file__).resolve().parent.parent / "static" / "dashboard"
ASSET_TYPES: dict[str, str] = {"dashboard.js": "text/javascript", "dashboard.css": "text/css"}
ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
}

Service = Annotated[DashboardService, Depends(get_dashboard_service)]
Source = Annotated[str | None, Query(description="Database file name in the data folder")]
Minutes = Annotated[int | None, Query(ge=1, le=525600, description="Only the last N minutes")]


@router.get("/dashboard", include_in_schema=False)
async def dashboard_page() -> FileResponse:
    """Serve the dashboard page."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@router.get("/dashboard/assets/{file_name}", include_in_schema=False)
async def dashboard_asset(file_name: str) -> FileResponse:
    """Serve the dashboard's script or style sheet (only those two files)."""
    media_type = ASSET_TYPES.get(file_name)
    if media_type is None:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(STATIC_DIR / file_name, media_type=media_type)


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
