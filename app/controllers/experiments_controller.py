"""Routes for the Experiments page and its read-only JSON API."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from app.controllers.dashboard_controller import ERROR_RESPONSES, STATIC_ROOT, serve_asset
from app.dependencies import get_experiments_service
from app.services.experiments_service import ExperimentsService
from app.views.experiments_view import CompareResponse, ReplayResponse, RunListResponse

router = APIRouter(tags=["experiments"])

EXPERIMENTS_DIR = STATIC_ROOT / "experiments"
EXPERIMENTS_ASSETS: dict[str, str] = {
    "experiments.js": "text/javascript",
    "experiments.css": "text/css",
}

Service = Annotated[ExperimentsService, Depends(get_experiments_service)]
Split = Annotated[Literal["all", "tune", "test"], Query(description="Which questions to use")]
RunFile = Annotated[str, Query(description="An eval run file name in the results folder")]


@router.get("/experiments", include_in_schema=False)
async def experiments_page() -> FileResponse:
    """Serve the Experiments page."""
    return FileResponse(EXPERIMENTS_DIR / "index.html", media_type="text/html")


@router.get("/experiments/assets/{file_name}", include_in_schema=False)
async def experiments_asset(file_name: str) -> FileResponse:
    """Serve the Experiments page's script or style sheet."""
    return serve_asset(EXPERIMENTS_DIR, file_name, EXPERIMENTS_ASSETS)


@router.get("/api/experiments/runs", response_model=RunListResponse)
async def list_runs(service: Service, split: Split = "all") -> RunListResponse:
    """List the saved eval runs with their summary numbers."""
    return service.runs(split)


@router.get("/api/experiments/replay", response_model=ReplayResponse, responses=ERROR_RESPONSES)
async def replay(
    service: Service, small: RunFile, big: RunFile, split: Split = "all"
) -> ReplayResponse:
    """Return the router replay and cascade replay tables for two saved runs."""
    return service.replay(small, big, split)


@router.get("/api/experiments/compare", response_model=CompareResponse, responses=ERROR_RESPONSES)
async def compare(
    service: Service, a: RunFile, b: RunFile, split: Split = "all"
) -> CompareResponse:
    """Return how two saved runs did on the questions they share."""
    return service.compare(a, b, split)
