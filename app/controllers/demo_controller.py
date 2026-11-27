"""Routes for starting, watching, and stopping simulated demo traffic."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import get_demo_service
from app.services.demo_service import DemoService
from app.views.demo_view import DemoOptionsResponse, DemoStartRequest, DemoStatusResponse
from app.views.error_view import ErrorResponse

router = APIRouter(prefix="/api/demo", tags=["demo"])

Service = Annotated[DemoService, Depends(get_demo_service)]
DEMO_ERRORS: dict[int | str, dict[str, object]] = {
    400: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
}


@router.get("/status", response_model=DemoStatusResponse)
async def demo_status(service: Service) -> DemoStatusResponse:
    """Report whether demos are available and the progress of the current or last run."""
    return DemoStatusResponse.from_status(service.status())


@router.get("/options", response_model=DemoOptionsResponse)
async def demo_options(service: Service) -> DemoOptionsResponse:
    """List the scenarios, models, and question files the demo form can offer."""
    return DemoOptionsResponse(
        scenarios=service.scenarios(),
        models=service.model_choices(),
        questions_files=service.question_files(),
        database_file=service.database_file,
    )


@router.post("/start", response_model=DemoStatusResponse, responses=DEMO_ERRORS)
async def demo_start(request: DemoStartRequest, service: Service) -> DemoStatusResponse:
    """Start a simulated traffic scenario (fake mode with a 'fake' database only)."""
    status = service.start(request.scenario, request.model, request.questions_file, request.seed)
    return DemoStatusResponse.from_status(status)


@router.post("/stop", response_model=DemoStatusResponse)
async def demo_stop(service: Service) -> DemoStatusResponse:
    """Stop the running demo, if any."""
    return DemoStatusResponse.from_status(await service.stop())
