"""Router for reading the request log."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.dependencies import get_request_repository
from app.repositories.base import RequestRepository
from app.views.request_log_view import RequestLogResponse

router = APIRouter(prefix="/v1", tags=["requests"])


@router.get("/requests", response_model=RequestLogResponse)
async def list_requests(
    repository: Annotated[RequestRepository, Depends(get_request_repository)],
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> RequestLogResponse:
    """Return the most recent recorded requests, newest first."""
    records = await repository.list_recent(limit)
    return RequestLogResponse.from_records(records)
