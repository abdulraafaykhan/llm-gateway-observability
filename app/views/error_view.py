"""Error response shapes for the API."""

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """The type and message of an error."""

    type: str
    message: str


class ErrorResponse(BaseModel):
    """Body returned when a request fails."""

    error: ErrorDetail
