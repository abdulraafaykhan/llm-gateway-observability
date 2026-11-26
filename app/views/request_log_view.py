"""Response shapes for the request log endpoint."""

from pydantic import BaseModel

from app.models.request_record import RequestRecord


class RequestLogEntry(BaseModel):
    """One recorded request as returned by GET /v1/requests."""

    id: int | None
    created_at: str
    model_name: str
    model_id: str
    question: str
    answer: str | None
    status: str
    error_type: str | None
    error_message: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    requested_model: str
    route_reason: str
    fallback_used: bool


class RequestLogResponse(BaseModel):
    """Body returned by GET /v1/requests."""

    items: list[RequestLogEntry]

    @classmethod
    def from_records(cls, records: list[RequestRecord]) -> "RequestLogResponse":
        """Build the response from stored records."""
        return cls(items=[RequestLogEntry(**record.model_dump()) for record in records])
