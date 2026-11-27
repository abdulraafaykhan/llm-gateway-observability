"""Response shapes for the request log endpoint."""

from pydantic import BaseModel, Field

from app.models.request_record import AttemptRecord, RequestRecord
from app.views.chat_view import AttemptView


class RequestLogEntry(BaseModel):
    """One recorded request as returned by GET /v1/requests and the dashboard."""

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
    attempts: list[AttemptView] = Field(default_factory=list)

    @classmethod
    def from_record(
        cls, record: RequestRecord, attempts: list[AttemptRecord] | None = None
    ) -> "RequestLogEntry":
        """Build the entry from a stored request and, optionally, its attempts."""
        views = [AttemptView.from_record(attempt) for attempt in attempts or []]
        return cls(**record.model_dump(), attempts=views)


class RequestLogResponse(BaseModel):
    """Body returned by GET /v1/requests."""

    items: list[RequestLogEntry]

    @classmethod
    def from_records(cls, records: list[RequestRecord]) -> "RequestLogResponse":
        """Build the response from stored records (without attempts)."""
        return cls(items=[RequestLogEntry.from_record(record) for record in records])
