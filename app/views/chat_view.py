"""Request and response shapes for the chat endpoint."""

from pydantic import BaseModel, Field

from app.models.chat_outcome import ChatOutcome
from app.models.completion import Message
from app.models.request_record import AttemptRecord


class AttemptView(BaseModel):
    """One provider call within a request, as returned by the API."""

    attempt: int
    model: str
    status: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    error_message: str | None

    @classmethod
    def from_record(cls, record: AttemptRecord) -> "AttemptView":
        """Build the view from a stored attempt."""
        return cls(
            attempt=record.attempt,
            model=record.model_name,
            status=record.status,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            latency_ms=round(record.latency_ms, 1),
            cost_usd=record.cost_usd,
            error_message=record.error_message,
        )


class ChatRequest(BaseModel):
    """Body of POST /v1/chat."""

    messages: list[Message] = Field(min_length=1)
    model: str | None = Field(
        default=None, description='A model name such as "small" or "big", or "auto" to route.'
    )


class ChatResponse(BaseModel):
    """Body returned by POST /v1/chat; tokens, latency, and cost are totals over all attempts."""

    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    truncated: bool
    requested_model: str
    route_reason: str
    fallback_used: bool
    cost_usd: float
    attempts: list[AttemptView]

    @classmethod
    def from_outcome(cls, outcome: ChatOutcome) -> "ChatResponse":
        """Build the response from the chat outcome."""
        return cls(
            answer=outcome.answer,
            model=outcome.model_name,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            latency_ms=round(outcome.latency_ms, 1),
            truncated=outcome.truncated,
            requested_model=outcome.requested_model,
            route_reason=outcome.route_reason,
            fallback_used=outcome.fallback_used,
            cost_usd=outcome.cost_usd,
            attempts=[AttemptView.from_record(attempt) for attempt in outcome.attempts],
        )
