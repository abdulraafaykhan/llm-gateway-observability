"""Request and response shapes for the chat endpoint."""

from pydantic import BaseModel, Field

from app.models.chat_outcome import ChatOutcome
from app.models.completion import Message


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
        )
