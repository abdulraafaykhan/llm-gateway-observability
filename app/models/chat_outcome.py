"""Data shape for the final result of one chat request (all attempts together)."""

from pydantic import BaseModel

from app.models.request_record import AttemptRecord


class ChatOutcome(BaseModel):
    """The final answer plus totals across all attempts, and how the model was chosen."""

    answer: str
    truncated: bool
    model_name: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    requested_model: str
    route_reason: str
    fallback_used: bool
    attempts: list[AttemptRecord]
