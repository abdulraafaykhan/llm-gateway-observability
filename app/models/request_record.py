"""Data shape for one recorded chat request."""

from typing import Literal

from pydantic import BaseModel


class RequestRecord(BaseModel):
    """One row in the request log: a success, a cut-off answer, or a failure."""

    id: int | None = None
    created_at: str
    model_name: str
    model_id: str
    question: str
    answer: str | None
    status: Literal["ok", "truncated", "error"]
    error_type: str | None
    error_message: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    requested_model: str = ""
    route_reason: str = ""
    fallback_used: bool = False


class AttemptRecord(BaseModel):
    """One provider call within a request; a request has one attempt, or two with a fallback."""

    id: int | None = None
    request_id: int | None = None
    attempt: int
    model_name: str
    model_id: str
    status: Literal["ok", "truncated", "error"]
    error_type: str | None
    error_message: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
