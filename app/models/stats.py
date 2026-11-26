"""Data shapes for request statistics."""

from pydantic import BaseModel


class LatencyStats(BaseModel):
    """Latency summary over answered requests, in milliseconds."""

    count: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


class GroupStats(BaseModel):
    """Counts, tokens, cost, and latency for one group of requests."""

    requests: int
    answered: int
    truncated: int
    errors: int
    error_rate_percent: float
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    cost_per_1000_answered_usd: float
    latency: LatencyStats | None
    fallbacks: int = 0


class StatsReport(BaseModel):
    """Stats for all requests together and for each model."""

    generated_at: str
    since: str | None
    overall: GroupStats
    by_model: dict[str, GroupStats]
