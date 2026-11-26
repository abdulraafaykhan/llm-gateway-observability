"""Data shapes for the read-only dashboard."""

from pydantic import BaseModel


class SourceInfo(BaseModel):
    """One database file the dashboard can read."""

    name: str
    size_bytes: int
    modified_at: str
    is_default: bool


class TimeBucket(BaseModel):
    """Request counts, cost, and answered-only latency for one time slot."""

    start: str
    requests: int
    answered: int
    errors: int
    truncated: int
    fallbacks: int
    cost_usd: float
    p50_ms: float | None
    p95_ms: float | None


class TimeseriesResult(BaseModel):
    """Equal-width, gap-free time buckets for a range."""

    bucket_seconds: int
    buckets: list[TimeBucket]
