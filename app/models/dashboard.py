"""Data shapes for the read-only dashboard."""

from pydantic import BaseModel

from app.models.request_record import RequestRecord


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


class FailedRequest(BaseModel):
    """One failed request with its failure type."""

    record: RequestRecord
    failure_type: str


class FailureSummary(BaseModel):
    """Counts, cost, and tokens of failed requests, by failure type."""

    by_type: dict[str, int]
    errors: int
    truncated: int
    failed_cost_usd: float
    failed_input_tokens: int
    failed_output_tokens: int
    recent: list[FailedRequest]


class RoutingSummary(BaseModel):
    """Where "auto" and explicit requests went, and how often the fallback was used."""

    auto_requests: int
    auto_by_model: dict[str, int]
    explicit_by_model: dict[str, int]
    fallbacks: int
    fallbacks_rescued: int


class LatencyHistogram(BaseModel):
    """Answered requests counted per latency range, per model."""

    bin_labels: list[str]
    by_model: dict[str, list[int]]


class InsightsResult(BaseModel):
    """Failures, routing, latency spread, and the slowest answered requests."""

    failures: FailureSummary
    routing: RoutingSummary
    latency_histogram: LatencyHistogram
    slowest: list[RequestRecord]
