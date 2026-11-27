"""JSON response shapes for the dashboard API."""

from pydantic import BaseModel

from app.models.dashboard import (
    InsightsResult,
    LatencyHistogram,
    ModelAttemptStats,
    RequestWithAttempts,
    RoutingSummary,
    SourceInfo,
    TimeBucket,
    TimeseriesResult,
)
from app.models.stats import GroupStats, StatsReport
from app.views.request_log_view import RequestLogEntry


class SourceListResponse(BaseModel):
    """Body of GET /api/dashboard/sources."""

    sources: list[SourceInfo]
    default: str | None


class SummaryResponse(BaseModel):
    """Body of GET /api/dashboard/summary: the stats command's numbers for one source."""

    source: str
    minutes: int | None
    generated_at: str
    overall: GroupStats
    by_model: dict[str, GroupStats]

    @classmethod
    def from_report(
        cls, source: str, minutes: int | None, report: StatsReport
    ) -> "SummaryResponse":
        """Build the response from a stats report."""
        return cls(
            source=source,
            minutes=minutes,
            generated_at=report.generated_at,
            overall=report.overall,
            by_model=report.by_model,
        )


class TimeseriesResponse(BaseModel):
    """Body of GET /api/dashboard/timeseries."""

    source: str
    minutes: int | None
    bucket_seconds: int
    buckets: list[TimeBucket]

    @classmethod
    def from_result(
        cls, source: str, minutes: int | None, result: TimeseriesResult
    ) -> "TimeseriesResponse":
        """Build the response from time buckets."""
        return cls(
            source=source,
            minutes=minutes,
            bucket_seconds=result.bucket_seconds,
            buckets=result.buckets,
        )


class RequestsResponse(BaseModel):
    """Body of GET /api/dashboard/requests."""

    source: str
    items: list[RequestLogEntry]

    @classmethod
    def from_records(cls, source: str, items: list[RequestWithAttempts]) -> "RequestsResponse":
        """Build the response from stored records and their attempts."""
        return cls(source=source, items=[entry_for(item) for item in items])


def entry_for(item: RequestWithAttempts) -> RequestLogEntry:
    """Build a log entry with its attempts."""
    return RequestLogEntry.from_record(item.record, item.attempts)


class FailureLogEntry(RequestLogEntry):
    """A recorded failed request plus its failure type."""

    failure_type: str


class FailuresView(BaseModel):
    """The failures panel: counts by type, what failures and wasted attempts cost, recent ones."""

    by_type: dict[str, int]
    errors: int
    truncated: int
    failed_cost_usd: float
    failed_input_tokens: int
    failed_output_tokens: int
    wasted_cost_usd: float
    wasted_tokens: int
    recent: list[FailureLogEntry]


class InsightsResponse(BaseModel):
    """Body of GET /api/dashboard/insights."""

    source: str
    minutes: int | None
    failures: FailuresView
    routing: RoutingSummary
    latency_histogram: LatencyHistogram
    slowest: list[RequestLogEntry]
    attempts_by_model: dict[str, ModelAttemptStats]

    @classmethod
    def from_result(
        cls, source: str, minutes: int | None, result: InsightsResult
    ) -> "InsightsResponse":
        """Build the response from computed insights."""
        failures = result.failures
        recent = [
            FailureLogEntry(
                **RequestLogEntry.from_record(item.record, item.attempts).model_dump(),
                failure_type=item.failure_type,
            )
            for item in failures.recent
        ]
        return cls(
            source=source,
            minutes=minutes,
            failures=FailuresView(**failures.model_dump(exclude={"recent"}), recent=recent),
            routing=result.routing,
            latency_histogram=result.latency_histogram,
            slowest=[entry_for(item) for item in result.slowest],
            attempts_by_model=result.attempts_by_model,
        )
