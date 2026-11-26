"""JSON response shapes for the dashboard API."""

from pydantic import BaseModel

from app.models.dashboard import SourceInfo, TimeBucket, TimeseriesResult
from app.models.request_record import RequestRecord
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
    def from_records(cls, source: str, records: list[RequestRecord]) -> "RequestsResponse":
        """Build the response from stored records."""
        return cls(
            source=source, items=[RequestLogEntry(**record.model_dump()) for record in records]
        )
