"""Read-only data for the dashboard: lists database files and summarizes their request logs."""

import bisect
import math
import re
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.models.dashboard import (
    FailedRequest,
    FailureSummary,
    InsightsResult,
    LatencyHistogram,
    ModelAttemptStats,
    RequestWithAttempts,
    RoutingSummary,
    SourceInfo,
    TimeBucket,
    TimeseriesResult,
)
from app.models.request_record import AttemptRecord, RequestRecord
from app.models.stats import StatsReport
from app.repositories.base import RequestRepository
from app.services.failure_types import FAILURE_TYPES, classify_failure
from app.services.stats_service import ANSWERED_STATUSES, StatsService, percentile

SOURCE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+\.db$")
BUCKET_SIZES_SECONDS: tuple[int, ...] = (10, 30, 60, 300, 900, 3600, 21600)
MAX_BUCKETS: int = 120
AUTO_MODEL: str = "auto"
INSIGHT_LIST_LIMIT: int = 10
LATENCY_BIN_EDGES_MS: tuple[float, ...] = (500, 1000, 2000, 5000, 10000, 20000, 60000)
LATENCY_BIN_LABELS: tuple[str, ...] = (
    "<0.5s",
    "0.5-1s",
    "1-2s",
    "2-5s",
    "5-10s",
    "10-20s",
    "20-60s",
    ">60s",
)

RepositoryFactory = Callable[[Path], RequestRepository]


class UnknownSourceError(Exception):
    """Raised when a source name is not a .db file listed in the data folder."""

    def __init__(self, name: str, valid_names: list[str]) -> None:
        self.name: str = name
        self.valid_names: list[str] = valid_names
        valid = ", ".join(valid_names) or "none"
        super().__init__(f"Unknown data source '{name}'. Valid names: {valid}")


class NoDataError(Exception):
    """Raised when the data folder has no database files."""

    def __init__(self, data_dir: str) -> None:
        super().__init__(f"No database files found in {data_dir}")


class DashboardService:
    """Reads request logs from the .db files in one folder; never writes to them."""

    def __init__(
        self, data_dir: str, default_database_path: str, repository_factory: RepositoryFactory
    ) -> None:
        self._data_dir: Path = Path(data_dir)
        self._default_database_path: Path = Path(default_database_path)
        self._repository_factory: RepositoryFactory = repository_factory

    def list_sources(self) -> list[SourceInfo]:
        """Return every .db file directly in the data folder, sorted by name."""
        paths = self._source_paths()
        default = self._default_name([path.name for path in paths])
        sources = []
        for path in paths:
            stat = path.stat()
            sources.append(
                SourceInfo(
                    name=path.name,
                    size_bytes=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                    is_default=path.name == default,
                )
            )
        return sources

    def default_source(self) -> str | None:
        """Return the default source name, or None when there are no sources."""
        return self._default_name(self._source_names())

    def resolve_source(self, source: str | None) -> str:
        """Return a valid source name (the default for None), or raise a clear error."""
        names = self._source_names()
        if not names:
            raise NoDataError(str(self._data_dir))
        if source is None:
            return self._default_name(names) or names[0]
        if not SOURCE_NAME_PATTERN.fullmatch(source) or source not in names:
            raise UnknownSourceError(source, names)
        return source

    async def summary(self, source: str | None, minutes: int | None) -> StatsReport:
        """Return the same report the stats command prints, for the source and range."""
        repository = self._repository(source)
        return await StatsService(repository).build_report(since_iso(utc_now(), minutes))

    async def timeseries(self, source: str | None, minutes: int | None) -> TimeseriesResult:
        """Return gap-free time buckets for the source and range."""
        repository = self._repository(source)
        now = utc_now()
        records = await repository.list_all(since_iso(now, minutes))
        if minutes is not None:
            end = now.timestamp()
            return build_buckets(records, end - minutes * 60, end)
        if not records:
            return TimeseriesResult(bucket_seconds=BUCKET_SIZES_SECONDS[0], buckets=[])
        times = [record_timestamp(record) for record in records]
        return build_buckets(records, min(times), max(times))

    async def insights(self, source: str | None, minutes: int | None) -> InsightsResult:
        """Return failures, routing, latency spread, and slowest requests for the range."""
        repository = self._repository(source)
        records = await repository.list_all(since_iso(utc_now(), minutes))
        attempts = await repository.list_attempts(record_ids(records))
        return build_insights(records, attempts)

    async def recent(
        self, source: str | None, limit: int, minutes: int | None
    ) -> list[RequestWithAttempts]:
        """Return up to limit records with their attempts, newest first."""
        repository = self._repository(source)
        if minutes is None:
            records = await repository.list_recent(limit)
        else:
            window = await repository.list_all(since_iso(utc_now(), minutes))
            records = newest_first(window)[:limit]
        attempts = await repository.list_attempts(record_ids(records))
        return attach_attempts(records, attempts)

    def _repository(self, source: str | None) -> RequestRepository:
        """Build a read-only repository for a validated source name."""
        return self._repository_factory(self._data_dir / self.resolve_source(source))

    def _source_paths(self) -> list[Path]:
        """List .db files with safe names directly in the data folder."""
        if not self._data_dir.is_dir():
            return []
        paths = [
            path
            for path in self._data_dir.iterdir()
            if path.is_file() and SOURCE_NAME_PATTERN.fullmatch(path.name)
        ]
        return sorted(paths, key=lambda path: path.name)

    def _source_names(self) -> list[str]:
        """Return the source names, sorted."""
        return [path.name for path in self._source_paths()]

    def _default_name(self, names: list[str]) -> str | None:
        """Pick DATABASE_PATH's file when it is in the data folder, else the first name."""
        if not names:
            return None
        default = self._default_database_path
        in_folder = default.resolve().parent == self._data_dir.resolve()
        return default.name if in_folder and default.name in names else names[0]


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(UTC)


def since_iso(now: datetime, minutes: int | None) -> str | None:
    """Return the UTC ISO start of the range, or None for all time."""
    return None if minutes is None else (now - timedelta(minutes=minutes)).isoformat()


def record_timestamp(record: RequestRecord) -> float:
    """Return the record's created_at as seconds since the Unix epoch."""
    return datetime.fromisoformat(record.created_at).timestamp()


def choose_bucket_seconds(window_seconds: float) -> int:
    """Pick the smallest bucket size that gives at most MAX_BUCKETS buckets."""
    for size in BUCKET_SIZES_SECONDS:
        if max(1, math.ceil(window_seconds / size)) <= MAX_BUCKETS:
            return size
    return BUCKET_SIZES_SECONDS[-1]


def build_buckets(records: list[RequestRecord], start: float, end: float) -> TimeseriesResult:
    """Group records into epoch-aligned buckets from start to end, including empty ones."""
    size = choose_bucket_seconds(end - start)
    first = math.floor(start / size) * size
    last = math.floor(end / size) * size
    groups: list[list[RequestRecord]] = [[] for _ in range((last - first) // size + 1)]
    for record in records:
        index = (math.floor(record_timestamp(record) / size) * size - first) // size
        if 0 <= index < len(groups):
            groups[index].append(record)
    buckets = [make_bucket(first + index * size, group) for index, group in enumerate(groups)]
    return TimeseriesResult(bucket_seconds=size, buckets=buckets)


def make_bucket(start_seconds: int, records: list[RequestRecord]) -> TimeBucket:
    """Summarize one bucket; latency uses answered rows only."""
    latencies = [record.latency_ms for record in records if record.status in ANSWERED_STATUSES]
    return TimeBucket(
        start=datetime.fromtimestamp(start_seconds, UTC).isoformat(),
        requests=len(records),
        answered=len(latencies),
        errors=sum(1 for record in records if record.status == "error"),
        truncated=sum(1 for record in records if record.status == "truncated"),
        fallbacks=sum(1 for record in records if record.fallback_used),
        cost_usd=sum(record.cost_usd for record in records),
        p50_ms=percentile(latencies, 50) if latencies else None,
        p95_ms=percentile(latencies, 95) if latencies else None,
    )


AttemptsById = dict[int, list[AttemptRecord]]


def record_ids(records: list[RequestRecord]) -> list[int]:
    """Return the ids of stored records."""
    return [record.id for record in records if record.id is not None]


def attempts_of(record: RequestRecord, attempts_by_id: AttemptsById) -> list[AttemptRecord]:
    """Return a record's attempts, or an empty list for rows recorded before attempts."""
    return attempts_by_id.get(record.id, []) if record.id is not None else []


def attach_attempts(
    records: list[RequestRecord], attempts_by_id: AttemptsById
) -> list[RequestWithAttempts]:
    """Pair each record with its attempts."""
    return [
        RequestWithAttempts(record=record, attempts=attempts_of(record, attempts_by_id))
        for record in records
    ]


def build_insights(records: list[RequestRecord], attempts_by_id: AttemptsById) -> InsightsResult:
    """Compute every insight panel from one list of rows and their attempts."""
    return InsightsResult(
        failures=summarize_failures(records, attempts_by_id),
        routing=summarize_routing(records),
        latency_histogram=build_latency_histogram(records),
        slowest=attach_attempts(slowest_answered(records), attempts_by_id),
        attempts_by_model=summarize_attempts_by_model(attempts_by_id),
    )


def wasted_attempts(attempts_by_id: AttemptsById) -> list[AttemptRecord]:
    """Every attempt but the last is wasted; the last one too when it failed."""
    wasted: list[AttemptRecord] = []
    for attempts in attempts_by_id.values():
        if not attempts:
            continue
        wasted.extend(attempts[:-1])
        if attempts[-1].status == "error":
            wasted.append(attempts[-1])
    return wasted


def model_attempt_stats(attempts: list[AttemptRecord]) -> ModelAttemptStats:
    """Total up a list of attempts made on one model."""
    return ModelAttemptStats(
        attempts=len(attempts),
        answered=sum(1 for attempt in attempts if attempt.status in ANSWERED_STATUSES),
        truncated=sum(1 for attempt in attempts if attempt.status == "truncated"),
        errors=sum(1 for attempt in attempts if attempt.status == "error"),
        input_tokens=sum(attempt.input_tokens for attempt in attempts),
        output_tokens=sum(attempt.output_tokens for attempt in attempts),
        cost_usd=sum(attempt.cost_usd for attempt in attempts),
    )


def summarize_attempts_by_model(attempts_by_id: AttemptsById) -> dict[str, ModelAttemptStats]:
    """Group every attempt by its model and total each group, sorted by model name."""
    by_model: dict[str, list[AttemptRecord]] = {}
    for attempts in attempts_by_id.values():
        for attempt in attempts:
            by_model.setdefault(attempt.model_name, []).append(attempt)
    return {name: model_attempt_stats(by_model[name]) for name in sorted(by_model)}


def newest_first(records: list[RequestRecord]) -> list[RequestRecord]:
    """Sort rows newest first (by created_at, then id)."""
    return sorted(records, key=lambda record: (record.created_at, record.id or 0), reverse=True)


def summarize_failures(
    records: list[RequestRecord], attempts_by_id: AttemptsById
) -> FailureSummary:
    """Count failed rows by type, sum what they and wasted attempts cost, list recent ones."""
    failed = [record for record in records if record.status == "error"]
    by_type = dict.fromkeys(FAILURE_TYPES, 0)
    for record in failed:
        by_type[classify_failure(record.error_message)] += 1
    recent = [
        FailedRequest(
            record=record,
            attempts=attempts_of(record, attempts_by_id),
            failure_type=classify_failure(record.error_message),
        )
        for record in newest_first(failed)[:INSIGHT_LIST_LIMIT]
    ]
    wasted = wasted_attempts(attempts_by_id)
    return FailureSummary(
        by_type=by_type,
        errors=len(failed),
        truncated=sum(1 for record in records if record.status == "truncated"),
        failed_cost_usd=sum(record.cost_usd for record in failed),
        failed_input_tokens=sum(record.input_tokens for record in failed),
        failed_output_tokens=sum(record.output_tokens for record in failed),
        wasted_cost_usd=sum(attempt.cost_usd for attempt in wasted),
        wasted_tokens=sum(attempt.input_tokens + attempt.output_tokens for attempt in wasted),
        recent=recent,
    )


def count_by_model(records: list[RequestRecord]) -> dict[str, int]:
    """Count rows per final model, sorted by model name."""
    return dict(sorted(Counter(record.model_name for record in records).items()))


def summarize_routing(records: list[RequestRecord]) -> RoutingSummary:
    """Split "auto" and explicit requests by final model, and count fallbacks and rescues."""
    auto = [record for record in records if record.requested_model == AUTO_MODEL]
    explicit = [record for record in records if record.requested_model not in (AUTO_MODEL, "")]
    fallbacks = [record for record in records if record.fallback_used]
    return RoutingSummary(
        auto_requests=len(auto),
        auto_by_model=count_by_model(auto),
        explicit_by_model=count_by_model(explicit),
        fallbacks=len(fallbacks),
        fallbacks_rescued=sum(1 for record in fallbacks if record.status in ANSWERED_STATUSES),
    )


def latency_bin(latency_ms: float) -> int:
    """Return the histogram bin index; a value on an edge goes into the higher bin."""
    return bisect.bisect_right(LATENCY_BIN_EDGES_MS, latency_ms)


def build_latency_histogram(records: list[RequestRecord]) -> LatencyHistogram:
    """Count answered rows per latency range for each model."""
    by_model: dict[str, list[int]] = {}
    for record in records:
        if record.status not in ANSWERED_STATUSES:
            continue
        counts = by_model.setdefault(record.model_name, [0] * len(LATENCY_BIN_LABELS))
        counts[latency_bin(record.latency_ms)] += 1
    return LatencyHistogram(
        bin_labels=list(LATENCY_BIN_LABELS), by_model=dict(sorted(by_model.items()))
    )


def slowest_answered(records: list[RequestRecord]) -> list[RequestRecord]:
    """Return the answered rows with the highest latency, slowest first."""
    answered = [record for record in records if record.status in ANSWERED_STATUSES]
    answered.sort(key=lambda record: record.latency_ms, reverse=True)
    return answered[:INSIGHT_LIST_LIMIT]
