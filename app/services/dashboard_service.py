"""Read-only data for the dashboard: lists database files and summarizes their request logs."""

import math
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.models.dashboard import SourceInfo, TimeBucket, TimeseriesResult
from app.models.request_record import RequestRecord
from app.models.stats import StatsReport
from app.repositories.base import RequestRepository
from app.services.stats_service import ANSWERED_STATUSES, StatsService, percentile

SOURCE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+\.db$")
BUCKET_SIZES_SECONDS: tuple[int, ...] = (10, 30, 60, 300, 900, 3600, 21600)
MAX_BUCKETS: int = 120

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

    async def recent(
        self, source: str | None, limit: int, minutes: int | None
    ) -> list[RequestRecord]:
        """Return up to limit records, newest first."""
        repository = self._repository(source)
        if minutes is None:
            return await repository.list_recent(limit)
        records = await repository.list_all(since_iso(utc_now(), minutes))
        newest = sorted(records, key=lambda record: (record.created_at, record.id or 0))
        return list(reversed(newest))[:limit]

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
