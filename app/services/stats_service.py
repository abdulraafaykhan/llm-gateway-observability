"""Computes cost, failure, and latency statistics from recorded requests."""

import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime

from app.models.request_record import RequestRecord
from app.models.stats import GroupStats, LatencyStats, StatsReport
from app.repositories.base import RequestRepository

ANSWERED_STATUSES: frozenset[str] = frozenset({"ok", "truncated"})


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank: sort the list, read the value at position ceil(pct*n/100)."""
    if not values:
        raise ValueError("percentile() needs at least one value")
    sorted_values = sorted(values)
    rank = max(math.ceil(pct * len(sorted_values) / 100), 1)
    return sorted_values[rank - 1]


def compute_latency_stats(latencies: list[float]) -> LatencyStats | None:
    """Summarize latencies, or return None when there are none."""
    if not latencies:
        return None
    return LatencyStats(
        count=len(latencies),
        avg_ms=statistics.fmean(latencies),
        p50_ms=percentile(latencies, 50),
        p95_ms=percentile(latencies, 95),
        p99_ms=percentile(latencies, 99),
        max_ms=max(latencies),
    )


def compute_group_stats(records: list[RequestRecord]) -> GroupStats:
    """Compute counts, token sums, cost, and answered-only latency for a group of records."""
    answered_records = [record for record in records if record.status in ANSWERED_STATUSES]
    requests = len(records)
    answered = len(answered_records)
    errors = sum(1 for record in records if record.status == "error")
    total_cost_usd = sum(record.cost_usd for record in records)
    return GroupStats(
        requests=requests,
        answered=answered,
        truncated=sum(1 for record in records if record.status == "truncated"),
        errors=errors,
        error_rate_percent=errors / requests * 100 if requests else 0.0,
        input_tokens=sum(record.input_tokens for record in records),
        output_tokens=sum(record.output_tokens for record in records),
        total_cost_usd=total_cost_usd,
        cost_per_1000_answered_usd=total_cost_usd / answered * 1000 if answered else 0.0,
        latency=compute_latency_stats([record.latency_ms for record in answered_records]),
        fallbacks=sum(1 for record in records if record.fallback_used),
    )


class StatsService:
    """Builds a stats report from the request repository."""

    def __init__(self, repository: RequestRepository) -> None:
        self._repository: RequestRepository = repository

    async def build_report(self, since: str | None) -> StatsReport:
        """Load records (optionally since a UTC ISO time) and compute overall and per-model."""
        records = await self._repository.list_all(since)
        records_by_model: dict[str, list[RequestRecord]] = defaultdict(list)
        for record in records:
            records_by_model[record.model_name].append(record)
        return StatsReport(
            generated_at=datetime.now(UTC).isoformat(),
            since=since,
            overall=compute_group_stats(records),
            by_model={
                name: compute_group_stats(records_by_model[name])
                for name in sorted(records_by_model)
            },
        )
