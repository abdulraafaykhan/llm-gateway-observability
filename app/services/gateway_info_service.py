"""Read-only information for the Gateway page: settings, live model health, route preview."""

import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.config import Settings
from app.models.completion import Message
from app.models.request_record import AttemptRecord
from app.models.routing import RoutingConfig
from app.repositories.base import RequestRepository
from app.routing.base import Router
from app.services.failure_types import classify_failure
from app.services.model_registry import ModelRegistry
from app.services.stats_service import ANSWERED_STATUSES, percentile
from app.views.gateway_view import (
    GatewayConfigResponse,
    ModelConfigView,
    ModelHealthView,
    RoutePreviewResponse,
)

DEGRADED_ERROR_RATE_PERCENT: float = 10.0
HEALTHY = "Healthy"
DEGRADED = "Degraded"
NO_TRAFFIC = "No recent traffic"


class GatewayInfoService:
    """Builds the Gateway page data; it never writes anything and never calls a model."""

    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry,
        routing_config: RoutingConfig,
        router: Router,
        repository: RequestRepository,
    ) -> None:
        self._settings: Settings = settings
        self._registry: ModelRegistry = registry
        self._routing_config: RoutingConfig = routing_config
        self._router: Router = router
        self._repository: RequestRepository = repository

    def config(self) -> GatewayConfigResponse:
        """Return an explicit list of non-secret settings (no key, no paths)."""
        settings = self._settings
        models = [
            ModelConfigView(**self._registry.get(name).model_dump())
            for name in self._registry.names()
        ]
        return GatewayConfigResponse(
            mode=settings.provider_mode,
            default_model_name=settings.default_model_name,
            max_output_tokens=settings.max_output_tokens,
            temperature=settings.temperature,
            request_timeout_seconds=settings.request_timeout_seconds,
            database_file=Path(settings.database_path).name,
            models=models,
            routing=self._routing_config,
        )

    async def model_health(self, minutes: int) -> list[ModelHealthView]:
        """Return one entry per configured model from the attempts in the last N minutes."""
        since = (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat()
        records = await self._repository.list_all(since)
        ids = [record.id for record in records if record.id is not None]
        by_model: dict[str, list[AttemptRecord]] = {}
        for attempts in (await self._repository.list_attempts(ids)).values():
            for attempt in attempts:
                by_model.setdefault(attempt.model_name, []).append(attempt)
        return [health_of(name, by_model.get(name, [])) for name in self._registry.names()]

    def preview(self, question: str) -> RoutePreviewResponse:
        """Run only the routing rules on one user message."""
        decision = self._router.choose([Message(role="user", content=question)])
        fired = {hit.rule for hit in decision.hits}
        return RoutePreviewResponse(
            model=decision.model_name,
            score=decision.score,
            threshold=decision.threshold,
            hits=decision.hits,
            reason=decision.reason,
            not_fired=[name for name in self._router.rule_names() if name not in fired],
        )


def health_word(attempts: int, error_rate_percent: float, rate_limit_errors: int) -> str:
    """Return "No recent traffic", "Degraded", or "Healthy"."""
    if attempts == 0:
        return NO_TRAFFIC
    if error_rate_percent >= DEGRADED_ERROR_RATE_PERCENT or rate_limit_errors > 0:
        return DEGRADED
    return HEALTHY


def health_of(name: str, attempts: list[AttemptRecord]) -> ModelHealthView:
    """Summarize one model's attempts."""
    answered = [attempt for attempt in attempts if attempt.status in ANSWERED_STATUSES]
    failed = [attempt for attempt in attempts if attempt.status == "error"]
    rate_limited = sum(
        1 for attempt in failed if classify_failure(attempt.error_message) == "rate_limit"
    )
    error_rate = len(failed) / len(attempts) * 100 if attempts else 0.0
    latencies = [attempt.latency_ms for attempt in answered]
    return ModelHealthView(
        name=name,
        attempts=len(attempts),
        answered=len(answered),
        truncated=sum(1 for attempt in attempts if attempt.status == "truncated"),
        errors=len(failed),
        rate_limit_errors=rate_limited,
        error_rate_percent=error_rate,
        p50_ms=percentile(latencies, 50) if latencies else None,
        p95_ms=percentile(latencies, 95) if latencies else None,
        avg_output_tokens=(
            statistics.fmean(attempt.output_tokens for attempt in answered) if answered else None
        ),
        cost_usd=sum(attempt.cost_usd for attempt in attempts),
        health=health_word(len(attempts), error_rate, rate_limited),
    )
