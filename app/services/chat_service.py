"""Runs a chat request: picks or routes the model, falls back once if needed, and records it."""

import logging
import time
from datetime import UTC, datetime
from typing import Literal, NamedTuple

from app.models.chat_outcome import ChatOutcome
from app.models.completion import CompletionResult, Message
from app.models.model_spec import ModelSpec
from app.models.request_record import AttemptRecord, RequestRecord
from app.models.routing import RoutingConfig
from app.providers.base import LLMProvider, ProviderError
from app.repositories.base import RequestRepository
from app.routing.base import Router
from app.services.model_registry import ModelRegistry, UnknownModelError

logger = logging.getLogger(__name__)

AUTO_MODEL: str = "auto"
EXPLICIT_REASON: str = "explicit"


class Attempt(NamedTuple):
    """One provider call: the model used, its result or error, and what it cost."""

    model_spec: ModelSpec
    result: CompletionResult | None
    error: ProviderError | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


class ChatService:
    """Chooses the model (routing "auto"), calls the provider, and records one row per request."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: ModelRegistry,
        repository: RequestRepository,
        router: Router,
        routing_config: RoutingConfig,
        default_model_name: str,
    ) -> None:
        self._provider: LLMProvider = provider
        self._registry: ModelRegistry = registry
        self._repository: RequestRepository = repository
        self._router: Router = router
        self._routing_config: RoutingConfig = routing_config
        self._default_model_name: str = default_model_name

    async def chat(self, messages: list[Message], model_name: str | None) -> ChatOutcome:
        """Answer the messages; raise ProviderError if the final attempt failed."""
        requested = model_name or self._default_model_name
        model_spec, route_reason = self._choose_model(messages, requested)
        created_at = datetime.now(UTC).isoformat()
        started = time.perf_counter()
        attempts = [await self._attempt(model_spec, messages)]
        if self._should_fall_back(requested, attempts[0]):
            attempts.append(await self._fall_back(attempts[0], messages))
        record = _build_record(
            attempts,
            created_at=created_at,
            question=_question_text(messages),
            latency_ms=_elapsed_ms(started),
            requested_model=requested,
            route_reason=route_reason,
        )
        attempt_records = _attempt_records(attempts)
        await self._record(record, attempt_records)
        final_error = attempts[-1].error
        if final_error is not None:
            raise final_error
        return _build_outcome(record, attempt_records)

    def _choose_model(self, messages: list[Message], requested: str) -> tuple[ModelSpec, str]:
        """Return the model to try first and the reason ("explicit" or the router's reason)."""
        if requested == AUTO_MODEL:
            decision = self._router.choose(messages)
            logger.info("Routed to %s: %s", decision.model_name, decision.reason)
            return self._registry.get(decision.model_name), decision.reason
        if requested not in self._registry.names():
            raise UnknownModelError(requested, [*self._registry.names(), AUTO_MODEL])
        return self._registry.get(requested), EXPLICIT_REASON

    async def _attempt(self, model_spec: ModelSpec, messages: list[Message]) -> Attempt:
        """Call the provider once, timing only this call, and price the tokens it used."""
        started = time.perf_counter()
        try:
            result = await self._provider.complete(model_spec, messages)
        except ProviderError as error:
            latency_ms = _elapsed_ms(started)
            logger.error("Provider call failed for model %s: %s", model_spec.name, error)
            cost = model_spec.cost_usd(error.input_tokens, error.output_tokens)
            return Attempt(
                model_spec, None, error, error.input_tokens, error.output_tokens, latency_ms, cost
            )
        latency_ms = _elapsed_ms(started)
        _log_success(model_spec, result, latency_ms)
        cost = model_spec.cost_usd(result.input_tokens, result.output_tokens)
        return Attempt(
            model_spec, result, None, result.input_tokens, result.output_tokens, latency_ms, cost
        )

    def _should_fall_back(self, requested: str, first: Attempt) -> bool:
        """Fall back for routed requests whose first model failed (or was cut off, if enabled)."""
        config = self._routing_config
        if requested != AUTO_MODEL or not config.fallback_enabled:
            return False
        if first.model_spec.name == config.fallback_model:
            return False
        if first.error is not None:
            return True
        cut_off = first.result is not None and first.result.truncated
        return cut_off and config.fallback_on_truncated

    async def _fall_back(self, first: Attempt, messages: list[Message]) -> Attempt:
        """Log why, then make the single fallback attempt on the fallback model."""
        fallback_spec = self._registry.get(self._routing_config.fallback_model)
        why = "failed" if first.error is not None else "was cut off"
        logger.warning(
            "Model %s %s; falling back to %s", first.model_spec.name, why, fallback_spec.name
        )
        return await self._attempt(fallback_spec, messages)

    async def _record(self, record: RequestRecord, attempts: list[AttemptRecord]) -> None:
        """Save the request and its attempts; log and continue if saving fails."""
        try:
            await self._repository.add(record, attempts)
        except Exception:
            logger.exception("Failed to record request for model %s", record.model_name)


def _question_text(messages: list[Message]) -> str:
    """Return the last user message's content, or the last message's content if none."""
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return messages[-1].content


def _elapsed_ms(started: float) -> float:
    """Return milliseconds since the given perf_counter value."""
    return (time.perf_counter() - started) * 1000


def _log_success(model_spec: ModelSpec, result: CompletionResult, latency_ms: float) -> None:
    """Log one INFO line for a completed call, plus a WARNING if it was cut off."""
    logger.info(
        "Chat completed: model=%s input_tokens=%d output_tokens=%d latency_ms=%.1f",
        model_spec.name,
        result.input_tokens,
        result.output_tokens,
        latency_ms,
    )
    if result.truncated:
        logger.warning(
            "Answer was cut off at the token limit: model=%s output_tokens=%d",
            model_spec.name,
            result.output_tokens,
        )


def _build_record(
    attempts: list[Attempt],
    *,
    created_at: str,
    question: str,
    latency_ms: float,
    requested_model: str,
    route_reason: str,
) -> RequestRecord:
    """Build the single row for a request: the final model, plus totals over all attempts."""
    final = attempts[-1]
    result, error = final.result, final.error
    return RequestRecord(
        created_at=created_at,
        model_name=final.model_spec.name,
        model_id=final.model_spec.model_id,
        question=question,
        answer=result.text if result is not None else None,
        status=_attempt_status(final),
        error_type="provider_error" if error is not None else None,
        error_message=str(error) if error is not None else None,
        input_tokens=sum(attempt.input_tokens for attempt in attempts),
        output_tokens=sum(attempt.output_tokens for attempt in attempts),
        latency_ms=latency_ms,
        cost_usd=sum(attempt.cost_usd for attempt in attempts),
        requested_model=requested_model,
        route_reason=route_reason,
        fallback_used=len(attempts) > 1,
    )


def _attempt_status(attempt: Attempt) -> Literal["ok", "truncated", "error"]:
    """Return "error", "truncated", or "ok" for one attempt."""
    if attempt.error is not None:
        return "error"
    return "truncated" if attempt.result is not None and attempt.result.truncated else "ok"


def _attempt_record(number: int, attempt: Attempt) -> AttemptRecord:
    """Build the stored record of one attempt."""
    error = attempt.error
    return AttemptRecord(
        attempt=number,
        model_name=attempt.model_spec.name,
        model_id=attempt.model_spec.model_id,
        status=_attempt_status(attempt),
        error_type="provider_error" if error is not None else None,
        error_message=str(error) if error is not None else None,
        input_tokens=attempt.input_tokens,
        output_tokens=attempt.output_tokens,
        latency_ms=attempt.latency_ms,
        cost_usd=attempt.cost_usd,
    )


def _attempt_records(attempts: list[Attempt]) -> list[AttemptRecord]:
    """Number the attempts from 1 and build their records."""
    return [_attempt_record(number, attempt) for number, attempt in enumerate(attempts, start=1)]


def _build_outcome(record: RequestRecord, attempts: list[AttemptRecord]) -> ChatOutcome:
    """Turn a successful request's record and attempts into the outcome for the caller."""
    return ChatOutcome(
        attempts=attempts,
        answer=record.answer or "",
        truncated=record.status == "truncated",
        model_name=record.model_name,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        latency_ms=record.latency_ms,
        cost_usd=record.cost_usd,
        requested_model=record.requested_model,
        route_reason=record.route_reason,
        fallback_used=record.fallback_used,
    )
