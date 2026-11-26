"""Runs a chat request against the chosen model and records it."""

import logging
import time
from datetime import UTC, datetime

from app.models.completion import CompletionResult, Message
from app.models.model_spec import ModelSpec
from app.models.request_record import RequestRecord
from app.providers.base import LLMProvider, ProviderError
from app.repositories.base import RequestRepository
from app.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ChatService:
    """Looks up the model, calls the provider, measures latency, and records the request."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: ModelRegistry,
        repository: RequestRepository,
        default_model_name: str,
    ) -> None:
        self._provider: LLMProvider = provider
        self._registry: ModelRegistry = registry
        self._repository: RequestRepository = repository
        self._default_model_name: str = default_model_name

    async def chat(
        self, messages: list[Message], model_name: str | None
    ) -> tuple[CompletionResult, ModelSpec, float]:
        """Answer the messages and return the result, the model used, and latency in ms."""
        model_spec = self._registry.get(model_name or self._default_model_name)
        created_at = datetime.now(UTC).isoformat()
        question = _question_text(messages)
        started = time.perf_counter()
        try:
            result = await self._provider.complete(model_spec, messages)
        except ProviderError as error:
            latency_ms = _elapsed_ms(started)
            logger.error("Provider call failed for model %s: %s", model_spec.name, error)
            await self._record(_error_record(model_spec, created_at, question, error, latency_ms))
            raise
        latency_ms = _elapsed_ms(started)
        _log_success(model_spec, result, latency_ms)
        await self._record(_success_record(model_spec, created_at, question, result, latency_ms))
        return result, model_spec, latency_ms

    async def _record(self, record: RequestRecord) -> None:
        """Save the record; log and continue if saving fails, so the caller's answer is kept."""
        try:
            await self._repository.add(record)
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


def _success_record(
    model_spec: ModelSpec,
    created_at: str,
    question: str,
    result: CompletionResult,
    latency_ms: float,
) -> RequestRecord:
    """Build the record for a call that returned an answer."""
    return RequestRecord(
        created_at=created_at,
        model_name=model_spec.name,
        model_id=model_spec.model_id,
        question=question,
        answer=result.text,
        status="truncated" if result.truncated else "ok",
        error_type=None,
        error_message=None,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=latency_ms,
        cost_usd=model_spec.cost_usd(result.input_tokens, result.output_tokens),
    )


def _error_record(
    model_spec: ModelSpec,
    created_at: str,
    question: str,
    error: ProviderError,
    latency_ms: float,
) -> RequestRecord:
    """Build the record for a call that failed at the provider."""
    return RequestRecord(
        created_at=created_at,
        model_name=model_spec.name,
        model_id=model_spec.model_id,
        question=question,
        answer=None,
        status="error",
        error_type="provider_error",
        error_message=str(error),
        input_tokens=error.input_tokens,
        output_tokens=error.output_tokens,
        latency_ms=latency_ms,
        cost_usd=model_spec.cost_usd(error.input_tokens, error.output_tokens),
    )
