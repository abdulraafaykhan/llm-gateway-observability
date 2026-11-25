"""Runs a chat request against the chosen model."""

import logging
import time

from app.models.completion import CompletionResult, Message
from app.models.model_spec import ModelSpec
from app.providers.base import LLMProvider, ProviderError
from app.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ChatService:
    """Looks up the model, calls the provider, and measures latency."""

    def __init__(
        self, provider: LLMProvider, registry: ModelRegistry, default_model_name: str
    ) -> None:
        self._provider: LLMProvider = provider
        self._registry: ModelRegistry = registry
        self._default_model_name: str = default_model_name

    async def chat(
        self, messages: list[Message], model_name: str | None
    ) -> tuple[CompletionResult, ModelSpec, float]:
        """Answer the messages and return the result, the model used, and latency in ms."""
        model_spec = self._registry.get(model_name or self._default_model_name)
        started = time.perf_counter()
        try:
            result = await self._provider.complete(model_spec, messages)
        except ProviderError as error:
            logger.error("Provider call failed for model %s: %s", model_spec.name, error)
            raise
        latency_ms = (time.perf_counter() - started) * 1000
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
        return result, model_spec, latency_ms
