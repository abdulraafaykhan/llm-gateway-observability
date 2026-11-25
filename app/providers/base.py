"""The provider interface every AI company adapter implements."""

from abc import ABC, abstractmethod

from app.models.completion import CompletionResult, Message
from app.models.model_spec import ModelSpec


class ProviderError(Exception):
    """Any failure from a provider, including timeouts and a bad API key."""


class LLMProvider(ABC):
    """Interface for sending chat messages to a model and getting an answer."""

    @abstractmethod
    async def complete(self, model_spec: ModelSpec, messages: list[Message]) -> CompletionResult:
        """Send the messages to the model and return its answer and token counts."""
