"""The provider interface every AI company adapter implements."""

from abc import ABC, abstractmethod

from app.models.completion import CompletionResult, Message
from app.models.model_spec import ModelSpec


class ProviderError(Exception):
    """Any failure from a provider, including timeouts and a bad API key."""

    def __init__(
        self,
        message: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        finish_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.input_tokens: int = input_tokens
        self.output_tokens: int = output_tokens
        self.finish_reason: str | None = finish_reason


class LLMProvider(ABC):
    """Interface for sending chat messages to a model and getting an answer."""

    @abstractmethod
    async def complete(self, model_spec: ModelSpec, messages: list[Message]) -> CompletionResult:
        """Send the messages to the model and return its answer and token counts."""
