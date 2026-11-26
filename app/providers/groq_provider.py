"""Groq implementation of the provider interface."""

from typing import Any

import groq
from groq import AsyncGroq

from app.models.completion import CompletionResult, Message
from app.models.model_spec import ModelSpec
from app.providers.base import LLMProvider, ProviderError


class GroqProvider(LLMProvider):
    """Sends chat requests to Groq using the official AsyncGroq client."""

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float,
        max_output_tokens: int,
        temperature: float,
    ) -> None:
        self._client: AsyncGroq = AsyncGroq(api_key=api_key)
        self._timeout_seconds: float = timeout_seconds
        self._max_output_tokens: int = max_output_tokens
        self._temperature: float = temperature

    async def complete(self, model_spec: ModelSpec, messages: list[Message]) -> CompletionResult:
        """Send the messages to Groq and return the answer and token counts."""
        try:
            response = await self._client.chat.completions.create(
                model=model_spec.model_id,
                messages=[message.model_dump() for message in messages],
                temperature=self._temperature,
                max_tokens=self._max_output_tokens,
                timeout=self._timeout_seconds,
            )
        except groq.APIStatusError as error:
            detail = _status_error_detail(error)
            raise ProviderError(f"Groq API error (HTTP {error.status_code}): {detail}") from error
        except groq.APITimeoutError as error:
            message = f"Groq request timed out after {self._timeout_seconds} seconds"
            raise ProviderError(message) from error
        except groq.APIConnectionError as error:
            raise ProviderError("Could not connect to Groq") from error
        except groq.GroqError as error:
            raise ProviderError(f"Groq request failed: {error}") from error
        return _to_completion_result(response)


def _to_completion_result(response: Any) -> CompletionResult:
    """Turn a Groq chat completion into a CompletionResult, rejecting empty answers."""
    choice = response.choices[0] if response.choices else None
    text = choice.message.content if choice is not None else None
    if not text or not text.strip():
        raise _empty_answer_error(response, choice)
    if response.usage is None:
        raise ProviderError("Groq response did not include token usage")
    return CompletionResult(
        text=text,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        truncated=choice.finish_reason == "length",
    )


def _empty_answer_error(response: Any, choice: Any) -> ProviderError:
    """Build the empty-answer error with the real stop reason and token counts from Groq."""
    finish_reason = choice.finish_reason if choice is not None else None
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage is not None else 0
    output_tokens = usage.completion_tokens if usage is not None else 0
    output_text = str(output_tokens) if usage is not None else "unknown"
    return ProviderError(
        f"Groq returned an empty answer (finish_reason={finish_reason}, "
        f"output_tokens={output_text})",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=finish_reason,
    )


def _status_error_detail(error: groq.APIStatusError) -> str:
    """Pull the short error message out of a Groq HTTP error body."""
    body = error.body
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and isinstance(inner.get("message"), str):
            return inner["message"]
    return error.message
