"""Google Gemini implementation of the provider interface, using plain httpx."""

from typing import Any, NamedTuple

import httpx

from app.models.completion import CompletionResult, Message
from app.models.model_spec import ModelSpec
from app.providers.base import LLMProvider, ProviderError

GEMINI_API_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
API_KEY_HEADER: str = "x-goog-api-key"
ERROR_DETAIL_LIMIT: int = 300
FINISH_REASONS: dict[str, str] = {"STOP": "stop", "MAX_TOKENS": "length"}
GEMINI_ROLES: dict[str, str] = {"user": "user", "assistant": "model"}


class GeminiAnswer(NamedTuple):
    """One Gemini answer with its token counts and normalized finish reason."""

    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None


class GeminiProvider(LLMProvider):
    """Sends chat requests to the Gemini generateContent endpoint; the key goes in a header."""

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float,
        max_output_tokens: int,
        temperature: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key: str = api_key
        self._timeout_seconds: float = timeout_seconds
        self._max_output_tokens: int = max_output_tokens
        self._temperature: float = temperature
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=GEMINI_API_BASE_URL,
            headers={API_KEY_HEADER: api_key},
            timeout=timeout_seconds,
            transport=transport,
        )

    def build_body(self, messages: list[Message]) -> dict[str, Any]:
        """Return the generateContent body: system text, chat turns, and generation settings."""
        body: dict[str, Any] = {}
        system_texts = [message.content for message in messages if message.role == "system"]
        if system_texts:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_texts)}]}
        body["contents"] = [
            {"role": GEMINI_ROLES[message.role], "parts": [{"text": message.content}]}
            for message in messages
            if message.role != "system"
        ]
        body["generationConfig"] = {
            "maxOutputTokens": self._max_output_tokens,
            "temperature": self._temperature,
        }
        return body

    async def generate(self, model_id: str, messages: list[Message]) -> GeminiAnswer:
        """Send one generateContent request and return the answer, or raise ProviderError."""
        try:
            response = await self._client.post(
                f"/models/{model_id}:generateContent", json=self.build_body(messages)
            )
        except httpx.TimeoutException as error:
            message = f"Gemini request timed out after {self._timeout_seconds} seconds"
            raise ProviderError(message) from error
        except httpx.TransportError as error:
            raise ProviderError("Could not connect to Gemini") from error
        except httpx.HTTPError as error:
            raise ProviderError(f"Gemini request failed: {type(error).__name__}") from error
        if response.status_code != 200:
            detail = self._without_key(error_detail(response))[:ERROR_DETAIL_LIMIT]
            raise ProviderError(f"Gemini API error (HTTP {response.status_code}): {detail}")
        try:
            data = response.json()
        except ValueError as error:
            raise ProviderError("Gemini returned a response that is not valid JSON") from error
        return parse_generate_response(data)

    async def complete(self, model_spec: ModelSpec, messages: list[Message]) -> CompletionResult:
        """Send the messages to Gemini and return the answer and token counts."""
        answer = await self.generate(model_spec.model_id, messages)
        return CompletionResult(
            text=answer.text,
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            truncated=answer.finish_reason == "length",
        )

    def _without_key(self, text: str) -> str:
        """Remove the API key from text, in case an error body ever echoes it back."""
        return text.replace(self._api_key, "[hidden]") if self._api_key else text


def parse_generate_response(data: Any) -> GeminiAnswer:
    """Turn a generateContent JSON body into a GeminiAnswer, rejecting empty answers."""
    data = data if isinstance(data, dict) else {}
    usage = data.get("usageMetadata") or {}
    input_tokens = int(usage.get("promptTokenCount") or 0)
    output_tokens = int(usage.get("candidatesTokenCount") or 0) + int(
        usage.get("thoughtsTokenCount") or 0
    )
    candidates = data.get("candidates") or []
    candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    finish_reason = normalize_finish_reason(candidate.get("finishReason"))
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    if not text.strip():
        block_reason = (data.get("promptFeedback") or {}).get("blockReason")
        block_text = f", block_reason={block_reason}" if block_reason else ""
        raise ProviderError(
            f"Gemini returned an empty answer (finish_reason={finish_reason}, "
            f"output_tokens={output_tokens}{block_text})",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )
    return GeminiAnswer(text, input_tokens, output_tokens, finish_reason)


def normalize_finish_reason(reason: Any) -> str | None:
    """Map Gemini finish reasons to the gateway's: STOP is stop, MAX_TOKENS is length."""
    if not isinstance(reason, str) or not reason:
        return None
    return FINISH_REASONS.get(reason, reason.lower())


def error_detail(response: httpx.Response) -> str:
    """Pull error.message out of a Gemini error body, or fall back to the raw body text."""
    try:
        body = response.json()
    except ValueError:
        body = None
    detail = response.text
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict) and isinstance(inner.get("message"), str):
            detail = inner["message"]
    return detail
