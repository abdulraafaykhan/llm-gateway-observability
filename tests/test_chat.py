"""Tests for POST /v1/chat using a fake provider (no network, no API key)."""

from pathlib import Path

import httpx

from app.dependencies import get_chat_service
from app.main import create_app
from app.models.completion import CompletionResult, Message
from app.models.model_spec import ModelSpec
from app.providers.base import LLMProvider, ProviderError
from app.services.chat_service import ChatService
from app.services.model_registry import ModelRegistry

MODELS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.toml"
VALID_BODY = {"messages": [{"role": "user", "content": "What is the capital of France?"}]}


class FakeProvider(LLMProvider):
    """Provider that returns a fixed answer, or raises ProviderError when told to."""

    def __init__(self, should_fail: bool = False, truncated: bool = False) -> None:
        self.should_fail: bool = should_fail
        self.truncated: bool = truncated

    async def complete(self, model_spec: ModelSpec, messages: list[Message]) -> CompletionResult:
        """Return a fixed result or raise ProviderError."""
        if self.should_fail:
            raise ProviderError("Fake provider failure")
        return CompletionResult(
            text="Paris", input_tokens=12, output_tokens=3, truncated=self.truncated
        )


def build_client(provider: LLMProvider) -> httpx.AsyncClient:
    """Build an HTTP client for an app whose ChatService uses the given provider."""
    app = create_app()
    registry = ModelRegistry(MODELS_CONFIG_PATH)

    def get_fake_chat_service() -> ChatService:
        """Return a ChatService wired with the fake provider."""
        return ChatService(provider=provider, registry=registry, default_model_name="big")

    app.dependency_overrides[get_chat_service] = get_fake_chat_service
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_chat_returns_answer_with_all_fields() -> None:
    """A valid request returns 200 with all response fields."""
    async with build_client(FakeProvider()) as client:
        response = await client.post("/v1/chat", json=VALID_BODY)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "answer",
        "model",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "truncated",
    }
    assert body["answer"] == "Paris"
    assert body["model"] == "big"
    assert body["input_tokens"] == 12
    assert body["output_tokens"] == 3
    assert isinstance(body["latency_ms"], float)
    assert body["truncated"] is False


async def test_chat_with_truncated_answer_returns_200_and_flag() -> None:
    """A cut-off answer still returns 200, with truncated set to true."""
    async with build_client(FakeProvider(truncated=True)) as client:
        response = await client.post("/v1/chat", json=VALID_BODY)
    assert response.status_code == 200
    assert response.json()["truncated"] is True


async def test_chat_with_unknown_model_returns_400() -> None:
    """An unknown model name returns 400 with error type unknown_model."""
    async with build_client(FakeProvider()) as client:
        response = await client.post("/v1/chat", json={**VALID_BODY, "model": "nonexistent"})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "unknown_model"


async def test_chat_with_provider_error_returns_502() -> None:
    """A provider failure returns 502 with error type provider_error."""
    async with build_client(FakeProvider(should_fail=True)) as client:
        response = await client.post("/v1/chat", json=VALID_BODY)
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "provider_error"


async def test_chat_with_empty_messages_returns_422() -> None:
    """An empty messages list fails validation with 422."""
    async with build_client(FakeProvider()) as client:
        response = await client.post("/v1/chat", json={"messages": []})
    assert response.status_code == 422
