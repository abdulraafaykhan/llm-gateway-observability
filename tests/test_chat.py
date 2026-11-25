"""Tests for POST /v1/chat and the request log (no network, no API key)."""

from pathlib import Path

import httpx

from app.dependencies import get_chat_service
from app.main import create_app
from app.models.completion import CompletionResult, Message
from app.models.model_spec import ModelSpec
from app.models.request_record import RequestRecord
from app.providers.base import LLMProvider, ProviderError
from app.repositories.base import RequestRepository
from app.repositories.sqlite_repository import SqliteRequestRepository
from app.services.chat_service import ChatService
from app.services.model_registry import ModelRegistry

MODELS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.toml"
VALID_BODY = {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
PRICED_MODELS_TOML = """
[models.big]
provider = "groq"
model_id = "test-big-model"
input_price_per_million_tokens = 0.5
output_price_per_million_tokens = 1.5
"""


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


class InMemoryRequestRepository(RequestRepository):
    """Repository that keeps records in a list."""

    def __init__(self) -> None:
        self.records: list[RequestRecord] = []

    async def initialize(self) -> None:
        """Nothing to create for an in-memory list."""

    async def add(self, record: RequestRecord) -> None:
        """Append the record to the list."""
        self.records.append(record)

    async def list_recent(self, limit: int) -> list[RequestRecord]:
        """Return up to limit records, newest first."""
        return list(reversed(self.records))[:limit]


def build_client(
    provider: LLMProvider,
    repository: RequestRepository | None = None,
    registry: ModelRegistry | None = None,
) -> httpx.AsyncClient:
    """Build an HTTP client for an app whose ChatService uses the given fakes."""
    app = create_app()
    chat_registry = registry or ModelRegistry(MODELS_CONFIG_PATH)
    chat_repository = repository or InMemoryRequestRepository()

    def get_fake_chat_service() -> ChatService:
        """Return a ChatService wired with the fakes."""
        return ChatService(
            provider=provider,
            registry=chat_registry,
            repository=chat_repository,
            default_model_name="big",
        )

    app.dependency_overrides[get_chat_service] = get_fake_chat_service
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def make_record(created_at: str, question: str) -> RequestRecord:
    """Build a successful record for repository tests."""
    return RequestRecord(
        created_at=created_at,
        model_name="big",
        model_id="test-big-model",
        question=question,
        answer="Paris",
        status="ok",
        error_type=None,
        error_message=None,
        input_tokens=12,
        output_tokens=3,
        latency_ms=150.5,
        cost_usd=0.0000105,
    )


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


async def test_successful_chat_records_one_ok_row(tmp_path: Path) -> None:
    """A successful call leaves one ok record with tokens, latency, and cost."""
    config_path = tmp_path / "models.toml"
    config_path.write_text(PRICED_MODELS_TOML, encoding="utf-8")
    registry = ModelRegistry(config_path)
    repository = InMemoryRequestRepository()
    async with build_client(FakeProvider(), repository, registry) as client:
        response = await client.post("/v1/chat", json=VALID_BODY)
    assert response.status_code == 200
    assert len(repository.records) == 1
    record = repository.records[0]
    assert record.status == "ok"
    assert record.model_name == "big"
    assert record.input_tokens == 12
    assert record.output_tokens == 3
    assert record.latency_ms >= 0
    assert record.cost_usd == registry.get("big").cost_usd(12, 3)
    assert record.cost_usd > 0


async def test_provider_error_records_one_error_row() -> None:
    """A provider failure leaves one error record and still returns 502."""
    repository = InMemoryRequestRepository()
    async with build_client(FakeProvider(should_fail=True), repository) as client:
        response = await client.post("/v1/chat", json=VALID_BODY)
    assert response.status_code == 502
    assert len(repository.records) == 1
    record = repository.records[0]
    assert record.status == "error"
    assert record.error_type == "provider_error"
    assert record.answer is None


async def test_truncated_answer_records_one_truncated_row() -> None:
    """A cut-off answer leaves one truncated record and returns 200."""
    repository = InMemoryRequestRepository()
    async with build_client(FakeProvider(truncated=True), repository) as client:
        response = await client.post("/v1/chat", json=VALID_BODY)
    assert response.status_code == 200
    assert len(repository.records) == 1
    assert repository.records[0].status == "truncated"


async def test_sqlite_repository_round_trip(tmp_path: Path) -> None:
    """SQLite stores records and lists them newest first."""
    repository = SqliteRequestRepository(tmp_path / "data" / "gateway.db")
    await repository.initialize()
    await repository.add(make_record("2026-11-25T10:00:00+00:00", "first"))
    await repository.add(make_record("2026-11-25T10:00:01+00:00", "second"))
    records = await repository.list_recent(10)
    assert [record.question for record in records] == ["second", "first"]
    assert all(record.id is not None for record in records)
    assert records[0].cost_usd == 0.0000105
