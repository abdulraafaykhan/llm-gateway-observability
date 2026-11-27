"""Tests for POST /v1/chat and the request log (no network, no API key)."""

import asyncio
import itertools
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite
import httpx
import pytest

from app.cli import eval_cli
from app.clients.base import GatewayClient
from app.config import Settings
from app.dependencies import (
    ProviderSelector,
    build_llm_provider,
    get_chat_service,
    get_dashboard_service,
    get_demo_service,
    get_experiments_service,
    get_gateway_info_service,
    sqlite_repository_factory,
)
from app.main import create_app
from app.models.completion import CompletionResult, Message
from app.models.eval import EvalOutcome, EvalQuestion, EvalRunResult, GatewayReply
from app.models.fake_profile import FakeProfile
from app.models.model_spec import ModelSpec
from app.models.request_record import AttemptRecord, RequestRecord
from app.models.routing import RouteDecision, RoutingConfig
from app.providers import fake_provider
from app.providers.base import LLMProvider, ProviderError
from app.providers.gemini_provider import GeminiProvider
from app.providers.groq_provider import GroqProvider
from app.repositories.base import RequestRepository
from app.repositories.sqlite_repository import CREATE_TABLE_SQL, SqliteRequestRepository
from app.routing.base import Router
from app.routing.factory import build_rule_router
from app.services.cascade_replay import build_cascade_table, escalate
from app.services.chat_service import ChatService
from app.services.dashboard_service import DashboardService
from app.services.demo_service import DemoService
from app.services.eval_checker import check_answer
from app.services.eval_loader import load_eval_set
from app.services.eval_report import compare_runs, summarize
from app.services.eval_runner import EvalRunner
from app.services.experiments_service import ExperimentsService
from app.services.failure_types import classify_failure
from app.services.gateway_info_service import GatewayInfoService
from app.services.gemini_models_service import (
    FREE_TIER_REMINDER,
    PROBE_TEXT,
    fetch_models,
    generate_content_models,
    model_note,
    probe_model,
    render_models_table,
    render_probe_line,
)
from app.services.load_plan import build_schedule
from app.services.model_registry import ModelRegistry
from app.services.router_replay import build_replay_table, replay, replay_baselines
from app.services.routing_config import load_routing_config, validate_routing_models
from app.services.stats_service import StatsService, compute_group_stats, percentile
from app.views.cascade_text_view import render_cascade_tables
from app.views.eval_text_view import render_eval_summary
from app.views.stats_text_view import render_stats_report

MODELS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.toml"
ROUTING_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "routing.toml"
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

    def __init__(
        self,
        should_fail: bool = False,
        truncated: bool = False,
        error: ProviderError | None = None,
    ) -> None:
        self.should_fail: bool = should_fail
        self.truncated: bool = truncated
        self.error: ProviderError | None = error

    async def complete(self, model_spec: ModelSpec, messages: list[Message]) -> CompletionResult:
        """Return a fixed result or raise ProviderError (the given one, if any)."""
        if self.error is not None:
            raise self.error
        if self.should_fail:
            raise ProviderError("Fake provider failure")
        return CompletionResult(
            text="Paris", input_tokens=12, output_tokens=3, truncated=self.truncated
        )


class InMemoryRequestRepository(RequestRepository):
    """Repository that keeps records in a list."""

    def __init__(self) -> None:
        self.records: list[RequestRecord] = []
        self.attempts: dict[int, list[AttemptRecord]] = {}

    async def initialize(self) -> None:
        """Nothing to create for an in-memory list."""

    async def add(self, record: RequestRecord, attempts: list[AttemptRecord]) -> int:
        """Append the record with a new id, keep its attempts, and return the id."""
        request_id = len(self.records) + 1
        self.records.append(record.model_copy(update={"id": request_id}))
        if attempts:
            self.attempts[request_id] = [
                attempt.model_copy(update={"request_id": request_id}) for attempt in attempts
            ]
        return request_id

    async def list_attempts(self, request_ids: list[int]) -> dict[int, list[AttemptRecord]]:
        """Return the stored attempts of the given requests."""
        return {rid: self.attempts[rid] for rid in request_ids if rid in self.attempts}

    async def list_recent(self, limit: int) -> list[RequestRecord]:
        """Return up to limit records, newest first."""
        return list(reversed(self.records))[:limit]

    async def list_all(self, since: str | None = None) -> list[RequestRecord]:
        """Return all records, or those created at or after since, in insertion order."""
        if since is None:
            return list(self.records)
        return [record for record in self.records if record.created_at >= since]


def build_client(
    provider: LLMProvider,
    repository: RequestRepository | None = None,
    registry: ModelRegistry | None = None,
    router: Router | None = None,
    config: RoutingConfig | None = None,
) -> httpx.AsyncClient:
    """Build an HTTP client for an app whose ChatService uses the given fakes."""
    app = create_app()
    chat_registry = registry or ModelRegistry(MODELS_CONFIG_PATH)
    chat_repository = repository or InMemoryRequestRepository()
    routing_config = config or load_routing_config(str(ROUTING_CONFIG_PATH))
    chat_router = router or build_rule_router(routing_config)

    def get_fake_chat_service() -> ChatService:
        """Return a ChatService wired with the fakes."""
        return ChatService(
            provider=provider,
            registry=chat_registry,
            repository=chat_repository,
            router=chat_router,
            routing_config=routing_config,
            default_model_name="big",
        )

    app.dependency_overrides[get_chat_service] = get_fake_chat_service
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def make_record(**changes: Any) -> RequestRecord:
    """Build a successful record for tests; keyword arguments replace the defaults."""
    fields: dict[str, Any] = {
        "created_at": "2026-11-25T10:00:00+00:00",
        "model_name": "big",
        "model_id": "test-big-model",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "status": "ok",
        "error_type": None,
        "error_message": None,
        "input_tokens": 12,
        "output_tokens": 3,
        "latency_ms": 150.5,
        "cost_usd": 0.0000105,
    }
    fields.update(changes)
    return RequestRecord(**fields)


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
        "requested_model",
        "route_reason",
        "fallback_used",
        "cost_usd",
        "attempts",
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


async def test_empty_answer_error_records_tokens_and_cost() -> None:
    """A provider error carrying token counts is recorded with those tokens and their cost."""
    registry = ModelRegistry(MODELS_CONFIG_PATH)
    repository = InMemoryRequestRepository()
    error = ProviderError(
        "Groq returned an empty answer (finish_reason=length, output_tokens=4096)",
        input_tokens=50,
        output_tokens=4096,
        finish_reason="length",
    )
    async with build_client(FakeProvider(error=error), repository, registry) as client:
        response = await client.post("/v1/chat", json=VALID_BODY)
    assert response.status_code == 502
    assert len(repository.records) == 1
    record = repository.records[0]
    assert record.status == "error"
    assert record.error_message == str(error)
    assert (record.input_tokens, record.output_tokens) == (50, 4096)
    assert record.cost_usd == registry.get("big").cost_usd(50, 4096)
    assert record.cost_usd > 0


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
    await repository.add(make_record(created_at="2026-11-25T10:00:00+00:00", question="first"), [])
    await repository.add(make_record(created_at="2026-11-25T10:00:01+00:00", question="second"), [])
    records = await repository.list_recent(10)
    assert [record.question for record in records] == ["second", "first"]
    assert all(record.id is not None for record in records)
    assert records[0].cost_usd == 0.0000105


def test_percentile_uses_nearest_rank() -> None:
    """Nearest-rank percentiles on 1..100 (shuffled), a single value, and an empty list."""
    values = [float(value) for value in range(100, 0, -1)]
    values[10], values[60] = values[60], values[10]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0
    assert percentile(values, 99) == 99.0
    assert percentile(values, 100) == 100.0
    assert all(percentile([7.0], pct) == 7.0 for pct in (0, 50, 95, 99, 100))
    with pytest.raises(ValueError):
        percentile([], 50)


def test_compute_group_stats_counts_cost_and_answered_only_latency() -> None:
    """Counts, sums, and cost cover all rows; latency uses answered rows only."""
    records = [
        make_record(
            model_name="big", input_tokens=10, output_tokens=100, latency_ms=100.0, cost_usd=0.001
        ),
        make_record(
            model_name="big",
            status="truncated",
            input_tokens=20,
            output_tokens=200,
            latency_ms=300.0,
            cost_usd=0.002,
        ),
        make_record(
            model_name="big",
            status="error",
            answer=None,
            input_tokens=0,
            output_tokens=0,
            latency_ms=99999.0,
            cost_usd=0.0,
        ),
        make_record(
            model_name="small", input_tokens=5, output_tokens=50, latency_ms=200.0, cost_usd=0.0005
        ),
        make_record(
            model_name="small", input_tokens=5, output_tokens=50, latency_ms=400.0, cost_usd=0.0005
        ),
        make_record(
            model_name="small",
            status="error",
            answer=None,
            input_tokens=0,
            output_tokens=0,
            latency_ms=88888.0,
            cost_usd=0.0,
        ),
    ]
    stats = compute_group_stats(records)
    assert (stats.requests, stats.answered, stats.truncated, stats.errors) == (6, 4, 1, 2)
    assert stats.error_rate_percent == pytest.approx(2 / 6 * 100)
    assert (stats.input_tokens, stats.output_tokens) == (40, 400)
    assert stats.total_cost_usd == pytest.approx(0.004)
    assert stats.cost_per_1000_answered_usd == pytest.approx(1.0)
    assert stats.latency is not None
    assert stats.latency.count == 4
    assert stats.latency.avg_ms == pytest.approx(250.0)
    assert (stats.latency.p50_ms, stats.latency.p95_ms) == (200.0, 400.0)
    assert stats.latency.max_ms == 400.0


async def test_stats_service_splits_by_model_and_handles_empty() -> None:
    """The report splits records by model with sorted keys; no records means requests == 0."""
    repository = InMemoryRequestRepository()
    for model_name in ("small", "big", "small"):
        await repository.add(make_record(model_name=model_name), [])
    report = await StatsService(repository).build_report(since=None)
    assert report.overall.requests == 3
    assert list(report.by_model) == ["big", "small"]
    assert report.by_model["big"].requests == 1
    assert report.by_model["small"].requests == 2

    empty_report = await StatsService(InMemoryRequestRepository()).build_report(since=None)
    assert empty_report.overall.requests == 0
    assert render_stats_report(empty_report) == "No requests found."


async def test_sqlite_list_all_filters_by_since(tmp_path: Path) -> None:
    """list_all returns every row in id order, or only rows at or after since."""
    repository = SqliteRequestRepository(tmp_path / "gateway.db")
    await repository.initialize()
    created_times = [
        "2026-11-25T10:00:00+00:00",
        "2026-11-25T10:05:00+00:00",
        "2026-11-25T10:10:00+00:00",
    ]
    for index, created_at in enumerate(created_times):
        await repository.add(make_record(created_at=created_at, question=f"q{index}"), [])
    all_records = await repository.list_all()
    assert [record.question for record in all_records] == ["q0", "q1", "q2"]
    assert [record.id for record in all_records] == sorted(record.id for record in all_records)
    recent_records = await repository.list_all(since=created_times[1])
    assert [record.question for record in recent_records] == ["q1", "q2"]


FAKE_MODEL_SPEC = ModelSpec(
    name="big",
    provider="fake",
    model_id="fake-big",
    input_price_per_million_tokens=0.0,
    output_price_per_million_tokens=0.0,
)
FAKE_MESSAGES = [Message(role="user", content="What is the capital of France?")]


def make_fake_provider(seed: int = 42, **changes: Any) -> fake_provider.FakeProvider:
    """Build the simulated provider with one tiny-latency 'big' profile; kwargs override it."""
    fields: dict[str, Any] = {
        "base_latency_ms": 1,
        "ms_per_output_token": 0.0,
        "jitter_percent": 0,
        "slow_outlier_probability": 0.0,
        "slow_outlier_multiplier": 1.0,
        "error_probability": 0.0,
        "truncate_probability": 0.0,
        "min_output_tokens": 80,
        "max_output_tokens": 900,
        "max_concurrent_requests": 8,
    }
    fields.update(changes)
    return fake_provider.FakeProvider(profiles={"big": FakeProfile(**fields)}, seed=seed)


async def test_fake_provider_is_repeatable_for_a_seed() -> None:
    """The same seed gives the same 10 results; a different seed gives a different sequence."""

    async def run_ten(seed: int) -> list[tuple[int, bool]]:
        """Return (output_tokens, truncated) for 10 calls in order."""
        provider = make_fake_provider(seed, jitter_percent=15, truncate_probability=0.3)
        results = [await provider.complete(FAKE_MODEL_SPEC, FAKE_MESSAGES) for _ in range(10)]
        return [(result.output_tokens, result.truncated) for result in results]

    assert await run_ten(7) == await run_ten(7)
    assert await run_ten(7) != await run_ten(8)


async def test_fake_provider_queues_when_busy() -> None:
    """With 2 slots and 50 ms per call, 6 parallel calls take three waves (>= 0.13 s)."""
    provider = make_fake_provider(base_latency_ms=50, max_concurrent_requests=2)
    started = time.perf_counter()
    await asyncio.gather(*(provider.complete(FAKE_MODEL_SPEC, FAKE_MESSAGES) for _ in range(6)))
    assert time.perf_counter() - started >= 0.13


async def test_fake_provider_fails_and_cuts_off_when_told_to() -> None:
    """error_probability 1.0 raises ProviderError; truncate_probability 1.0 cuts the answer off."""
    with pytest.raises(ProviderError):
        await make_fake_provider(error_probability=1.0).complete(FAKE_MODEL_SPEC, FAKE_MESSAGES)
    result = await make_fake_provider(truncate_probability=1.0).complete(
        FAKE_MODEL_SPEC, FAKE_MESSAGES
    )
    assert result.truncated is True
    assert result.output_tokens == 900


def test_build_schedule_is_repeatable_and_sized_per_scenario() -> None:
    """Schedules are repeatable, ascending, and roughly the expected size per scenario."""
    spike = build_schedule("spike", 42)
    assert spike == build_schedule("spike", 42)
    assert spike == sorted(spike)
    assert 200 <= len(spike) <= 360
    assert spike[-1] < 60
    assert 10 <= len(build_schedule("smoke", 42)) <= 30
    assert 70 <= len(build_schedule("normal", 42)) <= 170
    with pytest.raises(ValueError):
        build_schedule("unknown", 42)


EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "eval" / "eval_set.jsonl"


def make_question(**changes: Any) -> EvalQuestion:
    """Build an eval question for tests; keyword arguments replace the defaults."""
    fields: dict[str, Any] = {
        "id": "q001",
        "question": "What is the capital of Japan?",
        "difficulty": "easy",
        "category": "fact",
        "split": "tune",
        "check": "contains_all",
        "expected": ["tokyo"],
    }
    fields.update(changes)
    return EvalQuestion(**fields)


def make_outcome(question_id: str, status: str, difficulty: str = "easy") -> EvalOutcome:
    """Build a graded outcome for comparison tests."""
    return EvalOutcome(
        **make_question(id=question_id, difficulty=difficulty).model_dump(),
        status=status,
        truncated=False,
        answer=None if status == "error" else "Tokyo",
        answered_by=None if status == "error" else "small",
        input_tokens=0,
        output_tokens=0,
        latency_ms=None,
        cost_usd=0.0,
        error_message="http 502: boom" if status == "error" else None,
        attempts=1,
    )


def make_run(label: str, outcomes: list[EvalOutcome]) -> EvalRunResult:
    """Wrap outcomes in a saved-run shape."""
    return EvalRunResult(
        label=label,
        model_requested=label,
        created_at="2026-11-26T08:00:00+00:00",
        split_filter="all",
        eval_set_path="eval/eval_set.jsonl",
        outcomes=outcomes,
    )


def test_check_answer_rules_and_normalization() -> None:
    """Text checks ignore case, markdown, and special hyphens; number checks are lenient."""
    any_question = make_question(check="contains_any", expected=["leonardo", "da vinci"])
    assert check_answer(any_question, "It was painted by Da Vinci.")
    assert not check_answer(any_question, "Michelangelo")
    all_question = make_question(expected=["new", "york"])
    assert check_answer(all_question, "NEW YORK")
    assert not check_answer(all_question, "New Jersey")
    assert check_answer(make_question(), "**Tokyo**")
    assert check_answer(make_question(expected=["x-ray"]), "An X\u2011ray")
    thousand = make_question(check="number", expected=["1234"])
    assert check_answer(thousand, "The total is 1,234 dollars.")
    assert check_answer(thousand, "The total is 1\u202f234.")
    decimal = make_question(check="number", expected=["12.5"])
    assert check_answer(decimal, "The answer is 12.50")
    assert not check_answer(decimal, "The answer is 13")
    assert not check_answer(decimal, "I am not sure.")


def test_load_eval_set_validates_lines(tmp_path: Path) -> None:
    """A valid file loads; repeated ids, unknown checks, and non-numeric keys are rejected."""
    lines = [
        make_question(id="q001").model_dump_json(),
        "",
        make_question(id="q002", check="number", expected=["391"]).model_dump_json(),
        make_question(id="q003", check="contains_any").model_dump_json(),
    ]
    valid_path = tmp_path / "valid.jsonl"
    valid_path.write_text("\n".join(lines), encoding="utf-8")
    assert [question.id for question in load_eval_set(str(valid_path))] == [
        "q001",
        "q002",
        "q003",
    ]
    bad_lines = {
        "repeated.jsonl": [lines[0], lines[0]],
        "unknown_check.jsonl": [lines[0].replace("contains_all", "regex")],
        "not_numeric.jsonl": [lines[2].replace('"391"', '"many"')],
    }
    for name, content in bad_lines.items():
        bad_path = tmp_path / name
        bad_path.write_text("\n".join(content), encoding="utf-8")
        with pytest.raises(ValueError):
            load_eval_set(str(bad_path))


def test_real_eval_set_has_balanced_cells() -> None:
    """The shipped eval set has 100 unique questions, 50 easy, 50 hard, 25 per cell."""
    questions = load_eval_set(str(EVAL_SET_PATH))
    assert len(questions) == 100
    assert len({question.id for question in questions}) == 100
    assert sum(1 for question in questions if question.difficulty == "easy") == 50
    assert sum(1 for question in questions if question.difficulty == "hard") == 50
    for difficulty in ("easy", "hard"):
        for split in ("tune", "test"):
            cell = [q for q in questions if q.difficulty == difficulty and q.split == split]
            assert len(cell) == 25, (difficulty, split)


class ScriptedGatewayClient(GatewayClient):
    """Gateway client that returns a fixed reply per question text and counts calls."""

    def __init__(self, replies: dict[str, GatewayReply]) -> None:
        self.replies: dict[str, GatewayReply] = replies
        self.calls: dict[str, int] = {}

    async def ask(self, question: str, model: str) -> GatewayReply:
        """Return the scripted reply for this question."""
        self.calls[question] = self.calls.get(question, 0) + 1
        return self.replies[question]


def make_reply(
    http_status: int = 200, answer: str | None = "Tokyo", **changes: Any
) -> GatewayReply:
    """Build a gateway reply for runner tests."""
    fields: dict[str, Any] = {
        "http_status": http_status,
        "answer": answer if http_status == 200 else None,
        "model": "small" if http_status == 200 else None,
        "input_tokens": 10,
        "output_tokens": 5,
        "latency_ms": 100.0,
        "truncated": False,
        "error_message": None if http_status == 200 else "boom",
        "cost_usd": 0.00002 if http_status == 200 else 0.0,
    }
    fields.update(changes)
    return GatewayReply(**fields)


async def test_eval_runner_grades_and_retries() -> None:
    """Passed, failed, truncated, retried 502 error, and non-retried 400 error outcomes."""
    replies = {
        "correct": make_reply(answer="Tokyo"),
        "wrong": make_reply(answer="Kyoto"),
        "cut off": make_reply(answer="Tokyo", truncated=True),
        "server error": make_reply(502),
        "bad request": make_reply(400),
    }
    questions = [
        make_question(id=f"q{index:03d}", question=text) for index, text in enumerate(replies)
    ]
    runner = EvalRunner(
        client=ScriptedGatewayClient(replies),
        concurrency=2,
        retries=2,
        retry_wait_seconds=0,
    )
    run = await runner.run(questions, "small", "small", "all", "test.jsonl")
    outcomes = {outcome.question: outcome for outcome in run.outcomes}
    assert [outcome.question for outcome in run.outcomes] == list(replies)
    assert outcomes["correct"].status == "passed"
    assert outcomes["correct"].cost_usd == 0.00002
    assert outcomes["wrong"].status == "failed"
    assert (outcomes["cut off"].status, outcomes["cut off"].truncated) == ("failed", True)
    assert (outcomes["server error"].status, outcomes["server error"].attempts) == ("error", 3)
    assert (outcomes["bad request"].status, outcomes["bad request"].attempts) == ("error", 1)


class SequenceGatewayClient(GatewayClient):
    """Gateway client that returns the next reply from a list on each call."""

    def __init__(self, replies: list[GatewayReply]) -> None:
        self.replies: list[GatewayReply] = replies
        self.questions: list[str] = []

    async def ask(self, question: str, model: str) -> GatewayReply:
        """Record the question and return the next scripted reply."""
        self.questions.append(question)
        return self.replies.pop(0)


async def test_eval_runner_retries_rate_limits_and_paces(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Gemini quota 502s are retried via the injected sleep; starts are paced; bad combo refused."""
    quota = "Gemini API error (HTTP 429): You exceeded your current quota, please check your plan."
    assert classify_failure(quota) == "rate_limit"
    client = SequenceGatewayClient(
        [
            make_reply(502, error_message=quota),
            make_reply(502, error_message=quota),
            make_reply(answer="Tokyo"),
            make_reply(answer="Tokyo"),
        ]
    )
    now = [0.0]
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        """Record the wait and move the fake clock forward instead of sleeping."""
        waits.append(seconds)
        now[0] += seconds

    logged: list[str] = []
    runner = EvalRunner(
        client=client,
        concurrency=1,
        retries=2,
        retry_wait_seconds=20.0,
        min_interval_seconds=5.0,
        sleep=fake_sleep,
        clock=lambda: now[0],
        log=logged.append,
    )
    questions = [make_question(id="q000", question="a"), make_question(id="q001", question="b")]
    run = await runner.run(questions, "gem-lite", "gem-lite", "all", "test.jsonl")
    first, second = run.outcomes
    assert (first.status, first.answer, first.attempts) == ("passed", "Tokyo", 3)
    assert (second.status, second.attempts) == ("passed", 1)
    assert client.questions == ["a", "a", "a", "b"]
    assert runner.rate_limit_retries == 2
    assert logged == [
        "retrying q000 in 20 s (attempt 2 of 3)",
        "retrying q000 in 20 s (attempt 3 of 3)",
    ]
    assert waits == [20.0, 20.0, 5.0]
    summary_text = render_eval_summary(summarize(run), "results/x.json", runner.rate_limit_retries)
    assert "  rate-limit retries: 2" in summary_text.splitlines()

    with pytest.raises(ValueError, match="--concurrency 1"):
        EvalRunner(client, concurrency=2, retries=2, retry_wait_seconds=1, min_interval_seconds=1)
    exit_code = eval_cli.main(
        ["--model", "small", "--concurrency", "2", "--min-interval-seconds", "5"]
    )
    error_lines = capsys.readouterr().err.strip().splitlines()
    assert exit_code == 1
    assert error_lines == ["--min-interval-seconds is only allowed with --concurrency 1"]


def test_compare_runs_groups_shared_questions() -> None:
    """Groups are counted on shared ids only, and errors count as not passed."""
    run_a = make_run(
        "small",
        [
            make_outcome("q001", "passed"),
            make_outcome("q002", "failed", "hard"),
            make_outcome("q003", "passed"),
            make_outcome("q004", "error", "hard"),
            make_outcome("q005", "passed"),
        ],
    )
    run_b = make_run(
        "big",
        [
            make_outcome("q001", "passed"),
            make_outcome("q002", "passed", "hard"),
            make_outcome("q003", "failed"),
            make_outcome("q004", "failed", "hard"),
            make_outcome("q099", "passed"),
        ],
    )
    report = compare_runs(run_a, run_b)
    assert (report.compared, report.only_in_a, report.only_in_b) == (4, 1, 1)
    assert report.both_passed.count == 1
    assert (report.only_b_passed.count, report.only_b_passed.hard) == (1, 1)
    assert report.only_a_passed.count == 1
    assert (report.both_not_passed.count, report.both_not_passed.hard) == (1, 1)
    assert report.score_a_percent == pytest.approx(50.0)
    assert report.score_b_percent == pytest.approx(50.0)


class FixedRouter(Router):
    """Router stub that always picks the same model."""

    def __init__(self, model_name: str) -> None:
        self.model_name: str = model_name

    def choose(self, messages: list[Message]) -> RouteDecision:
        """Return the fixed model with a stub reason."""
        return RouteDecision(
            model_name=self.model_name,
            score=0,
            threshold=2,
            hits=[],
            reason=f"stub: always {self.model_name}",
        )


class QuestionRouter(Router):
    """Router stub that picks big only for the listed question texts."""

    def __init__(self, big_questions: set[str]) -> None:
        self.big_questions: set[str] = big_questions

    def choose(self, messages: list[Message]) -> RouteDecision:
        """Pick big when the last message is one of the listed questions."""
        model_name = "big" if messages[-1].content in self.big_questions else "small"
        return RouteDecision(model_name=model_name, score=0, threshold=2, hits=[], reason="stub")


class PerModelProvider(LLMProvider):
    """Provider stub that returns or raises a scripted result per model name."""

    def __init__(self, behaviour: dict[str, CompletionResult | ProviderError]) -> None:
        self.behaviour: dict[str, CompletionResult | ProviderError] = behaviour

    async def complete(self, model_spec: ModelSpec, messages: list[Message]) -> CompletionResult:
        """Return or raise what was scripted for this model."""
        outcome = self.behaviour[model_spec.name]
        if isinstance(outcome, ProviderError):
            raise outcome
        return outcome


def answer(text: str = "Paris", truncated: bool = False) -> CompletionResult:
    """Build a provider result with fixed token counts."""
    return CompletionResult(text=text, input_tokens=12, output_tokens=3, truncated=truncated)


AUTO_BODY = {**VALID_BODY, "model": "auto"}


def test_rule_router_scores_and_explains() -> None:
    """Plain questions go small, keyword- and number-heavy ones go big, with clear reasons."""
    config = load_routing_config(str(ROUTING_CONFIG_PATH))
    router = build_rule_router(config)

    def ask(text: str, chosen_router: Router = router) -> RouteDecision:
        """Route one user message."""
        return chosen_router.choose([Message(role="user", content=text)])

    plain = ask("What is the capital of Japan?")
    assert (plain.model_name, plain.score) == ("small", 0)
    assert plain.reason == "score 0 (threshold 2): no rules matched"
    hard_text = "CALCULATE the total of 12, 15 and 18 dollars. Round to 2 decimal places."
    hard = ask(hard_text)
    assert (hard.model_name, hard.score) == ("big", 2)
    assert hard.reason == (
        "score 2 (threshold 2): keywords +1 [calculate], many_numbers +1 [4 numbers]"
    )
    assert ask(hard_text, build_rule_router(config, threshold=5)).model_name == "small"
    long_decision = ask("word " * 91)
    assert long_decision.score == 2
    assert long_decision.hits[0].rule == "long_question"


def test_routing_config_loads_and_validates(tmp_path: Path) -> None:
    """A valid file loads; a missing key and an unknown model name raise clear errors."""
    text = ROUTING_CONFIG_PATH.read_text(encoding="utf-8")
    valid_path = tmp_path / "routing.toml"
    valid_path.write_text(text, encoding="utf-8")
    config = load_routing_config(str(valid_path))
    assert (config.small_model, config.big_model, config.threshold) == ("small", "big", 2)
    assert "round to" in config.keywords
    missing_path = tmp_path / "missing.toml"
    missing_path.write_text(text.replace("min_numbers = 3\n", ""), encoding="utf-8")
    with pytest.raises(ValueError, match="min_numbers"):
        load_routing_config(str(missing_path))
    registry = ModelRegistry(MODELS_CONFIG_PATH)
    validate_routing_models(config, registry)
    with pytest.raises(ValueError, match="huge"):
        validate_routing_models(config.model_copy(update={"big_model": "huge"}), registry)


async def test_auto_routes_and_explicit_skips_router() -> None:
    """auto records the router's choice and reason; an explicit model records "explicit"."""
    repository = InMemoryRequestRepository()
    provider = PerModelProvider({"small": answer(), "big": answer()})
    async with build_client(provider, repository, router=FixedRouter("small")) as client:
        auto_response = await client.post("/v1/chat", json=AUTO_BODY)
        big_response = await client.post("/v1/chat", json={**VALID_BODY, "model": "big"})
    assert auto_response.json()["model"] == "small"
    assert big_response.json()["model"] == "big"
    auto_row, big_row = repository.records
    assert (auto_row.requested_model, auto_row.model_name) == ("auto", "small")
    assert auto_row.route_reason == "stub: always small"
    assert auto_row.fallback_used is False
    assert (big_row.requested_model, big_row.route_reason) == ("big", "explicit")


async def test_fallback_on_failure_sums_attempts() -> None:
    """A failed small attempt falls back to big; one row holds the totals of both attempts."""
    registry = ModelRegistry(MODELS_CONFIG_PATH)
    small_error = ProviderError("small failed", input_tokens=10, output_tokens=100)
    repository = InMemoryRequestRepository()
    provider = PerModelProvider({"small": small_error, "big": answer("From big")})
    async with build_client(provider, repository, registry, FixedRouter("small")) as client:
        response = await client.post("/v1/chat", json=AUTO_BODY)
    assert response.status_code == 200
    body = response.json()
    assert (body["answer"], body["model"], body["fallback_used"]) == ("From big", "big", True)
    expected_cost = registry.get("small").cost_usd(10, 100) + registry.get("big").cost_usd(12, 3)
    assert len(repository.records) == 1
    row = repository.records[0]
    assert (row.status, row.model_name, row.fallback_used) == ("ok", "big", True)
    assert (row.input_tokens, row.output_tokens) == (22, 103)
    assert row.cost_usd == expected_cost
    assert body["cost_usd"] == expected_cost

    both_fail = PerModelProvider({"small": small_error, "big": ProviderError("big failed")})
    failed_repository = InMemoryRequestRepository()
    async with build_client(both_fail, failed_repository, registry, FixedRouter("small")) as client:
        failed = await client.post("/v1/chat", json=AUTO_BODY)
    assert failed.status_code == 502
    assert len(failed_repository.records) == 1
    failed_row = failed_repository.records[0]
    assert (failed_row.status, failed_row.fallback_used) == ("error", True)
    assert failed_row.error_message == "big failed"


async def test_fallback_on_cut_off_only_when_enabled() -> None:
    """By default a cut-off answer is returned as is; fallback_on_truncated makes it fall back."""
    provider = PerModelProvider({"small": answer(truncated=True), "big": answer("From big")})
    repository = InMemoryRequestRepository()
    async with build_client(provider, repository, router=FixedRouter("small")) as client:
        default_response = await client.post("/v1/chat", json=AUTO_BODY)
    body = default_response.json()
    assert (body["model"], body["truncated"], body["fallback_used"]) == ("small", True, False)
    assert len(body["attempts"]) == 1
    row = repository.records[0]
    assert (row.status, row.fallback_used, row.model_name) == ("truncated", False, "small")

    enabled = load_routing_config(str(ROUTING_CONFIG_PATH)).model_copy(
        update={"fallback_on_truncated": True}
    )
    enabled_repository = InMemoryRequestRepository()
    async with build_client(
        provider, enabled_repository, router=FixedRouter("small"), config=enabled
    ) as client:
        enabled_response = await client.post("/v1/chat", json=AUTO_BODY)
    enabled_body = enabled_response.json()
    assert (enabled_body["model"], enabled_body["fallback_used"]) == ("big", True)
    enabled_row = enabled_repository.records[0]
    assert (enabled_row.status, enabled_row.fallback_used) == ("ok", True)


OLD_REQUESTS_TABLE_SQL = """
CREATE TABLE requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    status TEXT NOT NULL,
    error_type TEXT,
    error_message TEXT,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    cost_usd REAL NOT NULL
)
"""


async def test_sqlite_upgrades_old_database(tmp_path: Path) -> None:
    """initialize() adds the routing columns to an old table and keeps its rows."""
    database_path = tmp_path / "old.db"
    async with aiosqlite.connect(database_path) as connection:
        await connection.execute(OLD_REQUESTS_TABLE_SQL)
        await connection.execute(
            "INSERT INTO requests (created_at, model_name, model_id, question, answer, status, "
            "input_tokens, output_tokens, latency_ms, cost_usd) "
            "VALUES ('2026-11-25T10:00:00+00:00', 'big', 'm', 'old', 'a', 'ok', 1, 2, 3.0, 0.0)"
        )
        await connection.commit()
    repository = SqliteRequestRepository(database_path)
    await repository.initialize()
    async with aiosqlite.connect(database_path) as connection:
        async with connection.execute("PRAGMA table_info(requests)") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}
    assert {"requested_model", "route_reason", "fallback_used"} <= columns
    await repository.add(
        make_record(question="new", requested_model="auto", route_reason="r", fallback_used=True),
        [],
    )
    old_row, new_row = await repository.list_all()
    assert (old_row.question, old_row.requested_model, old_row.route_reason) == ("old", "", "")
    assert old_row.fallback_used is False
    assert (new_row.requested_model, new_row.route_reason, new_row.fallback_used) == (
        "auto",
        "r",
        True,
    )


def test_router_replay_counts_routing_fallback_and_cost() -> None:
    """Replay uses saved outcomes per choice, falls back on small errors, and adds wasted cost."""

    def outcome(question_id: str, text: str, status: str, cost: float) -> EvalOutcome:
        """Build a saved outcome with a question text and cost."""
        return make_outcome(question_id, status).model_copy(
            update={"question": text, "cost_usd": cost}
        )

    small_run = make_run(
        "small",
        [
            outcome("q001", "A", "passed", 0.001),
            outcome("q002", "B", "failed", 0.001),
            outcome("q003", "C", "error", 0.0005),
            outcome("q004", "D", "passed", 0.001),
        ],
    )
    big_run = make_run(
        "big",
        [
            outcome("q001", "A", "passed", 0.01),
            outcome("q002", "B", "passed", 0.01),
            outcome("q003", "C", "passed", 0.01),
            outcome("q004", "D", "failed", 0.01),
        ],
    )
    router = QuestionRouter({"B"})
    with_fallback = replay(small_run, big_run, router, fallback=True, split="all")
    assert (with_fallback.to_small, with_fallback.to_big) == (3, 1)
    assert (with_fallback.fallbacks, with_fallback.passed, with_fallback.total) == (1, 4, 4)
    assert with_fallback.cost_usd == pytest.approx(0.001 + 0.01 + 0.0005 + 0.01 + 0.001)
    without_fallback = replay(small_run, big_run, router, fallback=False, split="all")
    assert (without_fallback.fallbacks, without_fallback.passed) == (0, 3)
    assert without_fallback.cost_usd == pytest.approx(0.001 + 0.01 + 0.0005 + 0.001)
    all_big, all_small = replay_baselines(small_run, big_run, split="all")
    assert (all_big.to_big, all_big.passed, all_big.fallbacks) == (4, 3, 0)
    assert all_big.cost_usd == pytest.approx(0.04)
    assert (all_small.to_small, all_small.passed) == (4, 2)
    assert all_small.cost_usd == pytest.approx(0.0035)


MMLU_PRO_PATH = Path(__file__).resolve().parent.parent / "eval" / "mmlu_pro_subset.jsonl"
LOAD_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "config" / "load_questions.txt"
MIXED_LOAD_PATH = Path(__file__).resolve().parent.parent / "config" / "load_questions_mixed.txt"


def test_choice_check_reads_the_final_letter() -> None:
    """The choice check takes the last 'Answer: X' line, or a last line that is just a letter."""
    question = make_question(check="choice", expected=["C"])
    for text in ("Some reasoning...\nAnswer: C", "**Answer: C**", "The answer is (c).", "…\nC"):
        assert check_answer(question, text), text
    for text in (
        "Answer: B",
        "I have no idea which one it is.",
        "I think answer: c is likely, but not sure about the rest",
    ):
        assert not check_answer(question, text), text
    two_lines = "Answer: A\nWait, let me recheck.\nAnswer: D"
    assert check_answer(make_question(check="choice", expected=["D"]), two_lines)


def test_loader_accepts_choice_and_the_mmlu_pro_file(tmp_path: Path) -> None:
    """Choice keys must be one letter A-J; free-form categories load; the real file is valid."""
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text(make_question(check="choice", expected=["K"]).model_dump_json())
    with pytest.raises(ValueError):
        load_eval_set(str(bad_path))
    good = make_question(check="choice", expected=["C"], category="computer science")
    good_path = tmp_path / "good.jsonl"
    good_path.write_text(good.model_dump_json())
    assert load_eval_set(str(good_path))[0].category == "computer science"

    questions = load_eval_set(str(MMLU_PRO_PATH))
    assert len(questions) == 100
    assert len({question.id for question in questions}) == 100
    assert all(question.check == "choice" for question in questions)
    assert all(len(q.expected) == 1 and q.expected[0] in "ABCDEFGHIJ" for q in questions)
    assert sum(1 for question in questions if question.split == "tune") == 50
    assert sum(1 for question in questions if question.split == "test") == 50
    assert len({question.category for question in questions}) >= 10
    for question in questions:
        lines = question.question.splitlines()
        assert "Options:" in lines, question.id
        assert any(line.startswith("A. ") for line in lines), question.id
        assert any(line.startswith("J. ") for line in lines), question.id


def test_eval_summary_shows_na_for_empty_groups() -> None:
    """A run with only hard questions renders 'easy: n/a' instead of dividing by zero."""
    run = make_run("small", [make_outcome("m001", "passed", "hard")])
    text = render_eval_summary(summarize(run), "results/x.json")
    assert "easy: n/a" in text
    assert "hard: 1/1 (100.0%)" in text


def test_mixed_load_file_routes_hard_lines_to_big() -> None:
    """60 lines: the first 40 copy the base file, and the last 20 are all routed to big."""
    lines = [line for line in MIXED_LOAD_PATH.read_text(encoding="utf-8").splitlines() if line]
    base_lines = LOAD_QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 60
    assert lines[:40] == base_lines
    router = build_rule_router(load_routing_config(str(ROUTING_CONFIG_PATH)))
    for line in lines[40:]:
        decision = router.choose([Message(role="user", content=line)])
        assert decision.model_name == "big", (line, decision.reason)


def cascade_outcome(
    question_id: str,
    status: str,
    output_tokens: int = 100,
    cost: float = 0.001,
    split: str = "tune",
    truncated: bool = False,
) -> EvalOutcome:
    """Build a saved outcome with the fields the cascade replay reads."""
    return make_outcome(question_id, status).model_copy(
        update={
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "split": split,
            "truncated": truncated,
        }
    )


def test_cascade_replay_rows_follow_the_definitions() -> None:
    """Baselines, fallback-only, and a token limit count, rescue, lose, and cost as defined."""
    small_run = make_run(
        "small",
        [
            cascade_outcome("q1", "passed", 300, 0.001, "tune"),
            cascade_outcome("q2", "error", 0, 0.0005, "test"),
            cascade_outcome("q3", "failed", 4096, 0.002, "tune", truncated=True),
            cascade_outcome("q4", "passed", 3000, 0.003, "test"),
            cascade_outcome("q5", "failed", 200, 0.001, "tune"),
        ],
    )
    big_run = make_run(
        "big",
        [
            cascade_outcome("q1", "passed", cost=0.01),
            cascade_outcome("q2", "passed", cost=0.01),
            cascade_outcome("q3", "failed", cost=0.01),
            cascade_outcome("q4", "failed", cost=0.01),
            cascade_outcome("q5", "passed", cost=0.01),
        ],
    )
    table = build_cascade_table(small_run, big_run, "all", [1000])
    all_big, all_small, fallback, limit_row = table.rows
    assert [row.label for row in table.rows] == [
        "all big",
        "all small",
        "fallback only",
        "tokens > 1000",
    ]
    assert (all_big.kept_small, all_big.escalated, all_big.passed) == (0, 5, 3)
    assert (all_big.rescued, all_big.lost) == (2, 1)
    assert all_big.cost_usd == pytest.approx(0.05)
    assert (all_small.kept_small, all_small.escalated, all_small.passed) == (5, 0, 2)
    assert (all_small.rescued, all_small.lost) == (0, 0)
    assert all_small.cost_usd == pytest.approx(0.0075)
    assert (fallback.kept_small, fallback.escalated, fallback.passed) == (3, 2, 3)
    assert (fallback.rescued, fallback.lost) == (1, 0)
    assert fallback.cost_usd == pytest.approx(0.001 + 0.0105 + 0.012 + 0.003 + 0.001)
    assert (limit_row.limit, limit_row.kept_small, limit_row.escalated) == (1000, 2, 3)
    assert (limit_row.passed, limit_row.rescued, limit_row.lost) == (2, 1, 1)
    assert limit_row.cost_usd == pytest.approx(0.001 + 0.0105 + 0.012 + 0.013 + 0.001)


def test_cascade_replay_edge_cases() -> None:
    """Errors and cut-offs always escalate; unshared ids and other splits are ignored."""
    errored = cascade_outcome("q1", "error", 0)
    cut_off = cascade_outcome("q2", "failed", 50, truncated=True)
    for limit in (1, 10**9, None):
        assert escalate(errored, limit) and escalate(cut_off, limit)
    small_run = make_run(
        "small",
        [
            cascade_outcome("q1", "passed", split="tune"),
            cascade_outcome("q2", "passed", split="test"),
            cascade_outcome("q9", "passed", split="tune"),
        ],
    )
    big_run = make_run(
        "big",
        [
            cascade_outcome("q1", "passed", cost=0.0),
            cascade_outcome("q2", "passed", cost=0.0, split="test"),
            cascade_outcome("q8", "passed", cost=0.0),
        ],
    )
    assert build_cascade_table(small_run, big_run, "all", [500]).questions == 2
    tune_table = build_cascade_table(small_run, big_run, "tune", [500])
    assert tune_table.questions == 1
    assert "n/a" in render_cascade_tables([tune_table]).splitlines()[3]
    empty = build_cascade_table(small_run, make_run("big", []), "test", [500])
    assert (empty.questions, empty.rows, empty.small_tokens) == (0, [], None)
    assert "CASCADE REPLAY  |  split: test  |  no questions found in both runs." in (
        render_cascade_tables([empty])
    )


def test_cascade_token_stats_and_rendering() -> None:
    """Token stats use nearest-rank and skip errors; the rendered text has every section."""
    small_outcomes = [
        cascade_outcome(f"q{index}", "passed", tokens)
        for index, tokens in enumerate(range(10, 110, 10))
    ]
    small_outcomes.append(cascade_outcome("q99", "error", 99999))
    big_outcomes = [cascade_outcome(outcome.id, "passed", cost=0.01) for outcome in small_outcomes]
    table = build_cascade_table(
        make_run("small", small_outcomes), make_run("big", big_outcomes), "all", [500]
    )
    assert table.small_tokens is not None
    stats = table.small_tokens
    assert (stats.p50, stats.p75, stats.p90, stats.max) == (50.0, 80.0, 90.0, 100.0)
    text = render_cascade_tables([table])
    for label in ("all big", "all small", "fallback only", "tokens > 500"):
        assert f"  {label}" in text
    assert "SMALL'S OUTPUT TOKENS (answered questions): p50 50 | p75 80 | p90 90 | max 100" in text
    assert text.count("NOTES") == 1


STATIC_ROOT = Path(__file__).resolve().parent.parent / "app" / "static"


def minutes_ago(minutes: float) -> str:
    """Return a UTC ISO time the given number of minutes before now."""
    return (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat()


async def make_dashboard_db(path: Path, records: list[RequestRecord]) -> None:
    """Create a real SQLite request log with the given records."""
    repository = SqliteRequestRepository(path)
    await repository.initialize()
    for record in records:
        await repository.add(record, [])


def dashboard_client(data_dir: Path, default_name: str = "gateway.db") -> httpx.AsyncClient:
    """Build a client for an app whose dashboard reads the given folder."""
    app = create_app()
    service = DashboardService(
        str(data_dir), str(data_dir / default_name), sqlite_repository_factory
    )
    app.dependency_overrides[get_dashboard_service] = lambda: service
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_dashboard_sources_and_name_safety(tmp_path: Path) -> None:
    """Only .db files directly in the folder are sources; anything else is rejected."""
    await make_dashboard_db(tmp_path / "gateway.db", [make_record()])
    await make_dashboard_db(tmp_path / "fake.db", [make_record()])
    (tmp_path / "notes.txt").write_text("not a database", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    await make_dashboard_db(tmp_path / "sub" / "inner.db", [make_record()])
    async with dashboard_client(tmp_path) as client:
        body = (await client.get("/api/dashboard/sources")).json()
        assert [source["name"] for source in body["sources"]] == ["fake.db", "gateway.db"]
        assert body["default"] == "gateway.db"
        assert [source["is_default"] for source in body["sources"]] == [False, True]
        for bad in ("nope.db", "../x.db", "a/b.db", "x.txt", "inner.db"):
            response = await client.get("/api/dashboard/summary", params={"source": bad})
            assert response.status_code == 400, bad
            assert response.json()["error"]["type"] == "unknown_source"
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    async with dashboard_client(empty_dir) as client:
        assert (await client.get("/api/dashboard/sources")).json()["sources"] == []
        for path in ("summary", "timeseries", "requests"):
            response = await client.get(f"/api/dashboard/{path}")
            assert response.status_code == 404, path
            assert response.json()["error"]["type"] == "no_data"


async def test_dashboard_summary_matches_the_stats_numbers(tmp_path: Path) -> None:
    """The summary equals compute_group_stats for the same rows, and minutes drops old rows."""
    recent_records = [
        make_record(created_at=minutes_ago(5), model_name="big", latency_ms=900.0, cost_usd=0.002),
        make_record(created_at=minutes_ago(10), model_name="small", latency_ms=300.0),
        make_record(
            created_at=minutes_ago(15), model_name="small", status="truncated", latency_ms=700.0
        ),
        make_record(
            created_at=minutes_ago(20), status="error", answer=None, latency_ms=5000.0, cost_usd=0.0
        ),
    ]
    old_record = make_record(created_at=minutes_ago(180), latency_ms=100.0, cost_usd=0.5)
    await make_dashboard_db(tmp_path / "gateway.db", [old_record, *recent_records])
    async with dashboard_client(tmp_path) as client:
        everything = (await client.get("/api/dashboard/summary")).json()
        last_hour = (await client.get("/api/dashboard/summary", params={"minutes": 60})).json()
    for body, records in ((everything, [old_record, *recent_records]), (last_hour, recent_records)):
        expected = compute_group_stats(records)
        overall = body["overall"]
        assert (overall["requests"], overall["answered"], overall["errors"]) == (
            expected.requests,
            expected.answered,
            expected.errors,
        )
        assert overall["total_cost_usd"] == pytest.approx(expected.total_cost_usd)
        assert expected.latency is not None
        assert overall["latency"]["p95_ms"] == expected.latency.p95_ms
    assert last_hour["overall"]["requests"] == 4
    assert set(last_hour["by_model"]) == {"big", "small"}


async def test_dashboard_timeseries_buckets(tmp_path: Path) -> None:
    """Buckets are epoch-aligned and gap-free, sum to the rows, and widen for long ranges."""
    records = [make_record(created_at=minutes_ago(offset)) for offset in (1, 1.1, 2, 5, 8)]
    await make_dashboard_db(tmp_path / "gateway.db", records)
    async with dashboard_client(tmp_path) as client:
        body = (await client.get("/api/dashboard/timeseries", params={"minutes": 10})).json()
    size = body["bucket_seconds"]
    starts = [datetime.fromisoformat(bucket["start"]).timestamp() for bucket in body["buckets"]]
    assert size == 10
    assert all(start % size == 0 for start in starts)
    assert all(later - earlier == size for earlier, later in itertools.pairwise(starts))
    assert sum(bucket["requests"] for bucket in body["buckets"]) == len(records)
    empty_buckets = [bucket for bucket in body["buckets"] if bucket["requests"] == 0]
    assert empty_buckets and all(bucket["p95_ms"] is None for bucket in empty_buckets)

    long_dir = tmp_path / "long"
    long_dir.mkdir()
    days = [make_record(created_at=minutes_ago(day * 1440)) for day in (0, 1, 2, 3)]
    await make_dashboard_db(long_dir / "gateway.db", days)
    async with dashboard_client(long_dir) as client:
        long_body = (await client.get("/api/dashboard/timeseries")).json()
    assert long_body["bucket_seconds"] > size
    assert len(long_body["buckets"]) <= 120
    assert sum(bucket["requests"] for bucket in long_body["buckets"]) == len(days)


async def test_dashboard_requests_page_and_static_files(tmp_path: Path) -> None:
    """Requests come newest first with routing fields; the page and assets are safe to serve."""
    records = [
        make_record(created_at=minutes_ago(3), question="oldest"),
        make_record(created_at=minutes_ago(2), question="middle"),
        make_record(
            created_at=minutes_ago(1),
            question="newest",
            requested_model="auto",
            route_reason="score 0 (threshold 2): no rules matched",
            fallback_used=True,
        ),
    ]
    await make_dashboard_db(tmp_path / "gateway.db", records)
    async with dashboard_client(tmp_path) as client:
        items = (await client.get("/api/dashboard/requests", params={"limit": 2})).json()["items"]
        page = await client.get("/dashboard")
        script = await client.get("/dashboard/assets/dashboard.js")
        styles = await client.get("/dashboard/assets/dashboard.css")
    assert [item["question"] for item in items] == ["newest", "middle"]
    assert (items[0]["requested_model"], items[0]["fallback_used"]) == ("auto", True)
    assert items[0]["route_reason"] == "score 0 (threshold 2): no rules matched"
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert script.status_code == 200 and styles.status_code == 200
    assert_static_files_are_safe()


def assert_static_files_are_safe() -> None:
    """No file under app/static makes external requests or inserts HTML from strings."""
    static_files = [path for path in STATIC_ROOT.rglob("*") if path.is_file()]
    assert len(static_files) >= 8
    for path in static_files:
        text = path.read_text(encoding="utf-8")
        without_namespace = text.replace("http://www.w3.org/2000/svg", "")
        assert "http://" not in without_namespace, path.name
        assert "https://" not in without_namespace, path.name
        for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
            assert forbidden not in text, (path.name, forbidden)


async def table_snapshot(path: Path) -> tuple[int, list[str]]:
    """Return the row count of requests and the sorted table names."""
    async with aiosqlite.connect(path) as connection:
        async with connection.execute("SELECT COUNT(*) FROM requests") as cursor:
            row_count = (await cursor.fetchone() or (0,))[0]
        async with connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ) as cursor:
            tables = [row[0] for row in await cursor.fetchall()]
    return row_count, tables


async def test_dashboard_never_writes(tmp_path: Path) -> None:
    """Calling every dashboard endpoint leaves the rows, tables, and .db file list unchanged."""
    database_path = tmp_path / "gateway.db"
    await make_dashboard_db(database_path, [make_record(created_at=minutes_ago(1))])
    before = await table_snapshot(database_path)
    files_before = sorted(path.name for path in tmp_path.glob("*.db"))
    async with dashboard_client(tmp_path) as client:
        for path in ("sources", "summary", "timeseries", "requests"):
            assert (await client.get(f"/api/dashboard/{path}")).status_code == 200, path
        assert (
            await client.get("/api/dashboard/summary", params={"minutes": 5})
        ).status_code == 200
    assert await table_snapshot(database_path) == before
    assert sorted(path.name for path in tmp_path.glob("*.db")) == files_before


def test_classify_failure_types() -> None:
    """Error messages map to failure types in a fixed order, ignoring case."""
    empty = "Groq returned an empty answer (finish_reason=length, output_tokens=4096)"
    assert classify_failure(empty) == "empty_answer"
    for message in (
        "Rate limit reached for model",
        "Groq API error (HTTP 429)",
        "too many requests",
    ):
        assert classify_failure(message) == "rate_limit", message
    assert classify_failure("Request timed out") == "timeout"
    assert classify_failure("Invalid API key") == "auth"
    assert classify_failure("401 Unauthorized") == "auth"
    assert classify_failure(None) == "other"
    assert classify_failure("Could not connect to Groq") == "other"
    assert classify_failure("RATE LIMIT REACHED") == "rate_limit"
    assert classify_failure("groq returned an EMPTY ANSWER") == "empty_answer"


async def test_dashboard_insights_panels(tmp_path: Path) -> None:
    """Insights classify failures, split routing, bin latencies, and list the slowest rows."""

    def answered(age: float, latency: float, **changes: Any) -> RequestRecord:
        """Build an answered row created `age` minutes ago."""
        return make_record(created_at=minutes_ago(age), latency_ms=latency, **changes)

    def failed(age: float, message: str, **changes: Any) -> RequestRecord:
        """Build a failed row created `age` minutes ago."""
        fields: dict[str, Any] = {
            "status": "error",
            "answer": None,
            "error_type": "provider_error",
            "error_message": message,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }
        fields.update(changes)
        return make_record(created_at=minutes_ago(age), **fields)

    empty_answer = "Groq returned an empty answer (finish_reason=length, output_tokens=4096)"
    records = [
        failed(
            1,
            empty_answer,
            question="e1",
            requested_model="auto",
            fallback_used=True,
            input_tokens=50,
            output_tokens=4096,
            cost_usd=0.0025,
        ),
        answered(2, 400.0, requested_model="auto"),
        answered(3, 1000.0, model_name="small", requested_model="auto"),
        answered(4, 2500.0, model_name="small", status="truncated", requested_model="small"),
        answered(5, 6000.0, requested_model="auto", fallback_used=True),
        answered(6, 70000.0),
        answered(7, 15000.0, requested_model="big"),
        failed(
            8,
            "Groq API error (HTTP 429): Rate limit reached",
            question="e2",
            model_name="small",
            requested_model="auto",
        ),
        failed(9, "Request timed out after 30 seconds", question="e3"),
        *(
            failed(
                10 + index,
                "Groq API error (HTTP 401): Invalid API Key",
                question=f"auth{index}",
                model_name="small",
                requested_model="small",
                input_tokens=5,
            )
            for index in range(10)
        ),
        answered(300, 100.0, requested_model="big"),
    ]
    await make_dashboard_db(tmp_path / "gateway.db", records)
    async with dashboard_client(tmp_path) as client:
        body = (await client.get("/api/dashboard/insights", params={"minutes": 60})).json()
        all_time = (await client.get("/api/dashboard/insights")).json()
        bad = await client.get("/api/dashboard/insights", params={"source": "nope.db"})

    failures = body["failures"]
    assert failures["by_type"] == {
        "empty_answer": 1,
        "rate_limit": 1,
        "timeout": 1,
        "auth": 10,
        "other": 0,
    }
    assert (failures["errors"], failures["truncated"]) == (13, 1)
    assert failures["failed_cost_usd"] == pytest.approx(0.0025)
    assert (failures["failed_input_tokens"], failures["failed_output_tokens"]) == (100, 4096)
    expected_recent = ["e1", "e2", "e3", *(f"auth{index}" for index in range(7))]
    assert [item["question"] for item in failures["recent"]] == expected_recent
    assert failures["recent"][0]["failure_type"] == "empty_answer"

    routing = body["routing"]
    assert routing["auto_requests"] == 5
    assert routing["auto_by_model"] == {"big": 3, "small": 2}
    assert routing["explicit_by_model"] == {"big": 1, "small": 11}
    assert (routing["fallbacks"], routing["fallbacks_rescued"]) == (2, 1)

    histogram = body["latency_histogram"]
    assert histogram["bin_labels"] == [
        "<0.5s",
        "0.5-1s",
        "1-2s",
        "2-5s",
        "5-10s",
        "10-20s",
        "20-60s",
        ">60s",
    ]
    assert histogram["by_model"] == {
        "big": [1, 0, 0, 0, 1, 1, 0, 1],
        "small": [0, 0, 1, 1, 0, 0, 0, 0],
    }
    assert sum(sum(counts) for counts in histogram["by_model"].values()) == 6

    slowest = [item["latency_ms"] for item in body["slowest"]]
    assert slowest == [70000.0, 15000.0, 6000.0, 2500.0, 1000.0, 400.0]
    assert len(all_time["slowest"]) == 7 and len(all_time["slowest"]) <= 10
    assert all_time["routing"]["explicit_by_model"] == {"big": 2, "small": 11}

    assert bad.status_code == 400
    assert bad.json()["error"]["type"] == "unknown_source"

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    await make_dashboard_db(empty_dir / "gateway.db", [])
    async with dashboard_client(empty_dir) as client:
        empty = (await client.get("/api/dashboard/insights")).json()
    assert empty["failures"]["errors"] == 0
    assert set(empty["failures"]["by_type"].values()) == {0}
    assert empty["failures"]["recent"] == []
    assert empty["routing"] == {
        "auto_requests": 0,
        "auto_by_model": {},
        "explicit_by_model": {},
        "fallbacks": 0,
        "fallbacks_rescued": 0,
    }
    assert empty["latency_histogram"]["by_model"] == {}
    assert empty["slowest"] == []


async def test_chat_page_navigation_and_shared_files(tmp_path: Path) -> None:
    """The chat page, the root redirect, and the shared and chat assets are served."""
    async with dashboard_client(tmp_path) as client:
        chat_page = await client.get("/chat")
        root = await client.get("/")
        dashboard_page = await client.get("/dashboard")
        asset_paths = (
            "/shared/nav.js",
            "/shared/nav.css",
            "/chat/assets/chat.js",
            "/chat/assets/chat.css",
        )
        assets = [await client.get(path) for path in asset_paths]
    assert chat_page.status_code == 200
    assert chat_page.headers["content-type"].startswith("text/html")
    assert (root.status_code, root.headers["location"]) == (307, "/chat")
    assert [asset.status_code for asset in assets] == [200, 200, 200, 200]
    for page in (chat_page, dashboard_page):
        assert '<nav id="site-nav">' in page.text


CONCISE_INSTRUCTION = (
    "Answer in plain text. Be brief unless I ask for detail. Do not use tables or LaTeX."
)


def make_attempt(**changes: Any) -> AttemptRecord:
    """Build an attempt record for tests; keyword arguments replace the defaults."""
    fields: dict[str, Any] = {
        "attempt": 1,
        "model_name": "big",
        "model_id": "test-model",
        "status": "ok",
        "error_type": None,
        "error_message": None,
        "input_tokens": 10,
        "output_tokens": 20,
        "latency_ms": 100.0,
        "cost_usd": 0.001,
    }
    fields.update(changes)
    return AttemptRecord(**fields)


async def test_attempts_are_returned_and_stored_per_call() -> None:
    """Each provider call becomes one attempt in the response and in storage."""
    registry = ModelRegistry(MODELS_CONFIG_PATH)
    small_error = ProviderError("small failed", input_tokens=10, output_tokens=100)
    repository = InMemoryRequestRepository()
    provider = PerModelProvider({"small": small_error, "big": answer("From big")})
    async with build_client(provider, repository, registry, FixedRouter("small")) as client:
        auto_body = (await client.post("/v1/chat", json=AUTO_BODY)).json()
        big_body = (await client.post("/v1/chat", json={**VALID_BODY, "model": "big"})).json()
    first, second = auto_body["attempts"]
    assert (first["attempt"], first["model"], first["status"]) == (1, "small", "error")
    assert (second["attempt"], second["model"], second["status"]) == (2, "big", "ok")
    assert first["error_message"] == "small failed"
    assert (first["input_tokens"], first["output_tokens"]) == (10, 100)
    assert (second["input_tokens"], second["output_tokens"]) == (12, 3)
    assert first["cost_usd"] == registry.get("small").cost_usd(10, 100)
    assert second["cost_usd"] == registry.get("big").cost_usd(12, 3)
    assert all(attempt["latency_ms"] >= 0 for attempt in auto_body["attempts"])
    stored = repository.attempts[1]
    assert [attempt.status for attempt in stored] == ["error", "ok"]
    row = repository.records[0]
    assert row.input_tokens == sum(attempt.input_tokens for attempt in stored)
    assert row.output_tokens == sum(attempt.output_tokens for attempt in stored)
    assert row.cost_usd == sum(attempt.cost_usd for attempt in stored)
    assert len(big_body["attempts"]) == 1
    assert len(repository.attempts[2]) == 1

    both_fail = PerModelProvider({"small": small_error, "big": ProviderError("big failed")})
    failed_repository = InMemoryRequestRepository()
    async with build_client(both_fail, failed_repository, registry, FixedRouter("small")) as client:
        failed = await client.post("/v1/chat", json=AUTO_BODY)
    assert failed.status_code == 502
    assert [attempt.status for attempt in failed_repository.attempts[1]] == ["error", "error"]


async def test_sqlite_stores_attempts_with_their_request(tmp_path: Path) -> None:
    """add() stores a request and its attempts together; old databases gain the table."""
    repository = SqliteRequestRepository(tmp_path / "gateway.db")
    await repository.initialize()
    fallback_attempts = [
        make_attempt(
            attempt=1,
            model_name="small",
            status="error",
            error_type="provider_error",
            error_message="boom",
        ),
        make_attempt(attempt=2),
    ]
    first_id = await repository.add(make_record(question="fallback"), fallback_attempts)
    second_id = await repository.add(make_record(question="no attempts"), [])
    assert first_id > 0 and second_id != first_id
    stored = await repository.list_attempts([first_id, second_id])
    assert list(stored) == [first_id]
    assert [attempt.attempt for attempt in stored[first_id]] == [1, 2]
    assert [attempt.model_name for attempt in stored[first_id]] == ["small", "big"]
    assert {attempt.request_id for attempt in stored[first_id]} == {first_id}
    assert await repository.list_attempts([]) == {}
    assert [record.id for record in await repository.list_all()] == [first_id, second_id]

    old_path = tmp_path / "part6.db"
    async with aiosqlite.connect(old_path) as connection:
        await connection.execute(CREATE_TABLE_SQL)
        await connection.execute(
            "INSERT INTO requests (created_at, model_name, model_id, question, answer, status, "
            "input_tokens, output_tokens, latency_ms, cost_usd) "
            "VALUES ('2026-11-25T10:00:00+00:00', 'big', 'm', 'old', 'a', 'ok', 1, 2, 3.0, 0.0)"
        )
        await connection.commit()
    old_repository = SqliteRequestRepository(old_path)
    assert await old_repository.list_attempts([1]) == {}
    await old_repository.initialize()
    _, tables = await table_snapshot(old_path)
    assert "request_attempts" in tables
    (old_row,) = await old_repository.list_all()
    assert old_row.question == "old"
    assert old_row.id is not None
    assert await old_repository.list_attempts([old_row.id]) == {}


async def test_dashboard_attempts_and_wasted_spend(tmp_path: Path) -> None:
    """Dashboard rows carry their attempts; insights total attempts and wasted spend."""
    repository = SqliteRequestRepository(tmp_path / "gateway.db")
    await repository.initialize()
    await repository.add(make_record(created_at=minutes_ago(6), question="old"), [])
    await repository.add(
        make_record(created_at=minutes_ago(5), question="single"),
        [make_attempt(input_tokens=10, output_tokens=20, cost_usd=0.001)],
    )
    await repository.add(
        make_record(created_at=minutes_ago(4), question="fallback", fallback_used=True),
        [
            make_attempt(
                model_name="small",
                status="error",
                error_message="small failed",
                input_tokens=10,
                output_tokens=100,
                cost_usd=0.0003,
            ),
            make_attempt(attempt=2, input_tokens=12, output_tokens=30, cost_usd=0.002),
        ],
    )
    await repository.add(
        make_record(
            created_at=minutes_ago(3),
            question="both failed",
            status="error",
            answer=None,
            error_message="big failed",
        ),
        [
            make_attempt(
                model_name="small",
                status="error",
                error_message="small failed",
                input_tokens=5,
                output_tokens=50,
                cost_usd=0.0002,
            ),
            make_attempt(
                attempt=2,
                status="error",
                error_message="big failed",
                input_tokens=5,
                output_tokens=0,
                cost_usd=0.00005,
            ),
        ],
    )
    await repository.add(
        make_record(
            created_at=minutes_ago(2), question="cut off", model_name="small", status="truncated"
        ),
        [
            make_attempt(
                model_name="small",
                status="truncated",
                input_tokens=8,
                output_tokens=900,
                cost_usd=0.0003,
            )
        ],
    )
    async with dashboard_client(tmp_path) as client:
        items = (await client.get("/api/dashboard/requests")).json()["items"]
        insights = (await client.get("/api/dashboard/insights")).json()

    by_question = {item["question"]: item for item in items}
    assert by_question["old"]["attempts"] == []
    fallback = by_question["fallback"]["attempts"]
    assert [(attempt["model"], attempt["status"]) for attempt in fallback] == [
        ("small", "error"),
        ("big", "ok"),
    ]
    assert len(insights["failures"]["recent"][0]["attempts"]) == 2

    stats = insights["attempts_by_model"]
    big, small = stats["big"], stats["small"]
    assert (big["attempts"], big["answered"], big["truncated"], big["errors"]) == (3, 2, 0, 1)
    assert (big["input_tokens"], big["output_tokens"]) == (27, 50)
    assert big["cost_usd"] == pytest.approx(0.00305)
    assert (small["attempts"], small["answered"], small["truncated"], small["errors"]) == (
        3,
        1,
        1,
        2,
    )
    assert (small["input_tokens"], small["output_tokens"]) == (23, 1050)
    assert small["cost_usd"] == pytest.approx(0.0008)

    failures = insights["failures"]
    assert failures["wasted_cost_usd"] == pytest.approx(0.0003 + 0.0002 + 0.00005)
    assert failures["wasted_tokens"] == 110 + 55 + 5


async def test_fallback_setting_and_chat_assets(tmp_path: Path) -> None:
    """fallback_on_truncated is optional; markdown.js is served and loaded before chat.js."""
    text = ROUTING_CONFIG_PATH.read_text(encoding="utf-8")
    assert "fallback_on_truncated = false" in text
    assert load_routing_config(str(ROUTING_CONFIG_PATH)).fallback_on_truncated is False
    enabled_path = tmp_path / "enabled.toml"
    enabled_path.write_text(
        text.replace("fallback_on_truncated = false", "fallback_on_truncated = true"),
        encoding="utf-8",
    )
    assert load_routing_config(str(enabled_path)).fallback_on_truncated is True
    absent_path = tmp_path / "absent.toml"
    absent_path.write_text(text.replace("fallback_on_truncated = false\n", ""), encoding="utf-8")
    assert "fallback_on_truncated" not in absent_path.read_text(encoding="utf-8")
    assert load_routing_config(str(absent_path)).fallback_on_truncated is False

    async with dashboard_client(tmp_path) as client:
        script = await client.get("/chat/assets/markdown.js")
        page = await client.get("/chat")
    assert script.status_code == 200
    markdown_at = page.text.index('src="/chat/assets/markdown.js"')
    assert markdown_at < page.text.index('src="/chat/assets/chat.js"')
    chat_script = (STATIC_ROOT / "chat" / "chat.js").read_text(encoding="utf-8")
    assert chat_script.count(CONCISE_INSTRUCTION) == 1
    assert_static_files_are_safe()


def app_client(dependency: Any, service: Any) -> httpx.AsyncClient:
    """Build a client for an app with one service dependency replaced."""
    app = create_app()
    app.dependency_overrides[dependency] = lambda: service
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_gateway_info_models_config_and_preview(tmp_path: Path) -> None:
    """Model health comes from attempts; config has no secrets; route preview writes nothing."""
    models_path = tmp_path / "models.toml"
    models_path.write_text(
        MODELS_CONFIG_PATH.read_text(encoding="utf-8")
        + '\n[models.idle]\nprovider = "groq"\nmodel_id = "idle-model"\n'
        "input_price_per_million_tokens = 0.1\noutput_price_per_million_tokens = 0.2\n",
        encoding="utf-8",
    )
    registry = ModelRegistry(models_path)
    repository = SqliteRequestRepository(tmp_path / "gateway.db")
    await repository.initialize()
    rate_limit = "Groq API error (HTTP 429): Rate limit reached for model"
    empty = "Groq returned an empty answer (finish_reason=length, output_tokens=4096)"
    rows = [
        (5, [make_attempt(output_tokens=100, latency_ms=100.0, cost_usd=0.001)]),
        (5, [make_attempt(output_tokens=200, latency_ms=200.0, cost_usd=0.002)]),
        (
            4,
            [make_attempt(status="truncated", output_tokens=900, latency_ms=300.0, cost_usd=0.003)],
        ),
        (
            4,
            [
                make_attempt(
                    model_name="small",
                    status="error",
                    error_message=rate_limit,
                    output_tokens=0,
                    latency_ms=50.0,
                    cost_usd=0.0,
                ),
                make_attempt(attempt=2, output_tokens=300, latency_ms=400.0, cost_usd=0.004),
            ],
        ),
        (
            3,
            [
                make_attempt(
                    model_name="small",
                    status="error",
                    error_message=empty,
                    input_tokens=20,
                    output_tokens=4096,
                    latency_ms=900.0,
                    cost_usd=0.0012,
                )
            ],
        ),
        (
            2,
            [make_attempt(model_name="small", output_tokens=50, latency_ms=150.0, cost_usd=0.0005)],
        ),
        (300, [make_attempt(status="error", error_message="Request timed out")]),
    ]
    for age, attempts in rows:
        await repository.add(make_record(created_at=minutes_ago(age)), attempts)
    secret = "test-secret-key-123"
    settings = Settings(groq_api_key=secret, provider_mode="fake", database_path="data/fake.db")
    routing_config = load_routing_config(str(ROUTING_CONFIG_PATH))
    service = GatewayInfoService(
        settings, registry, routing_config, build_rule_router(routing_config), repository
    )
    rows_before = len(await repository.list_all())
    async with app_client(get_gateway_info_service, service) as client:
        health = (await client.get("/api/gateway/models", params={"minutes": 60})).json()
        config_response = await client.get("/api/gateway/config")
        plain = (
            await client.post("/api/gateway/route-preview", json={"question": "Hi there"})
        ).json()
        hard_question = "Calculate the total of 12, 15 and 18 dollars. Round to 2 decimal places."
        hard = (
            await client.post("/api/gateway/route-preview", json={"question": hard_question})
        ).json()
        too_short = await client.post("/api/gateway/route-preview", json={"question": ""})
        too_long = await client.post("/api/gateway/route-preview", json={"question": "a" * 4001})

    models = {entry["name"]: entry for entry in health["models"]}
    assert list(models)[:2] == ["small", "big"] and list(models)[-1] == "idle"
    big, small, idle = models["big"], models["small"], models["idle"]
    assert (big["attempts"], big["answered"], big["truncated"], big["errors"]) == (4, 4, 1, 0)
    assert (big["p50_ms"], big["p95_ms"], big["avg_output_tokens"]) == (200.0, 400.0, 375.0)
    assert big["cost_usd"] == pytest.approx(0.01)
    assert (big["error_rate_percent"], big["health"]) == (0.0, "Healthy")
    assert (small["attempts"], small["answered"], small["errors"]) == (3, 1, 2)
    assert (small["rate_limit_errors"], small["health"]) == (1, "Degraded")
    assert small["error_rate_percent"] == pytest.approx(200 / 3)
    assert (small["p50_ms"], small["p95_ms"], small["avg_output_tokens"]) == (150.0, 150.0, 50.0)
    assert small["cost_usd"] == pytest.approx(0.0017)
    assert (idle["attempts"], idle["p50_ms"], idle["avg_output_tokens"]) == (0, None, None)
    assert idle["health"] == "No recent traffic"

    config = config_response.json()
    config_names = [model["name"] for model in config["models"]]
    assert config_names[:2] == ["small", "big"] and config_names[-1] == "idle"
    assert config["models"][-1]["output_price_per_million_tokens"] == 0.2
    assert (config["routing"]["threshold"], config["database_file"]) == (2, "fake.db")
    text = config_response.text
    for forbidden in ("GROQ_API_KEY", "groq_api_key", secret, "data/", "config/", "\\"):
        assert forbidden not in text, forbidden

    all_rules = ["long_question", "keywords", "many_numbers", "multiple_questions"]
    assert (plain["model"], plain["score"], plain["not_fired"]) == ("small", 0, all_rules)
    assert (hard["model"], hard["score"]) == ("big", 2)
    assert [(hit["rule"], hit["points"]) for hit in hard["hits"]] == [
        ("keywords", 1),
        ("many_numbers", 1),
    ]
    assert hard["not_fired"] == ["long_question", "multiple_questions"]
    assert (too_short.status_code, too_long.status_code) == (422, 422)
    assert len(await repository.list_all()) == rows_before


def experiment_outcome(
    question_id: str, status: str, split: str, difficulty: str = "easy", cost: float = 0.001
) -> EvalOutcome:
    """Build a saved outcome with a split and a cost."""
    return make_outcome(question_id, status, difficulty).model_copy(
        update={"split": split, "cost_usd": cost, "question": f"Question {question_id}?"}
    )


async def test_experiments_runs_replay_and_compare(tmp_path: Path) -> None:
    """Runs, replays, and comparisons equal the existing functions' results."""
    results = tmp_path / "results"
    results.mkdir()
    small = make_run(
        "small",
        [
            experiment_outcome("q1", "passed", "tune"),
            experiment_outcome("q2", "failed", "test", "hard"),
            experiment_outcome("q3", "error", "tune", "hard", 0.0),
            experiment_outcome("q4", "passed", "test"),
        ],
    ).model_copy(update={"created_at": "2026-11-26T08:00:00+00:00"})
    big = make_run(
        "big",
        [
            experiment_outcome("q1", "passed", "tune", cost=0.01),
            experiment_outcome("q2", "passed", "test", "hard", 0.01),
            experiment_outcome("q3", "passed", "tune", "hard", 0.01),
            experiment_outcome("q4", "failed", "test", cost=0.01),
        ],
    ).model_copy(update={"created_at": "2026-11-26T09:00:00+00:00"})
    (results / "eval_small_1.json").write_text(small.model_dump_json(), encoding="utf-8")
    (results / "eval_big_1.json").write_text(big.model_dump_json(), encoding="utf-8")
    (results / "eval_bad.json").write_text("{not json", encoding="utf-8")
    (results / "notes.json").write_text(small.model_dump_json(), encoding="utf-8")
    routing_config = load_routing_config(str(ROUTING_CONFIG_PATH))
    service = ExperimentsService(str(results), routing_config)
    files = {"small": "eval_small_1.json", "big": "eval_big_1.json"}
    async with app_client(get_experiments_service, service) as client:
        listing = (await client.get("/api/experiments/runs")).json()
        tune = (await client.get("/api/experiments/runs", params={"split": "tune"})).json()
        replay_body = (await client.get("/api/experiments/replay", params=files)).json()
        compare_body = (
            await client.get(
                "/api/experiments/compare", params={"a": files["small"], "b": files["big"]}
            )
        ).json()
        bad = [
            await client.get("/api/experiments/replay", params={"small": name, "big": files["big"]})
            for name in ("../eval_small_1.json", "sub/eval_small_1.json", "eval_missing.json")
        ]

    assert [run["file"] for run in listing["runs"]] == ["eval_big_1.json", "eval_small_1.json"]
    assert [item["file"] for item in listing["skipped"]] == ["eval_bad.json"]
    for run_info, run in zip(listing["runs"], (big, small), strict=True):
        summary = summarize(run)
        assert (run_info["passed"], run_info["errors"]) == (summary.passed, summary.errors)
        assert run_info["score_percent"] == pytest.approx(summary.score_percent)
        assert run_info["total_cost_usd"] == pytest.approx(summary.total_cost_usd)
    tune_small = next(run for run in tune["runs"] if run["file"] == files["small"])
    assert (tune_small["questions"], tune_small["passed"]) == (2, 1)

    router = build_replay_table(small, big, routing_config, [0, 1, 2, 3, 4, 5], True, "all")
    cascade = build_cascade_table(small, big, "all", [500, 1000, 1500, 2000, 3000, 4000])
    assert [(row["label"], row["passed"]) for row in replay_body["router_rows"]] == [
        (row.label, row.passed) for row in router.rows
    ]
    assert [row["cost_usd"] for row in replay_body["router_rows"]] == pytest.approx(
        [row.cost_usd for row in router.rows]
    )
    assert [
        (row["label"], row["escalated"], row["passed"]) for row in replay_body["cascade_rows"]
    ] == [(row.label, row.escalated, row.passed) for row in cascade.rows]

    report = compare_runs(small, big)
    assert compare_body["compared"] == report.compared
    for group in ("both_passed", "only_a_passed", "only_b_passed", "both_not_passed"):
        expected = getattr(report, group)
        assert (compare_body[group]["count"], compare_body[group]["hard"]) == (
            expected.count,
            expected.hard,
        )
    assert [item["id"] for item in compare_body["only_b_passed_questions"]] == [
        pair.a.id for pair in report.only_b_passed.pairs
    ]
    for response in bad:
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "unknown_file"

    other = tmp_path / "other"
    other.mkdir()
    (other / "eval_a.json").write_text(small.model_dump_json(), encoding="utf-8")
    moved = small.model_copy(update={"eval_set_path": "eval/other_set.jsonl"})
    (other / "eval_b.json").write_text(moved.model_dump_json(), encoding="utf-8")
    other_service = ExperimentsService(str(other), routing_config)
    async with app_client(get_experiments_service, other_service) as client:
        mismatch = await client.get(
            "/api/experiments/compare", params={"a": "eval_a.json", "b": "eval_b.json"}
        )
    assert mismatch.status_code == 400
    assert mismatch.json()["error"]["type"] == "runs_not_comparable"


async def wait_for_demo_state(client: httpx.AsyncClient, wanted: str) -> dict[str, Any]:
    """Poll the demo status until it reaches a state (fails after about 3 seconds)."""
    for _ in range(60):
        status = (await client.get("/api/demo/status")).json()
        if status["state"] == wanted:
            return status
        await asyncio.sleep(0.05)
    raise AssertionError(f"demo never reached {wanted}: {status}")


async def test_demo_service_runs_stops_and_refuses(tmp_path: Path) -> None:
    """Demo runs go through ChatService, can be stopped, and are refused outside fake setups."""
    questions_dir = tmp_path / "config"
    questions_dir.mkdir()
    (questions_dir / "load_questions.txt").write_text("What is 2+2?\n\nName a colour.\n")
    repository = InMemoryRequestRepository()
    routing_config = load_routing_config(str(ROUTING_CONFIG_PATH))
    chat = ChatService(
        provider=FakeProvider(),
        registry=ModelRegistry(MODELS_CONFIG_PATH),
        repository=repository,
        router=build_rule_router(routing_config),
        routing_config=routing_config,
        default_model_name="big",
    )

    def demo(schedule: list[float], mode: str = "fake", database: str = "fake.db") -> DemoService:
        """Build a demo service with a fixed schedule."""
        return DemoService(
            chat,
            ["small", "big"],
            mode,
            database,
            str(questions_dir),
            schedule_builder=lambda scenario, seed: list(schedule),
        )

    async with app_client(get_demo_service, demo([0.0, 0.05, 0.1, 0.15, 0.2])) as client:
        started = await client.post("/api/demo/start", json={"scenario": "smoke"})
        busy = await client.post("/api/demo/start", json={"scenario": "smoke"})
        finished = await wait_for_demo_state(client, "finished")
        bad_scenario = await client.post("/api/demo/start", json={"scenario": "burst"})
        bad_model = await client.post(
            "/api/demo/start", json={"scenario": "smoke", "model": "huge"}
        )
        bad_files = [
            await client.post("/api/demo/start", json={"scenario": "smoke", "questions_file": name})
            for name in ("../load_questions.txt", "load_questions_missing.txt")
        ]
    assert (started.status_code, started.json()["state"], started.json()["planned"]) == (
        200,
        "running",
        5,
    )
    assert (busy.status_code, busy.json()["error"]["type"]) == (409, "demo_busy")
    assert (finished["planned"], finished["sent"], finished["completed"], finished["ok"]) == (
        5,
        5,
        5,
        5,
    )
    assert len(repository.records) == 5
    assert bad_scenario.status_code == 422
    assert (bad_model.status_code, bad_model.json()["error"]["type"]) == (400, "unknown_model")
    assert all(
        (response.status_code, response.json()["error"]["type"]) == (400, "unknown_file")
        for response in bad_files
    )

    async with app_client(get_demo_service, demo([0.0, 0.05, 2.0, 4.0])) as client:
        await client.post("/api/demo/start", json={"scenario": "spike", "model": "small"})
        await asyncio.sleep(0.2)
        stopped = (await client.post("/api/demo/stop")).json()
        records_after_stop = len(repository.records)
        await asyncio.sleep(0.3)
    assert (stopped["state"], stopped["sent"]) == ("stopped", 2)
    assert len(repository.records) == records_after_stop

    for service, words in (
        (demo([0.0], mode="real"), "not in fake mode"),
        (demo([0.0], database="gateway.db"), "must contain 'fake'"),
    ):
        async with app_client(get_demo_service, service) as client:
            refused = await client.post("/api/demo/start", json={"scenario": "smoke"})
            status = await client.get("/api/demo/status")
        assert (refused.status_code, refused.json()["error"]["type"]) == (409, "demo_unavailable")
        assert words in refused.json()["error"]["message"]
        assert status.status_code == 200
        assert status.json()["available"] is False and words in status.json()["reason"]

    async with app_client(get_demo_service, demo([0.0] * 1001)) as client:
        too_many = await client.post("/api/demo/start", json={"scenario": "spike"})
    assert (too_many.status_code, too_many.json()["error"]["type"]) == (400, "too_many_requests")


async def test_gateway_and_experiments_pages(tmp_path: Path) -> None:
    """Both new pages and their assets are served, with the four-link navigation."""
    async with dashboard_client(tmp_path) as client:
        pages = [await client.get(path) for path in ("/gateway", "/experiments")]
        asset_paths = (
            "/gateway/assets/gateway.js",
            "/gateway/assets/gateway.css",
            "/experiments/assets/experiments.js",
            "/experiments/assets/experiments.css",
        )
        assets = [await client.get(path) for path in asset_paths]
    for page in pages:
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert '<nav id="site-nav">' in page.text
    assert [asset.status_code for asset in assets] == [200, 200, 200, 200]
    nav_script = (STATIC_ROOT / "shared" / "nav.js").read_text(encoding="utf-8")
    for path in ('"/chat"', '"/dashboard"', '"/gateway"', '"/experiments"'):
        assert path in nav_script, path
    assert_static_files_are_safe()


GEMINI_SECRET = "gemini-test-secret-456"
GEMINI_SPEC_FIELDS: dict[str, Any] = {
    "input_price_per_million_tokens": 0.1,
    "output_price_per_million_tokens": 0.4,
}


async def test_gemini_provider_request_response_and_errors() -> None:
    """Gemini requests carry the key in a header only; answers, empties and errors map cleanly."""
    requests: list[httpx.Request] = []
    replies: list[httpx.Response | Exception] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request and hand back the next canned reply."""
        requests.append(request)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    provider = GeminiProvider(
        api_key=GEMINI_SECRET,
        timeout_seconds=5.0,
        max_output_tokens=64,
        temperature=0.0,
        transport=httpx.MockTransport(handler),
    )
    spec = ModelSpec(name="gem", provider="gemini", model_id="gemini-test", **GEMINI_SPEC_FIELDS)
    messages = [
        Message(role="system", content="Be brief."),
        Message(role="system", content="Use English."),
        Message(role="user", content="Hi"),
        Message(role="assistant", content="Hello"),
        Message(role="user", content="2+2?"),
    ]
    usage = {"promptTokenCount": 12, "candidatesTokenCount": 3, "thoughtsTokenCount": 20}
    answer_parts = [{"text": "4"}, {"text": " exactly"}]
    replies.append(
        httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": answer_parts}, "finishReason": "STOP"}],
                "usageMetadata": usage,
            },
        )
    )
    result = await provider.complete(spec, messages)
    assert (result.text, result.input_tokens, result.output_tokens) == ("4 exactly", 12, 23)
    assert result.truncated is False
    sent = requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent"
    )
    assert sent.headers["x-goog-api-key"] == GEMINI_SECRET
    assert GEMINI_SECRET not in str(sent.url)
    body = json.loads(sent.content)
    assert body["systemInstruction"] == {"parts": [{"text": "Be brief.\n\nUse English."}]}
    assert [(turn["role"], turn["parts"][0]["text"]) for turn in body["contents"]] == [
        ("user", "Hi"),
        ("model", "Hello"),
        ("user", "2+2?"),
    ]
    assert body["generationConfig"] == {"maxOutputTokens": 64, "temperature": 0.0}
    assert "thinkingConfig" not in body["generationConfig"]
    assert "systemInstruction" not in provider.build_body([Message(role="user", content="Hi")])

    cut_off = {"content": {"parts": [{"text": "partial"}]}, "finishReason": "MAX_TOKENS"}
    replies.append(httpx.Response(200, json={"candidates": [cut_off]}))
    truncated = await provider.complete(spec, messages)
    assert (truncated.truncated, truncated.output_tokens) == (True, 0)

    empty = {"content": {"parts": [{"text": "  "}]}, "finishReason": "SAFETY"}
    blocked = {
        "promptFeedback": {"blockReason": "SAFETY"},
        "usageMetadata": {"promptTokenCount": 7},
    }
    replies.extend(
        [
            httpx.Response(200, json={"candidates": [empty], "usageMetadata": usage}),
            httpx.Response(200, json=blocked),
            httpx.Response(429, json={"error": {"message": "Resource has been exhausted"}}),
            httpx.Response(500, json={"error": {"message": "x" * 400}}),
            httpx.Response(403, text=f"bad key {GEMINI_SECRET}"),
            httpx.ReadTimeout("slow"),
            httpx.ConnectError("down"),
        ]
    )
    errors: list[ProviderError] = []
    for _ in range(7):
        with pytest.raises(ProviderError) as caught:
            await provider.complete(spec, messages)
        errors.append(caught.value)
    empty_error, blocked_error, limited, server, denied, timed_out, offline = errors
    assert classify_failure(str(empty_error)) == "empty_answer"
    assert (empty_error.input_tokens, empty_error.output_tokens) == (12, 23)
    assert empty_error.finish_reason == "safety"
    assert classify_failure(str(blocked_error)) == "empty_answer"
    assert "block_reason=SAFETY" in str(blocked_error)
    assert (blocked_error.input_tokens, blocked_error.finish_reason) == (7, None)
    assert classify_failure(str(limited)) == "rate_limit"
    assert str(server) == f"Gemini API error (HTTP 500): {'x' * 300}"
    assert classify_failure(str(denied)) == "auth"
    assert classify_failure(str(timed_out)) == "timeout"
    assert str(offline) == "Could not connect to Gemini"
    for error in errors:
        assert GEMINI_SECRET not in str(error)


def write_models_toml(path: Path, extra: str = "") -> Path:
    """Write a models file with one Groq and one Gemini model, plus any extra text."""
    path.write_text(
        '[models.small]\nprovider = "groq"\nmodel_id = "g-small"\n'
        "input_price_per_million_tokens = 0.1\noutput_price_per_million_tokens = 0.2\n"
        '[models.gem]\nprovider = "gemini"\nmodel_id = "gemini-x"\n'
        f"input_price_per_million_tokens = 0.1\noutput_price_per_million_tokens = 0.2\n{extra}",
        encoding="utf-8",
    )
    return path


async def test_provider_selection_and_model_specs(tmp_path: Path) -> None:
    """Each model reaches its own provider; keys are checked only in real mode; specs validate."""
    registry = ModelRegistry(write_models_toml(tmp_path / "models.toml"))
    fake_config = str(MODELS_CONFIG_PATH.parent / "fake.toml")

    def settings(mode: str, gemini_key: str | None) -> Settings:
        """Build settings without reading .env."""
        return Settings(
            _env_file=None,
            provider_mode=mode,
            groq_api_key="groq-test-key",
            gemini_api_key=gemini_key,
            fake_config_path=fake_config,
        )

    selector = build_llm_provider(settings("real", GEMINI_SECRET), registry)
    assert isinstance(selector, ProviderSelector)
    assert isinstance(selector.provider_for("groq"), GroqProvider)
    assert isinstance(selector.provider_for("gemini"), GeminiProvider)
    fake = build_llm_provider(settings("fake", None), registry)
    assert isinstance(fake, fake_provider.FakeProvider)
    assert isinstance(
        build_llm_provider(settings("fake", ""), registry), fake_provider.FakeProvider
    )
    for missing in (None, "", "   "):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            build_llm_provider(settings("real", missing), registry)
    with pytest.raises(ValueError, match="Unknown provider 'other'"):
        build_llm_provider(
            settings("real", GEMINI_SECRET),
            ModelRegistry(
                write_models_toml(
                    tmp_path / "other.toml",
                    '[models.odd]\nprovider = "other"\nmodel_id = "o"\n'
                    "input_price_per_million_tokens = 0\noutput_price_per_million_tokens = 0\n",
                )
            ),
        )

    routed = ProviderSelector({"groq": FakeProvider(), "gemini": FakeProvider(should_fail=True)})
    assert (await routed.complete(registry.get("small"), [])).text == "Paris"
    with pytest.raises(ProviderError):
        await routed.complete(registry.get("gem"), [])

    groq = GroqProvider(
        api_key="groq-test-key", timeout_seconds=5.0, max_output_tokens=64, temperature=0.0
    )
    plain = ModelSpec(name="s", provider="groq", model_id="g", **GEMINI_SPEC_FIELDS)
    low = plain.model_copy(update={"reasoning_effort": "low"})
    hello = [Message(role="user", content="Hi")]
    assert "reasoning_effort" not in groq.build_request(plain, hello)
    assert groq.build_request(low, hello)["reasoning_effort"] == "low"
    assert groq.build_request(low, hello)["model"] == "g"

    for extra in ('reasoning_effort = "max"\n', 'reasoning_effort = "low"\n'):
        with pytest.raises(ValueError, match="reasoning_effort"):
            ModelRegistry(write_models_toml(tmp_path / "bad.toml", extra))

    shipped = ModelRegistry(MODELS_CONFIG_PATH)
    assert shipped.names()[:2] == ["small", "big"]
    assert [
        (spec.model_id, spec.input_price_per_million_tokens, spec.output_price_per_million_tokens)
        for spec in (shipped.get("small"), shipped.get("big"))
    ] == [("openai/gpt-oss-20b", 0.075, 0.30), ("openai/gpt-oss-120b", 0.15, 0.60)]
    assert shipped.get("small").reasoning_effort is None
    assert shipped.get("big").reasoning_effort is None


async def test_gemini_model_discovery_and_probe() -> None:
    """Listing follows pages, keeps generateContent models, sorts and notes them; probes report."""
    list_requests: list[httpx.Request] = []
    pages = {
        None: {
            "models": [
                {
                    "name": "models/gemini-b-flash",
                    "displayName": "B Flash",
                    "inputTokenLimit": 1000,
                    "outputTokenLimit": 100,
                    "supportedGenerationMethods": ["generateContent", "countTokens"],
                },
                {
                    "name": "models/embedding-x",
                    "displayName": "Embedder",
                    "supportedGenerationMethods": ["embedContent"],
                },
            ],
            "nextPageToken": "page-2",
        },
        "page-2": {
            "models": [
                {
                    "name": "models/gemini-a-preview-05",
                    "displayName": "A Preview",
                    "inputTokenLimit": 2000,
                    "outputTokenLimit": 200,
                    "supportedGenerationMethods": ["generateContent"],
                },
                {
                    "name": "models/gemini-c-exp",
                    "displayName": "C Exp",
                    "supportedGenerationMethods": ["generateContent"],
                },
            ]
        },
    }

    def list_handler(request: httpx.Request) -> httpx.Response:
        """Serve the two canned list pages."""
        list_requests.append(request)
        return httpx.Response(200, json=pages[request.url.params.get("pageToken")])

    models = await fetch_models(GEMINI_SECRET, 5.0, transport=httpx.MockTransport(list_handler))
    assert len(list_requests) == 2
    for request in list_requests:
        assert request.url.path == "/v1beta/models"
        assert request.url.params["pageSize"] == "1000"
        assert request.headers["x-goog-api-key"] == GEMINI_SECRET
        assert GEMINI_SECRET not in str(request.url)
    assert list_requests[1].url.params["pageToken"] == "page-2"
    usable = generate_content_models(models)
    assert [model.model_id for model in usable] == [
        "gemini-a-preview-05",
        "gemini-b-flash",
        "gemini-c-exp",
    ]
    assert [model_note(model.model_id) for model in usable] == ["preview", "", "experimental"]
    table = render_models_table(models)
    lines = table.splitlines()
    assert lines[0].split() == [
        "id",
        "display",
        "name",
        "input",
        "limit",
        "output",
        "limit",
        "note",
    ]
    assert lines[2].split() == ["gemini-a-preview-05", "A", "Preview", "2000", "200", "preview"]
    assert lines[3].split() == ["gemini-b-flash", "B", "Flash", "1000", "100"]
    assert lines[4].split() == ["gemini-c-exp", "C", "Exp", "?", "?", "experimental"]
    assert lines[-1] == FREE_TIER_REMINDER
    assert "models/" not in table and "embedding-x" not in table
    assert GEMINI_SECRET not in table

    probe_bodies: list[dict[str, Any]] = []
    probe_replies = [
        httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "OK"}]}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 6,
                    "candidatesTokenCount": 1,
                    "thoughtsTokenCount": 9,
                },
            },
        ),
        httpx.Response(429, json={"error": {"message": "Quota exceeded"}}),
    ]

    def probe_handler(request: httpx.Request) -> httpx.Response:
        """Record the probe body and return the next reply."""
        probe_bodies.append(json.loads(request.content))
        return probe_replies.pop(0)

    provider = GeminiProvider(
        api_key=GEMINI_SECRET,
        timeout_seconds=5.0,
        max_output_tokens=64,
        temperature=0.0,
        transport=httpx.MockTransport(probe_handler),
    )
    good = await probe_model(provider, "gemini-b-flash")
    bad = await probe_model(provider, "gemini-c-exp")
    assert PROBE_TEXT == "Reply with the word OK."
    for body in probe_bodies:
        assert body["contents"] == [{"role": "user", "parts": [{"text": PROBE_TEXT}]}]
        assert "systemInstruction" not in body
    assert (good.ok, good.input_tokens, good.output_tokens, good.finish_reason) == (
        True,
        6,
        10,
        "stop",
    )
    good_line = render_probe_line(good)
    assert good_line.startswith("gemini-b-flash  OK  latency=")
    assert "output_tokens=10" in good_line and "finish_reason=stop" in good_line
    bad_line = render_probe_line(bad)
    assert bad.ok is False and "FAILED" in bad_line
    assert "error=Gemini API error (HTTP 429): Quota exceeded" in bad_line
    assert GEMINI_SECRET not in good_line + bad_line
