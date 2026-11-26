"""Tests for POST /v1/chat and the request log (no network, no API key)."""

import asyncio
import time
from pathlib import Path
from typing import Any

import aiosqlite
import httpx
import pytest

from app.clients.base import GatewayClient
from app.dependencies import get_chat_service
from app.main import create_app
from app.models.completion import CompletionResult, Message
from app.models.eval import EvalOutcome, EvalQuestion, EvalRunResult, GatewayReply
from app.models.fake_profile import FakeProfile
from app.models.model_spec import ModelSpec
from app.models.request_record import RequestRecord
from app.models.routing import RouteDecision
from app.providers import fake_provider
from app.providers.base import LLMProvider, ProviderError
from app.repositories.base import RequestRepository
from app.repositories.sqlite_repository import SqliteRequestRepository
from app.routing.base import Router
from app.routing.factory import build_rule_router
from app.services.chat_service import ChatService
from app.services.eval_checker import check_answer
from app.services.eval_loader import load_eval_set
from app.services.eval_report import compare_runs
from app.services.eval_runner import EvalRunner
from app.services.load_plan import build_schedule
from app.services.model_registry import ModelRegistry
from app.services.router_replay import replay, replay_baselines
from app.services.routing_config import load_routing_config, validate_routing_models
from app.services.stats_service import StatsService, compute_group_stats, percentile
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

    async def initialize(self) -> None:
        """Nothing to create for an in-memory list."""

    async def add(self, record: RequestRecord) -> None:
        """Append the record to the list."""
        self.records.append(record)

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
) -> httpx.AsyncClient:
    """Build an HTTP client for an app whose ChatService uses the given fakes."""
    app = create_app()
    chat_registry = registry or ModelRegistry(MODELS_CONFIG_PATH)
    chat_repository = repository or InMemoryRequestRepository()
    routing_config = load_routing_config(str(ROUTING_CONFIG_PATH))
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
    await repository.add(make_record(created_at="2026-11-25T10:00:00+00:00", question="first"))
    await repository.add(make_record(created_at="2026-11-25T10:00:01+00:00", question="second"))
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
        await repository.add(make_record(model_name=model_name))
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
        await repository.add(make_record(created_at=created_at, question=f"q{index}"))
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


async def test_fallback_on_cut_off_only_for_auto() -> None:
    """A cut-off small answer falls back for auto, but not when small was asked for."""
    provider = PerModelProvider({"small": answer(truncated=True), "big": answer("From big")})
    repository = InMemoryRequestRepository()
    async with build_client(provider, repository, router=FixedRouter("small")) as client:
        auto_response = await client.post("/v1/chat", json=AUTO_BODY)
        small_response = await client.post("/v1/chat", json={**VALID_BODY, "model": "small"})
    assert (auto_response.json()["model"], auto_response.json()["fallback_used"]) == ("big", True)
    assert small_response.json()["truncated"] is True
    auto_row, small_row = repository.records
    assert (auto_row.status, auto_row.fallback_used) == ("ok", True)
    assert (small_row.status, small_row.fallback_used) == ("truncated", False)


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
        make_record(question="new", requested_model="auto", route_reason="r", fallback_used=True)
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
