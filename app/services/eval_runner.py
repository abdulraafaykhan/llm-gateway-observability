"""Sends eval questions through the gateway and grades each answer."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from app.clients.base import GatewayClient
from app.models.eval import EvalOutcome, EvalQuestion, EvalRunResult, GatewayReply
from app.services.eval_checker import check_answer
from app.services.failure_types import classify_failure


def is_rate_limited(reply: GatewayReply) -> bool:
    """True for an HTTP 429 or any failure that classify_failure() calls a rate limit."""
    if reply.http_status == 200:
        return False
    return reply.http_status == 429 or classify_failure(reply.error_message) == "rate_limit"


def should_retry(reply: GatewayReply) -> bool:
    """Retry connection errors, timeouts, rate limits, and server errors (5xx)."""
    status = reply.http_status
    if status == 200:
        return False
    return status == 0 or status >= 500 or is_rate_limited(reply)


def _no_log(line: str) -> None:
    """Default retry logger: print nothing."""


class EvalRunner:
    """Runs questions against one model with limited concurrency, retries, and pacing."""

    def __init__(
        self,
        client: GatewayClient,
        concurrency: int,
        retries: int,
        retry_wait_seconds: float,
        min_interval_seconds: float = 0.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] = _no_log,
    ) -> None:
        if min_interval_seconds > 0 and concurrency != 1:
            raise ValueError("--min-interval-seconds is only allowed with --concurrency 1")
        self._client: GatewayClient = client
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(concurrency)
        self._retries: int = retries
        self._retry_wait_seconds: float = retry_wait_seconds
        self._min_interval_seconds: float = min_interval_seconds
        self._sleep: Callable[[float], Awaitable[None]] = sleep
        self._clock: Callable[[], float] = clock
        self._log: Callable[[str], None] = log
        self._last_start: float | None = None
        self.rate_limit_retries: int = 0

    async def run(
        self,
        questions: list[EvalQuestion],
        model: str,
        label: str,
        split_filter: str,
        eval_set_path: str,
    ) -> EvalRunResult:
        """Ask every question, grade the answers, and return them in question order."""
        created_at = datetime.now(UTC).isoformat()
        outcomes = await asyncio.gather(
            *(self._evaluate(question, model) for question in questions)
        )
        return EvalRunResult(
            label=label,
            model_requested=model,
            created_at=created_at,
            split_filter=split_filter,
            eval_set_path=eval_set_path,
            outcomes=list(outcomes),
        )

    async def _evaluate(self, question: EvalQuestion, model: str) -> EvalOutcome:
        """Ask one question (with retries) and grade the final reply."""
        reply, attempts = await self._ask_with_retries(question, model)
        return self._grade(question, reply, attempts)

    async def _ask_with_retries(
        self, question: EvalQuestion, model: str
    ) -> tuple[GatewayReply, int]:
        """Ask until the reply should not be retried or the retries run out.

        The concurrency slot is held during the retry wait, so with --concurrency 1 the
        whole run pauses instead of other questions using up the quota in the meantime.
        """
        max_attempts = self._retries + 1
        attempts = 0
        async with self._semaphore:
            while True:
                attempts += 1
                await self._pace()
                reply = await self._client.ask(question.question, model)
                if not should_retry(reply) or attempts >= max_attempts:
                    return reply, attempts
                if is_rate_limited(reply):
                    self.rate_limit_retries += 1
                self._log(
                    f"retrying {question.id} in {self._retry_wait_seconds:g} s "
                    f"(attempt {attempts + 1} of {max_attempts})"
                )
                await self._sleep(self._retry_wait_seconds)

    async def _pace(self) -> None:
        """Wait so two request starts are at least min_interval_seconds apart."""
        if self._min_interval_seconds > 0 and self._last_start is not None:
            wait = self._last_start + self._min_interval_seconds - self._clock()
            if wait > 0:
                await self._sleep(wait)
        self._last_start = self._clock()

    def _grade(self, question: EvalQuestion, reply: GatewayReply, attempts: int) -> EvalOutcome:
        """Turn a reply into a passed, failed, or error outcome."""
        answered = reply.http_status == 200 and reply.answer is not None
        if not answered:
            status = "error"
        elif reply.truncated:
            status = "failed"
        else:
            status = "passed" if check_answer(question, reply.answer or "") else "failed"
        error_message = None
        if status == "error":
            error_message = f"http {reply.http_status}: {reply.error_message}"
        return EvalOutcome(
            **question.model_dump(),
            status=status,
            truncated=answered and reply.truncated,
            answer=reply.answer,
            answered_by=reply.model,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            latency_ms=reply.latency_ms,
            cost_usd=reply.cost_usd if answered else 0.0,
            error_message=error_message,
            attempts=attempts,
            route_reason=reply.route_reason,
            fallback_used=reply.fallback_used,
        )
