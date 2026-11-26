"""Sends eval questions through the gateway and grades each answer."""

import asyncio
from datetime import UTC, datetime

from app.clients.base import GatewayClient
from app.models.eval import EvalOutcome, EvalQuestion, EvalRunResult, GatewayReply
from app.services.eval_checker import check_answer


def should_retry(reply: GatewayReply) -> bool:
    """Retry connection errors, timeouts, rate limits (429), and server errors (5xx)."""
    status = reply.http_status
    return status != 200 and (status == 0 or status == 429 or status >= 500)


class EvalRunner:
    """Runs questions against one model with limited concurrency and retries."""

    def __init__(
        self,
        client: GatewayClient,
        concurrency: int,
        retries: int,
        retry_wait_seconds: float,
    ) -> None:
        self._client: GatewayClient = client
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(concurrency)
        self._retries: int = retries
        self._retry_wait_seconds: float = retry_wait_seconds

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
        reply, attempts = await self._ask_with_retries(question.question, model)
        return self._grade(question, reply, attempts)

    async def _ask_with_retries(self, question: str, model: str) -> tuple[GatewayReply, int]:
        """Ask until the reply should not be retried or the retries run out."""
        attempts = 0
        while True:
            attempts += 1
            async with self._semaphore:
                reply = await self._client.ask(question, model)
            if not should_retry(reply) or attempts > self._retries:
                return reply, attempts
            await asyncio.sleep(self._retry_wait_seconds)

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
