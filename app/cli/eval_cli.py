"""Command line: run the labelled eval set through the gateway against one model.

Run with: uv run python -m app.cli.eval_cli --model small
"""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.cli.load_test_cli import fetch_health, unknown_model_problem
from app.cli.stats_cli import positive_int
from app.clients.gateway_client import HttpGatewayClient
from app.models.eval import EvalQuestion, EvalRunResult
from app.services.eval_loader import load_eval_set
from app.services.eval_report import summarize
from app.services.eval_runner import EvalRunner
from app.views.eval_text_view import render_eval_set_overview, render_eval_summary

RESULTS_FOLDER = Path("results")
MAX_CONNECTIONS: int = 50


def non_negative_int(text: str) -> int:
    """Parse an integer that must be 0 or more."""
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"must be a whole number, got {text!r}") from error
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command-line options."""
    parser = argparse.ArgumentParser(description="Run the eval set through the LLM gateway.")
    parser.add_argument("--model", help="model name sent to the gateway (required to run)")
    parser.add_argument("--label", help="name for this run (default: the model name)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--eval-set", default="eval/eval_set.jsonl")
    parser.add_argument("--split", choices=["all", "tune", "test"], default="all")
    parser.add_argument("--limit", type=positive_int, help="use only the first N questions")
    parser.add_argument("--concurrency", type=positive_int, default=2)
    parser.add_argument("--retries", type=non_negative_int, default=2)
    parser.add_argument("--retry-wait-seconds", type=float, default=5.0)
    parser.add_argument(
        "--min-interval-seconds",
        type=float,
        default=0.0,
        help="minimum seconds between request starts (only with --concurrency 1)",
    )
    parser.add_argument("--validate-only", action="store_true", help="check the file and exit")
    args = parser.parse_args(argv)
    if not args.validate_only and args.model is None:
        parser.error("--model is required unless --validate-only is given")
    return args


def select_questions(
    questions: list[EvalQuestion], split: str, limit: int | None
) -> list[EvalQuestion]:
    """Apply the split filter, then keep the first `limit` questions."""
    selected = [question for question in questions if split in ("all", question.split)]
    return selected[:limit] if limit is not None else selected


def save_run(run: EvalRunResult) -> Path:
    """Save the full run as JSON in results/ and return the file path."""
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = RESULTS_FOLDER / f"eval_{run.label}_{stamp}.json"
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path


async def run_eval(args: argparse.Namespace, questions: list[EvalQuestion]) -> int:
    """Check the gateway, run the questions, save, and print the summary."""
    limits = httpx.Limits(max_connections=MAX_CONNECTIONS)
    async with httpx.AsyncClient(base_url=args.base_url, limits=limits) as http_client:
        health = await fetch_health(http_client)
        if health is None:
            print(f"Gateway not reachable at {args.base_url}", file=sys.stderr)
            return 1
        mode, known_names = health
        model_problem = unknown_model_problem(args.model, known_names)
        if model_problem is not None:
            print(model_problem, file=sys.stderr)
            return 1
        if mode == "fake":
            print(
                "Eval needs real model answers, but the gateway is in fake mode.", file=sys.stderr
            )
            return 1
        runner = EvalRunner(
            client=HttpGatewayClient(args.base_url, http_client),
            concurrency=args.concurrency,
            retries=args.retries,
            retry_wait_seconds=args.retry_wait_seconds,
            min_interval_seconds=args.min_interval_seconds,
            log=print,
        )
        print(f"running {len(questions)} questions against model '{args.model}'...")
        run = await runner.run(
            questions, args.model, args.label or args.model, args.split, args.eval_set
        )
    saved_path = save_run(run)
    print(render_eval_summary(summarize(run), str(saved_path), runner.rate_limit_retries))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the eval command and return the process exit code."""
    args = parse_args(argv)
    if args.min_interval_seconds > 0 and args.concurrency != 1:
        print("--min-interval-seconds is only allowed with --concurrency 1", file=sys.stderr)
        return 1
    try:
        questions = load_eval_set(args.eval_set)
    except (FileNotFoundError, ValueError) as error:
        print(f"Invalid eval set: {error}", file=sys.stderr)
        return 1
    if args.validate_only:
        print(render_eval_set_overview(questions, args.eval_set))
        return 0
    selected = select_questions(questions, args.split, args.limit)
    if not selected:
        print("No questions match the split and limit.", file=sys.stderr)
        return 1
    return asyncio.run(run_eval(args, selected))


if __name__ == "__main__":
    sys.exit(main())
