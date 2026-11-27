"""Command line: send open-loop load to the gateway, like real users would.

Run with: uv run python -m app.cli.load_test_cli --scenario normal
"""

import argparse
import asyncio
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from app.config import get_settings
from app.services.load_plan import SCENARIOS, build_schedule

REQUEST_TIMEOUT_SECONDS: float = 120.0
MAX_CONNECTIONS: int = 500
CONNECTION_OR_TIMEOUT: str = "connection/timeout"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command-line options."""
    parser = argparse.ArgumentParser(description="Send scheduled load to the LLM gateway.")
    parser.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", help="model name to send; omit to use the gateway default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--questions", default="config/load_questions.txt")
    parser.add_argument(
        "--allow-real", action="store_true", help="allow non-smoke scenarios in real mode"
    )
    return parser.parse_args(argv)


def load_questions(path: Path) -> list[str]:
    """Read one question per non-empty line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def build_body(question: str, model: str | None) -> dict[str, Any]:
    """Build the chat request body, adding the model only when one was given."""
    body: dict[str, Any] = {"messages": [{"role": "user", "content": question}]}
    if model is not None:
        body["model"] = model
    return body


async def fetch_health(client: httpx.AsyncClient) -> tuple[str, list[str]] | None:
    """Return the gateway's provider mode and model names from /health, or None if unreachable."""
    try:
        response = await client.get("/health")
        response.raise_for_status()
        body = response.json()
        models = body.get("models")
        names = [str(name) for name in models] if isinstance(models, list) else []
        return str(body.get("mode")), names
    except (httpx.HTTPError, ValueError, AttributeError):
        return None


def unknown_model_problem(model: str | None, known_names: list[str]) -> str | None:
    """Return the unknown-model message when the gateway does not know this model, else None."""
    if model is None or model == "auto" or model in known_names:
        return None
    return (
        f"Unknown model '{model}'. The gateway knows: {', '.join(known_names)}. "
        "Restart the gateway if you just edited config/models.toml."
    )


async def send_one(client: httpx.AsyncClient, body: dict[str, Any]) -> str:
    """Send one chat request and return its HTTP status code, or a connection/timeout label."""
    try:
        response = await client.post("/v1/chat", json=body)
    except httpx.TransportError:
        return CONNECTION_OR_TIMEOUT
    return str(response.status_code)


async def send_all(
    client: httpx.AsyncClient,
    schedule: list[float],
    questions: list[str],
    model: str | None,
    seed: int,
) -> list[str]:
    """Fire each request at its scheduled time without waiting for earlier answers."""
    picker = random.Random(seed)
    started = time.monotonic()
    tasks: list[asyncio.Task[str]] = []
    for send_at in schedule:
        delay = started + send_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        body = build_body(picker.choice(questions), model)
        tasks.append(asyncio.create_task(send_one(client, body)))
    return list(await asyncio.gather(*tasks))


def print_summary(outcomes: list[str], wall_seconds: float) -> None:
    """Print request counts by outcome and the stats command to run next."""
    counts = Counter(outcomes)
    print(f"requests sent: {len(outcomes)}")
    print(f"HTTP 200: {counts.pop('200', 0)}")
    connection_or_timeout = counts.pop(CONNECTION_OR_TIMEOUT, 0)
    for status in sorted(counts):
        print(f"HTTP {status}: {counts[status]}")
    print(f"connection/timeout: {connection_or_timeout}")
    print(f"wall-clock seconds: {wall_seconds:.1f}")
    minutes = math.ceil(wall_seconds / 60) + 1
    print("next:")
    print(
        f"  uv run python -m app.cli.stats_cli --db {get_settings().database_path} "
        f"--minutes {minutes}"
    )


def startup_problem(health: tuple[str, list[str]] | None, args: argparse.Namespace) -> str | None:
    """Return the reason the run must not start, or None when it can go ahead."""
    if health is None:
        return f"Gateway not reachable at {args.base_url}"
    mode, known_names = health
    model_problem = unknown_model_problem(args.model, known_names)
    if model_problem is not None:
        return model_problem
    if mode == "real" and args.scenario != "smoke" and not args.allow_real:
        return (
            f"Refusing to run '{args.scenario}' against a real-mode gateway: it would hit "
            "Groq rate limits and spoil the numbers. Use --scenario smoke, or pass --allow-real."
        )
    if not Path(args.questions).is_file():
        return f"Questions file not found: {args.questions}"
    return None


async def run(args: argparse.Namespace) -> int:
    """Check the gateway, run the scenario, and print the summary. Return the exit code."""
    limits = httpx.Limits(max_connections=MAX_CONNECTIONS, max_keepalive_connections=100)
    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=REQUEST_TIMEOUT_SECONDS, limits=limits
    ) as client:
        health = await fetch_health(client)
        problem = startup_problem(health, args)
        if problem is not None:
            print(problem, file=sys.stderr)
            return 1
        mode = health[0] if health is not None else None
        schedule = build_schedule(args.scenario, args.seed)
        questions = load_questions(Path(args.questions))
        print(f"scenario: {args.scenario} | planned requests: {len(schedule)} | mode: {mode}")
        started = time.monotonic()
        outcomes = await send_all(client, schedule, questions, args.model, args.seed)
        wall_seconds = time.monotonic() - started
    print_summary(outcomes, wall_seconds)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the load test command and return the process exit code."""
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
