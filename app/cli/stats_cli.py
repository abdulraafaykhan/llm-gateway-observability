"""Command line: print request stats from the SQLite request log.

Run with: uv run python -m app.cli.stats_cli [--minutes N] [--db PATH]
"""

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.config import get_settings
from app.repositories.sqlite_repository import SqliteRequestRepository
from app.services.stats_service import StatsService
from app.views.stats_text_view import render_stats_report


def positive_int(text: str) -> int:
    """Parse an integer that must be 1 or more."""
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"must be a whole number, got {text!r}") from error
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command-line options."""
    parser = argparse.ArgumentParser(description="Print stats from the LLM gateway request log.")
    parser.add_argument(
        "--minutes", type=positive_int, help="only include requests from the last N minutes"
    )
    parser.add_argument("--db", help="path to the SQLite database (default: DATABASE_PATH)")
    return parser.parse_args(argv)


async def build_report_text(database_path: Path, since: str | None) -> str:
    """Read the database and return the rendered report."""
    service = StatsService(SqliteRequestRepository(database_path))
    report = await service.build_report(since)
    return render_stats_report(report)


def main(argv: list[str] | None = None) -> int:
    """Run the stats command and return the process exit code."""
    args = parse_args(argv)
    database_path = Path(args.db) if args.db else Path(get_settings().database_path)
    if not database_path.is_file():
        print(f"Database not found: {database_path}", file=sys.stderr)
        return 1
    since = None
    if args.minutes is not None:
        since = (datetime.now(UTC) - timedelta(minutes=args.minutes)).isoformat()
    print(asyncio.run(build_report_text(database_path, since)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
