"""Command line: replay saved small and big eval runs as a token cascade (no model calls).

Run with:
    uv run python -m app.cli.cascade_replay_cli --small results\\eval_small_X.json
        --big results\\eval_big_X.json
"""

import argparse
import sys

from app.cli.eval_compare_cli import load_run
from app.services.cascade_replay import build_cascade_table
from app.views.cascade_text_view import render_cascade_tables

SPLITS_FOR_OPTION: dict[str, list[str]] = {
    "both": ["tune", "test"],
    "all": ["all"],
    "tune": ["tune"],
    "test": ["test"],
}


def limit_list(text: str) -> list[int]:
    """Parse comma-separated positive whole numbers, for example "500,1000"."""
    try:
        values = [int(part.strip()) for part in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"must be positive whole numbers like 500,1000, got {text!r}"
        ) from error
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError(f"must be positive whole numbers, got {text!r}")
    return values


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command-line options."""
    parser = argparse.ArgumentParser(description="Replay saved eval runs as a token cascade.")
    parser.add_argument("--small", required=True, help="saved eval result JSON for small")
    parser.add_argument("--big", required=True, help="saved eval result JSON for big")
    parser.add_argument("--split", choices=list(SPLITS_FOR_OPTION), default="both")
    parser.add_argument("--limits", type=limit_list, default=[500, 1000, 1500, 2000, 3000, 4000])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the cascade replay command and return the process exit code."""
    args = parse_args(argv)
    try:
        small_run = load_run(args.small)
        big_run = load_run(args.big)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    tables = [
        build_cascade_table(small_run, big_run, split, args.limits)
        for split in SPLITS_FOR_OPTION[args.split]
    ]
    print(render_cascade_tables(tables))
    return 0


if __name__ == "__main__":
    sys.exit(main())
