"""Command line: replay saved small and big eval runs through the router (no model calls).

Run with:
    uv run python -m app.cli.router_replay_cli --small results\\eval_small_X.json
        --big results\\eval_big_X.json
"""

import argparse
import sys

from app.cli.eval_compare_cli import load_run
from app.config import get_settings
from app.services.router_replay import build_replay_table
from app.services.routing_config import load_routing_config
from app.views.replay_text_view import render_replay_table


def threshold_list(text: str) -> list[int]:
    """Parse comma-separated whole numbers, for example "0,1,2"."""
    try:
        values = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"must be whole numbers like 0,1,2, got {text!r}"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError("needs at least one threshold")
    return values


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command-line options."""
    parser = argparse.ArgumentParser(description="Replay saved eval runs through the router.")
    parser.add_argument("--small", required=True, help="saved eval result JSON for small")
    parser.add_argument("--big", required=True, help="saved eval result JSON for big")
    parser.add_argument("--split", choices=["all", "tune", "test"], default="all")
    parser.add_argument("--thresholds", type=threshold_list, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--no-fallback", action="store_true", help="turn the fallback off")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the replay command and return the process exit code."""
    args = parse_args(argv)
    try:
        small_run = load_run(args.small)
        big_run = load_run(args.big)
        config = load_routing_config(get_settings().routing_config_path)
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    table = build_replay_table(
        small_run, big_run, config, args.thresholds, not args.no_fallback, args.split
    )
    print(render_replay_table(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
