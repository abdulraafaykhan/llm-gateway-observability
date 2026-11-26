"""Command line: compare two saved eval runs (A then B).

Run with:
    uv run python -m app.cli.eval_compare_cli results/eval_small_X.json results/eval_big_X.json
"""

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from app.models.eval import EvalRunResult
from app.services.eval_report import compare_runs
from app.views.eval_text_view import render_comparison


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the two result file paths."""
    parser = argparse.ArgumentParser(description="Compare two saved eval runs.")
    parser.add_argument("run_a", help="saved eval result JSON for run A")
    parser.add_argument("run_b", help="saved eval result JSON for run B")
    return parser.parse_args(argv)


def load_run(path: str) -> EvalRunResult:
    """Load one saved run, raising ValueError with a clear message if it cannot be read."""
    run_path = Path(path)
    if not run_path.is_file():
        raise ValueError(f"Eval result file not found: {run_path}")
    try:
        return EvalRunResult.model_validate_json(run_path.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise ValueError(f"Not a valid eval result file: {run_path} ({error})") from error


def main(argv: list[str] | None = None) -> int:
    """Run the compare command and return the process exit code."""
    args = parse_args(argv)
    try:
        run_a = load_run(args.run_a)
        run_b = load_run(args.run_b)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print(render_comparison(compare_runs(run_a, run_b), run_a.label, run_b.label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
