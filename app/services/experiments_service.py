"""Reads saved eval runs and runs the existing summary, replay, and compare functions on them."""

import re
from pathlib import Path

from pydantic import ValidationError

from app.models.eval import EvalRunResult
from app.models.routing import RoutingConfig
from app.services.cascade_replay import build_cascade_table
from app.services.eval_report import compare_runs, summarize
from app.services.router_replay import build_replay_table
from app.views.experiments_view import (
    CompareResponse,
    ReplayResponse,
    RunInfo,
    RunListResponse,
    SkippedFile,
)

RUN_FILE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+\.json$")
RUN_FILE_GLOB = "eval_*.json"
# The same settings the command-line replay tools use by default.
ROUTER_THRESHOLDS: list[int] = [0, 1, 2, 3, 4, 5]
CASCADE_LIMITS: list[int] = [500, 1000, 1500, 2000, 3000, 4000]
SKIP_REASON_CHARS: int = 160


class UnknownFileError(Exception):
    """Raised when a file name is not one of the listed files."""

    def __init__(self, name: str, valid_names: list[str], detail: str | None = None) -> None:
        self.name: str = name
        valid = ", ".join(valid_names) or "none"
        message = detail or f"Unknown file '{name}'."
        super().__init__(f"{message} Valid names: {valid}")


class RunsNotComparableError(Exception):
    """Raised when two runs were not made on the same eval set."""


class ExperimentsService:
    """Lists saved eval runs and computes summaries, replays, and comparisons (read-only)."""

    def __init__(self, results_dir: str, routing_config: RoutingConfig) -> None:
        self._results_dir: Path = Path(results_dir)
        self._routing_config: RoutingConfig = routing_config

    def file_names(self) -> list[str]:
        """Return the eval_*.json files directly in the results folder, sorted."""
        if not self._results_dir.is_dir():
            return []
        return sorted(
            path.name
            for path in self._results_dir.glob(RUN_FILE_GLOB)
            if path.is_file() and RUN_FILE_PATTERN.fullmatch(path.name)
        )

    def load(self, name: str) -> EvalRunResult:
        """Read one listed run file, or raise UnknownFileError."""
        names = self.file_names()
        if not RUN_FILE_PATTERN.fullmatch(name) or name not in names:
            raise UnknownFileError(name, names)
        try:
            return read_run(self._results_dir / name)
        except ValueError as error:
            raise UnknownFileError(
                name, names, f"File '{name}' is not a valid eval run."
            ) from error

    def runs(self, split: str) -> RunListResponse:
        """Summarize every readable run, newest first; unreadable files are listed as skipped."""
        runs: list[RunInfo] = []
        skipped: list[SkippedFile] = []
        for name in self.file_names():
            try:
                run = read_run(self._results_dir / name)
            except ValueError as error:
                skipped.append(SkippedFile(file=name, reason=str(error)[:SKIP_REASON_CHARS]))
                continue
            filtered = filter_split(run, split)
            if filtered is not None:
                runs.append(RunInfo.from_summary(name, filtered, summarize(filtered)))
        runs.sort(key=lambda info: info.created_at, reverse=True)
        return RunListResponse(split=split, runs=runs, skipped=skipped)

    def replay(self, small: str, big: str, split: str) -> ReplayResponse:
        """Run the existing router and cascade replays on two comparable runs."""
        small_run, big_run = self._comparable_pair(small, big, split)
        router = build_replay_table(
            small_run, big_run, self._routing_config, ROUTER_THRESHOLDS, True, split
        )
        cascade = build_cascade_table(small_run, big_run, split, CASCADE_LIMITS)
        return ReplayResponse.from_tables(small, big, router, cascade)

    def compare(self, a: str, b: str, split: str) -> CompareResponse:
        """Run the existing compare_runs() on two comparable runs."""
        run_a, run_b = self._comparable_pair(a, b, split)
        report = compare_runs(run_a, run_b)
        return CompareResponse.from_report(split, (a, b), (run_a.label, run_b.label), report)

    def _comparable_pair(
        self, first: str, second: str, split: str
    ) -> tuple[EvalRunResult, EvalRunResult]:
        """Load two runs made on the same eval set, filtered to the split."""
        run_a, run_b = self.load(first), self.load(second)
        if same_eval_set(run_a) != same_eval_set(run_b):
            raise RunsNotComparableError(
                f"'{first}' uses {run_a.eval_set_path} but '{second}' uses {run_b.eval_set_path}; "
                "only runs of the same eval set can be compared."
            )
        return (
            filter_split(run_a, split) or run_a.model_copy(update={"outcomes": []}),
            filter_split(run_b, split) or run_b.model_copy(update={"outcomes": []}),
        )


def read_run(path: Path) -> EvalRunResult:
    """Parse a file as an eval run, raising ValueError with a short reason."""
    try:
        return EvalRunResult.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        first = error.errors()[0] if error.errors() else {"msg": str(error)}
        raise ValueError(f"not a valid eval run ({first['msg']})") from error
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"could not be read ({error})") from error


def same_eval_set(run: EvalRunResult) -> str:
    """Return the eval set path with one kind of slash, for comparing runs."""
    return run.eval_set_path.replace("\\", "/")


def filter_split(run: EvalRunResult, split: str) -> EvalRunResult | None:
    """Keep only outcomes of the split ("all" keeps everything); None if nothing is left."""
    if split == "all":
        return run if run.outcomes else None
    outcomes = [outcome for outcome in run.outcomes if outcome.split == split]
    return run.model_copy(update={"outcomes": outcomes}) if outcomes else None
