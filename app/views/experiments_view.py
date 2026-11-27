"""JSON shapes for the Experiments page: saved eval runs, replays, and run comparison."""

from pydantic import BaseModel

from app.models.cascade import CascadeRow, CascadeTable, TokenStats
from app.models.eval import (
    ComparisonGroup,
    ComparisonReport,
    EvalRunResult,
    EvalSummary,
    RoutingCounts,
    ScoreCount,
)
from app.models.routing import ReplayRow, ReplayTable

QUESTION_PREVIEW_CHARS: int = 120


class RunInfo(BaseModel):
    """One saved eval run with the numbers from the eval summary."""

    file: str
    label: str
    model_requested: str
    created_at: str
    split_filter: str
    eval_set_path: str
    questions: int
    passed: int
    failed: int
    errors: int
    truncated: int
    score_percent: float
    by_difficulty: dict[str, ScoreCount]
    total_cost_usd: float
    latency_avg_ms: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    routing: RoutingCounts | None

    @classmethod
    def from_summary(cls, file: str, run: EvalRunResult, summary: EvalSummary) -> "RunInfo":
        """Combine the run's metadata with its summary numbers."""
        return cls(
            file=file,
            label=run.label,
            model_requested=run.model_requested,
            created_at=run.created_at,
            split_filter=run.split_filter,
            eval_set_path=run.eval_set_path,
            questions=summary.questions,
            passed=summary.passed,
            failed=summary.failed,
            errors=summary.errors,
            truncated=summary.truncated,
            score_percent=summary.score_percent,
            by_difficulty={
                name: score for name, score in summary.by_difficulty.items() if score.total
            },
            total_cost_usd=summary.total_cost_usd,
            latency_avg_ms=summary.latency_avg_ms,
            latency_p50_ms=summary.latency_p50_ms,
            latency_p95_ms=summary.latency_p95_ms,
            routing=summary.routing,
        )


class SkippedFile(BaseModel):
    """A results file that could not be read as an eval run."""

    file: str
    reason: str


class RunListResponse(BaseModel):
    """Body of GET /api/experiments/runs."""

    split: str
    runs: list[RunInfo]
    skipped: list[SkippedFile]


class RouterReplayRowView(BaseModel):
    """One router replay row, as structured data."""

    label: str
    to_small: int
    to_big: int
    fallbacks: int
    passed: int
    total: int
    score_percent: float
    cost_usd: float

    @classmethod
    def from_row(cls, row: ReplayRow) -> "RouterReplayRowView":
        """Add the score to a replay row."""
        return cls(**row.model_dump(), score_percent=percent_of(row.passed, row.total))


class CascadeRowView(BaseModel):
    """One cascade replay row, as structured data."""

    label: str
    limit: int | None
    kept_small: int
    escalated: int
    passed: int
    total: int
    rescued: int
    lost: int
    score_percent: float
    cost_usd: float
    cost_vs_all_big_percent: float | None

    @classmethod
    def from_row(cls, row: CascadeRow, all_big_cost: float) -> "CascadeRowView":
        """Add the score and the cost relative to "all big" to a cascade row."""
        relative = row.cost_usd / all_big_cost * 100 if all_big_cost else None
        return cls(
            **row.model_dump(),
            score_percent=percent_of(row.passed, row.total),
            cost_vs_all_big_percent=relative,
        )


class ReplayResponse(BaseModel):
    """Body of GET /api/experiments/replay."""

    split: str
    small_file: str
    big_file: str
    small_label: str
    big_label: str
    questions: int
    router_rows: list[RouterReplayRowView]
    cascade_rows: list[CascadeRowView]
    small_tokens: TokenStats | None

    @classmethod
    def from_tables(
        cls, small_file: str, big_file: str, router: ReplayTable, cascade: CascadeTable
    ) -> "ReplayResponse":
        """Turn the existing replay tables into structured rows."""
        all_big_cost = cascade.rows[0].cost_usd if cascade.rows else 0.0
        return cls(
            split=router.split,
            small_file=small_file,
            big_file=big_file,
            small_label=router.small_label,
            big_label=router.big_label,
            questions=router.questions,
            router_rows=[RouterReplayRowView.from_row(row) for row in router.rows],
            cascade_rows=[CascadeRowView.from_row(row, all_big_cost) for row in cascade.rows],
            small_tokens=cascade.small_tokens,
        )


class GroupCountView(BaseModel):
    """The size of one comparison group, split by difficulty."""

    count: int
    easy: int
    hard: int


class QuestionView(BaseModel):
    """One question in a comparison group."""

    id: str
    difficulty: str
    question: str
    expected: list[str]


class CompareResponse(BaseModel):
    """Body of GET /api/experiments/compare: the result of compare_runs()."""

    split: str
    a_file: str
    b_file: str
    a_label: str
    b_label: str
    compared: int
    only_in_a: int
    only_in_b: int
    score_a_percent: float
    score_b_percent: float
    both_passed: GroupCountView
    only_b_passed: GroupCountView
    only_a_passed: GroupCountView
    both_not_passed: GroupCountView
    only_b_passed_questions: list[QuestionView]
    only_a_passed_questions: list[QuestionView]
    both_not_passed_questions: list[QuestionView]

    @classmethod
    def from_report(
        cls, split: str, files: tuple[str, str], labels: tuple[str, str], report: ComparisonReport
    ) -> "CompareResponse":
        """Turn a comparison report into counts and question lists."""
        return cls(
            split=split,
            a_file=files[0],
            b_file=files[1],
            a_label=labels[0],
            b_label=labels[1],
            compared=report.compared,
            only_in_a=report.only_in_a,
            only_in_b=report.only_in_b,
            score_a_percent=report.score_a_percent,
            score_b_percent=report.score_b_percent,
            both_passed=group_counts(report.both_passed),
            only_b_passed=group_counts(report.only_b_passed),
            only_a_passed=group_counts(report.only_a_passed),
            both_not_passed=group_counts(report.both_not_passed),
            only_b_passed_questions=group_questions(report.only_b_passed),
            only_a_passed_questions=group_questions(report.only_a_passed),
            both_not_passed_questions=group_questions(report.both_not_passed),
        )


def percent_of(part: int, total: int) -> float:
    """Return part / total * 100, or 0.0 for an empty total."""
    return part / total * 100 if total else 0.0


def group_counts(group: ComparisonGroup) -> GroupCountView:
    """Return a group's count split by difficulty."""
    return GroupCountView(count=group.count, easy=group.easy, hard=group.hard)


def group_questions(group: ComparisonGroup) -> list[QuestionView]:
    """List the questions of a group, with the text cut to a short preview."""
    return [
        QuestionView(
            id=pair.a.id,
            difficulty=pair.a.difficulty,
            question=pair.a.question[:QUESTION_PREVIEW_CHARS],
            expected=pair.a.expected,
        )
        for pair in group.pairs
    ]
