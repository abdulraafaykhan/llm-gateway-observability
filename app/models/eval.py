"""Data shapes for the labelled eval set, eval runs, summaries, and comparisons."""

from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints

Difficulty = Literal["easy", "hard"]
Category = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Split = Literal["tune", "test"]
CheckType = Literal["contains_all", "contains_any", "number", "choice"]
OutcomeStatus = Literal["passed", "failed", "error"]


class EvalQuestion(BaseModel):
    """One labelled question with a rule for checking the answer."""

    id: str
    question: str
    difficulty: Difficulty
    category: Category
    split: Split
    check: CheckType
    expected: list[str]


class GatewayReply(BaseModel):
    """What the gateway returned for one question; http_status 0 means connection or timeout."""

    http_status: int
    answer: str | None
    model: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float | None
    truncated: bool
    error_message: str | None
    cost_usd: float = 0.0
    route_reason: str = ""
    fallback_used: bool = False


class EvalOutcome(BaseModel):
    """The graded result for one question."""

    id: str
    difficulty: Difficulty
    category: Category
    split: Split
    question: str
    check: CheckType
    expected: list[str]
    status: OutcomeStatus
    truncated: bool
    answer: str | None
    answered_by: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float | None
    cost_usd: float
    error_message: str | None
    attempts: int
    route_reason: str = ""
    fallback_used: bool = False


class EvalRunResult(BaseModel):
    """A full eval run against one model, as saved to results/."""

    label: str
    model_requested: str
    created_at: str
    split_filter: Literal["all", "tune", "test"]
    eval_set_path: str
    outcomes: list[EvalOutcome]


class ScoreCount(BaseModel):
    """Passed out of total for one slice of questions."""

    passed: int
    total: int
    percent: float


class RoutingCounts(BaseModel):
    """Where an "auto" run's answered questions went, and how many fell back."""

    to_small: int
    to_big: int
    fallbacks: int


class EvalSummary(BaseModel):
    """Score, counts, cost, and latency for one eval run."""

    label: str
    model_requested: str
    created_at: str
    split_filter: str
    questions: int
    passed: int
    failed: int
    errors: int
    truncated: int
    score_percent: float
    by_difficulty: dict[str, ScoreCount]
    by_split: dict[str, ScoreCount]
    total_cost_usd: float
    latency_avg_ms: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    failed_outcomes: list[EvalOutcome]
    error_outcomes: list[EvalOutcome]
    routing: RoutingCounts | None = None


class OutcomePair(BaseModel):
    """The outcomes of the same question in run A and run B."""

    a: EvalOutcome
    b: EvalOutcome


class ComparisonGroup(BaseModel):
    """Questions that fall in one pass/fail group, with counts by difficulty."""

    count: int
    easy: int
    hard: int
    pairs: list[OutcomePair]


class ComparisonReport(BaseModel):
    """How two runs did on the questions they share."""

    compared: int
    only_in_a: int
    only_in_b: int
    score_a_percent: float
    score_b_percent: float
    both_passed: ComparisonGroup
    only_a_passed: ComparisonGroup
    only_b_passed: ComparisonGroup
    both_not_passed: ComparisonGroup
