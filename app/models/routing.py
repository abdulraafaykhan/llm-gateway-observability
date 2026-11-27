"""Data shapes for routing decisions, routing settings, and router replay results."""

from pydantic import BaseModel


class RuleHit(BaseModel):
    """One routing rule that fired, with the points it added."""

    rule: str
    points: int
    detail: str = ""


class RouteDecision(BaseModel):
    """Which model the router picked, and why."""

    model_name: str
    score: int
    threshold: int
    hits: list[RuleHit]
    reason: str


class RoutingConfig(BaseModel):
    """Router and fallback settings plus the settings for each scoring rule."""

    small_model: str
    big_model: str
    threshold: int
    fallback_enabled: bool
    fallback_model: str
    fallback_on_truncated: bool = False
    medium_words: int
    long_words: int
    keywords_points: int
    keywords: list[str]
    many_numbers_points: int
    min_numbers: int
    multiple_questions_points: int
    min_question_marks: int


class ReplayRow(BaseModel):
    """What one routing setting would have scored and cost on saved eval results."""

    label: str
    to_small: int
    to_big: int
    fallbacks: int
    passed: int
    total: int
    cost_usd: float


class ReplayTable(BaseModel):
    """Baseline rows plus one row per threshold for a router replay."""

    small_label: str
    big_label: str
    split: str
    questions: int
    fallback: bool
    rows: list[ReplayRow]
