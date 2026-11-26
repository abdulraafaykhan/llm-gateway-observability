"""Data shapes for the offline token-cascade replay."""

from pydantic import BaseModel


class CascadeRow(BaseModel):
    """What one setting (a baseline or a token limit) would have scored and cost."""

    label: str
    limit: int | None
    kept_small: int
    escalated: int
    passed: int
    total: int
    rescued: int
    lost: int
    cost_usd: float


class TokenStats(BaseModel):
    """Spread of small's output tokens on answered questions."""

    p50: float
    p75: float
    p90: float
    max: float


class CascadeTable(BaseModel):
    """Baseline and cascade rows for one split."""

    split: str
    questions: int
    small_label: str
    big_label: str
    rows: list[CascadeRow]
    small_tokens: TokenStats | None
