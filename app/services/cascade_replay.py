"""Replays saved small and big eval runs as a token cascade, with no model calls (pure)."""

from app.models.cascade import CascadeRow, CascadeTable, TokenStats
from app.models.eval import EvalOutcome, EvalRunResult
from app.services.router_replay import OutcomePairs, shared_pairs
from app.services.stats_service import percentile


def escalate(small: EvalOutcome, limit: int | None) -> bool:
    """Escalate when small errored, was cut off, or used more than `limit` output tokens."""
    if small.status == "error" or small.truncated:
        return True
    return limit is not None and small.output_tokens > limit


def _row(
    pairs: OutcomePairs, label: str, limit: int | None, escalated_flags: list[bool], small_ran: bool
) -> CascadeRow:
    """Count kept/escalated, passes, rescued, lost, and cost for one set of decisions."""
    passed = rescued = lost = 0
    cost_usd = 0.0
    for (small, big), escalated in zip(pairs, escalated_flags, strict=True):
        final = big if escalated else small
        cost_usd += final.cost_usd
        if escalated and small_ran:
            cost_usd += small.cost_usd
        final_passed = final.status == "passed"
        small_passed = small.status == "passed"
        passed += final_passed
        rescued += final_passed and not small_passed
        lost += small_passed and not final_passed
    escalated_count = sum(escalated_flags)
    return CascadeRow(
        label=label,
        limit=limit,
        kept_small=len(pairs) - escalated_count,
        escalated=escalated_count,
        passed=passed,
        total=len(pairs),
        rescued=rescued,
        lost=lost,
        cost_usd=cost_usd,
    )


def cascade_row(pairs: OutcomePairs, label: str, limit: int | None) -> CascadeRow:
    """Small answers first; escalated questions pay for both models."""
    flags = [escalate(small, limit) for small, _ in pairs]
    return _row(pairs, label, limit, flags, small_ran=True)


def small_token_stats(pairs: OutcomePairs) -> TokenStats | None:
    """Nearest-rank p50/p75/p90 and max of small's output tokens, skipping errors."""
    tokens = [float(small.output_tokens) for small, _ in pairs if small.status != "error"]
    if not tokens:
        return None
    return TokenStats(
        p50=percentile(tokens, 50),
        p75=percentile(tokens, 75),
        p90=percentile(tokens, 90),
        max=max(tokens),
    )


def build_cascade_table(
    small_run: EvalRunResult, big_run: EvalRunResult, split: str, limits: list[int]
) -> CascadeTable:
    """Build the all-big, all-small, fallback-only, and one-per-limit rows for a split."""
    pairs = shared_pairs(small_run, big_run, split)
    rows: list[CascadeRow] = []
    if pairs:
        rows = [
            _row(pairs, "all big", None, [True] * len(pairs), small_ran=False),
            _row(pairs, "all small", None, [False] * len(pairs), small_ran=False),
            cascade_row(pairs, "fallback only", None),
            *(cascade_row(pairs, f"tokens > {limit}", limit) for limit in limits),
        ]
    return CascadeTable(
        split=split,
        questions=len(pairs),
        small_label=small_run.label,
        big_label=big_run.label,
        rows=rows,
        small_tokens=small_token_stats(pairs),
    )
