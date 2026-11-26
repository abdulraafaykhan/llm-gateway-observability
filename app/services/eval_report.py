"""Summarizes one eval run and compares two runs (pure functions)."""

import statistics

from app.models.eval import (
    ComparisonGroup,
    ComparisonReport,
    EvalOutcome,
    EvalRunResult,
    EvalSummary,
    OutcomePair,
    RoutingCounts,
    ScoreCount,
)
from app.services.stats_service import percentile

AUTO_MODEL: str = "auto"
DIFFICULTIES: tuple[str, ...] = ("easy", "hard")
SPLITS: tuple[str, ...] = ("tune", "test")


def score_count(outcomes: list[EvalOutcome]) -> ScoreCount:
    """Count passed out of total, with the percentage (0.0 when empty)."""
    passed = sum(1 for outcome in outcomes if outcome.status == "passed")
    total = len(outcomes)
    return ScoreCount(passed=passed, total=total, percent=passed / total * 100 if total else 0.0)


def summarize(run: EvalRunResult) -> EvalSummary:
    """Compute the score, counts, cost, and answered-only latency for one run."""
    outcomes = run.outcomes
    overall = score_count(outcomes)
    latencies = [
        outcome.latency_ms
        for outcome in outcomes
        if outcome.status != "error" and outcome.latency_ms is not None
    ]
    return EvalSummary(
        label=run.label,
        model_requested=run.model_requested,
        created_at=run.created_at,
        split_filter=run.split_filter,
        questions=overall.total,
        passed=overall.passed,
        failed=sum(1 for outcome in outcomes if outcome.status == "failed"),
        errors=sum(1 for outcome in outcomes if outcome.status == "error"),
        truncated=sum(1 for outcome in outcomes if outcome.truncated),
        score_percent=overall.percent,
        by_difficulty={
            name: score_count([o for o in outcomes if o.difficulty == name])
            for name in DIFFICULTIES
        },
        by_split={name: score_count([o for o in outcomes if o.split == name]) for name in SPLITS},
        total_cost_usd=sum(outcome.cost_usd for outcome in outcomes),
        latency_avg_ms=statistics.fmean(latencies) if latencies else None,
        latency_p50_ms=percentile(latencies, 50) if latencies else None,
        latency_p95_ms=percentile(latencies, 95) if latencies else None,
        failed_outcomes=[outcome for outcome in outcomes if outcome.status == "failed"],
        error_outcomes=[outcome for outcome in outcomes if outcome.status == "error"],
        routing=routing_counts(outcomes) if run.model_requested == AUTO_MODEL else None,
    )


def routing_counts(outcomes: list[EvalOutcome]) -> RoutingCounts:
    """Count answered outcomes by the model that answered, plus fallbacks (all outcomes)."""
    answered = [outcome for outcome in outcomes if outcome.status != "error"]
    return RoutingCounts(
        to_small=sum(1 for outcome in answered if outcome.answered_by == "small"),
        to_big=sum(1 for outcome in answered if outcome.answered_by == "big"),
        fallbacks=sum(1 for outcome in outcomes if outcome.fallback_used),
    )


def build_group(pairs: list[OutcomePair]) -> ComparisonGroup:
    """Wrap pairs in a group with counts by difficulty."""
    return ComparisonGroup(
        count=len(pairs),
        easy=sum(1 for pair in pairs if pair.a.difficulty == "easy"),
        hard=sum(1 for pair in pairs if pair.a.difficulty == "hard"),
        pairs=pairs,
    )


def compare_runs(run_a: EvalRunResult, run_b: EvalRunResult) -> ComparisonReport:
    """Group the shared questions by which run passed them (errors count as not passed)."""
    outcomes_b = {outcome.id: outcome for outcome in run_b.outcomes}
    ids_a = {outcome.id for outcome in run_a.outcomes}
    pairs = [
        OutcomePair(a=outcome, b=outcomes_b[outcome.id])
        for outcome in run_a.outcomes
        if outcome.id in outcomes_b
    ]
    groups: dict[tuple[bool, bool], list[OutcomePair]] = {
        (True, True): [],
        (True, False): [],
        (False, True): [],
        (False, False): [],
    }
    for pair in pairs:
        groups[(pair.a.status == "passed", pair.b.status == "passed")].append(pair)
    compared = len(pairs)
    return ComparisonReport(
        compared=compared,
        only_in_a=len(ids_a - outcomes_b.keys()),
        only_in_b=len(outcomes_b.keys() - ids_a),
        score_a_percent=score_count([pair.a for pair in pairs]).percent,
        score_b_percent=score_count([pair.b for pair in pairs]).percent,
        both_passed=build_group(groups[(True, True)]),
        only_a_passed=build_group(groups[(True, False)]),
        only_b_passed=build_group(groups[(False, True)]),
        both_not_passed=build_group(groups[(False, False)]),
    )
