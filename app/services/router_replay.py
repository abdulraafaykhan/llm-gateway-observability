"""Replays saved small and big eval runs through the router, with no model calls (pure)."""

from app.models.completion import Message
from app.models.eval import EvalOutcome, EvalRunResult
from app.models.routing import ReplayRow, ReplayTable, RoutingConfig
from app.routing.base import Router
from app.routing.factory import build_rule_router

OutcomePairs = list[tuple[EvalOutcome, EvalOutcome]]


def shared_pairs(small_run: EvalRunResult, big_run: EvalRunResult, split: str) -> OutcomePairs:
    """Pair small and big outcomes for ids in both runs, keeping only the chosen split."""
    big_by_id = {outcome.id: outcome for outcome in big_run.outcomes}
    return [
        (outcome, big_by_id[outcome.id])
        for outcome in small_run.outcomes
        if outcome.id in big_by_id and split in ("all", outcome.split)
    ]


def needs_fallback(outcome: EvalOutcome) -> bool:
    """A saved small outcome triggers the fallback when it errored or was cut off."""
    return outcome.status == "error" or outcome.truncated


def tally(pairs: OutcomePairs, picks_big: list[bool], fallback: bool, label: str) -> ReplayRow:
    """Count routing, fallbacks, passes, and cost for one set of choices."""
    to_big = sum(picks_big)
    fallbacks = passed = 0
    cost_usd = 0.0
    for (small, big), use_big in zip(pairs, picks_big, strict=True):
        final = big if use_big else small
        if not use_big and fallback and needs_fallback(small):
            final = big
            fallbacks += 1
            cost_usd += small.cost_usd
        cost_usd += final.cost_usd
        passed += final.status == "passed"
    return ReplayRow(
        label=label,
        to_small=len(pairs) - to_big,
        to_big=to_big,
        fallbacks=fallbacks,
        passed=passed,
        total=len(pairs),
        cost_usd=cost_usd,
    )


def replay(
    small_run: EvalRunResult,
    big_run: EvalRunResult,
    router: Router,
    fallback: bool,
    split: str,
    label: str = "router",
) -> ReplayRow:
    """Score what the router would have achieved using the saved answers of each model."""
    pairs = shared_pairs(small_run, big_run, split)
    picks_big = [
        router.choose([Message(role="user", content=small.question)]).model_name
        == big_run.model_requested
        for small, _ in pairs
    ]
    return tally(pairs, picks_big, fallback, label)


def replay_baselines(
    small_run: EvalRunResult, big_run: EvalRunResult, split: str
) -> list[ReplayRow]:
    """Return the "all big" and "all small" rows (no fallback)."""
    pairs = shared_pairs(small_run, big_run, split)
    return [
        tally(pairs, [True] * len(pairs), False, "all big"),
        tally(pairs, [False] * len(pairs), False, "all small"),
    ]


def build_replay_table(
    small_run: EvalRunResult,
    big_run: EvalRunResult,
    config: RoutingConfig,
    thresholds: list[int],
    fallback: bool,
    split: str,
) -> ReplayTable:
    """Build the baseline rows plus one row per threshold."""
    rows = replay_baselines(small_run, big_run, split)
    for threshold in thresholds:
        router = build_rule_router(config, threshold=threshold)
        rows.append(replay(small_run, big_run, router, fallback, split, f"threshold {threshold}"))
    return ReplayTable(
        small_label=small_run.label,
        big_label=big_run.label,
        split=split,
        questions=len(shared_pairs(small_run, big_run, split)),
        fallback=fallback,
        rows=rows,
    )
