"""Plain-text rendering of eval summaries, comparisons, and the eval set overview."""

from collections import Counter

from app.models.eval import (
    ComparisonGroup,
    ComparisonReport,
    EvalOutcome,
    EvalQuestion,
    EvalSummary,
    ScoreCount,
)

SNIPPET_LENGTH: int = 80


def render_eval_summary(summary: EvalSummary, saved_path: str) -> str:
    """Render one run's score, counts, cost, latency, failures, and errors."""
    lines = [
        f"EVAL RESULT  |  label: {summary.label}  |  requested model: {summary.model_requested}"
        f"  |  split: {summary.split_filter}  |  {summary.created_at}",
        "",
        f"  questions: {summary.questions} | passed: {summary.passed} | failed: {summary.failed}"
        f" | errors: {summary.errors} | truncated (counted as failed): {summary.truncated}",
        f"  score: {summary.score_percent:.1f}% (passed / questions)",
        *render_routing_lines(summary),
        "  "
        + " | ".join(
            f"{name}: {render_score(score)}" for name, score in summary.by_difficulty.items()
        ),
        "  " + render_split_line(summary),
        f"  cost: ${summary.total_cost_usd:.6f} | latency ms (answered): {render_latency(summary)}",
        f"  saved to: {saved_path}",
    ]
    if summary.failed_outcomes:
        lines += ["", "FAILED QUESTIONS", *map(render_failed_line, summary.failed_outcomes)]
    if summary.error_outcomes:
        lines += ["", "ERRORS", *map(render_error_line, summary.error_outcomes)]
    return "\n".join(lines)


def render_routing_lines(summary: EvalSummary) -> list[str]:
    """Render the routing line for an "auto" run, or nothing for other runs."""
    routing = summary.routing
    if routing is None:
        return []
    return [
        f"  routing: to small {routing.to_small} | to big {routing.to_big} "
        f"| fallbacks {routing.fallbacks}"
    ]


def render_score(score: ScoreCount) -> str:
    """Render passed/total with the percentage, or n/a when the group has no questions."""
    if score.total == 0:
        return "n/a"
    return f"{score.passed}/{score.total} ({score.percent:.1f}%)"


def render_split_line(summary: EvalSummary) -> str:
    """Render the split scores, or only the filtered split when a filter was used."""
    names = list(summary.by_split) if summary.split_filter == "all" else [summary.split_filter]
    return " | ".join(f"{name}: {render_score(summary.by_split[name])}" for name in names)


def render_latency(summary: EvalSummary) -> str:
    """Render avg, p50, and p95 latency, or a note when nothing was answered."""
    if summary.latency_avg_ms is None:
        return "no answered questions"
    return (
        f"avg {summary.latency_avg_ms:.1f} | p50 {summary.latency_p50_ms:.1f} "
        f"| p95 {summary.latency_p95_ms:.1f}"
    )


def render_expected(outcome: EvalOutcome) -> str:
    """Join the expected strings with ' / '."""
    return " / ".join(outcome.expected)


def snippet(text: str | None) -> str:
    """Return the first characters of a text on one line."""
    return (text or "").replace("\r", " ").replace("\n", " ")[:SNIPPET_LENGTH]


def render_failed_line(outcome: EvalOutcome) -> str:
    """Render one failed question with what was expected and what came back."""
    cut_off = " (truncated)" if outcome.truncated else ""
    return (
        f"  {outcome.id} [{outcome.difficulty}] expected: {render_expected(outcome)} "
        f'| got: "{snippet(outcome.answer)}"{cut_off}'
    )


def render_error_line(outcome: EvalOutcome) -> str:
    """Render one errored question with its error message."""
    return f"  {outcome.id} [{outcome.difficulty}] {outcome.error_message}"


def render_comparison(report: ComparisonReport, label_a: str, label_b: str) -> str:
    """Render the four pass/fail groups, the scores, and the lists worth reviewing."""
    skipped = ""
    if report.only_in_a or report.only_in_b:
        skipped = f" (skipped: {report.only_in_a} only in A, {report.only_in_b} only in B)"
    lines = [
        f"EVAL COMPARISON  |  A: {label_a}  |  B: {label_b}  |  "
        f"compared: {report.compared} questions{skipped}",
        "",
        f"  both passed: {render_group_counts(report.both_passed)}",
        f"  only B passed: {render_group_counts(report.only_b_passed)}",
        f"  only A passed: {render_group_counts(report.only_a_passed)}",
        f"  neither passed: {render_group_counts(report.both_not_passed)}",
        f"  score A: {report.score_a_percent:.1f}% | score B: {report.score_b_percent:.1f}%",
    ]
    if report.only_b_passed.pairs:
        heading = f"ONLY {label_b} PASSED  (these are the questions {label_a} needs help with)"
        lines += ["", heading, *render_question_lines(report.only_b_passed)]
    if report.both_not_passed.pairs:
        heading = "NEITHER PASSED  (check if the answer key is wrong)"
        lines += ["", heading, *render_question_lines(report.both_not_passed)]
    return "\n".join(lines)


def render_group_counts(group: ComparisonGroup) -> str:
    """Render a group's count with the easy/hard split."""
    return f"{group.count} (easy {group.easy}, hard {group.hard})"


def render_question_lines(group: ComparisonGroup) -> list[str]:
    """Render one line per question in a comparison group."""
    return [
        f"  {pair.a.id} [{pair.a.difficulty}] {snippet(pair.a.question)} "
        f"| expected: {render_expected(pair.a)}"
        for pair in group.pairs
    ]


def render_eval_set_overview(questions: list[EvalQuestion], path: str) -> str:
    """Render the question count, a difficulty x split table, and category counts."""
    cells = Counter((question.difficulty, question.split) for question in questions)
    categories = Counter(question.category for question in questions)
    lines = [
        f"eval set: {path} | questions: {len(questions)}",
        "",
        f"  {'':6}{'tune':>6}{'test':>6}",
        *(
            f"  {difficulty:6}{cells[(difficulty, 'tune')]:>6}{cells[(difficulty, 'test')]:>6}"
            for difficulty in ("easy", "hard")
        ),
        "",
        "  categories: " + " | ".join(f"{name} {categories[name]}" for name in sorted(categories)),
    ]
    return "\n".join(lines)
