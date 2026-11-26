"""Plain-text rendering of a stats report for the terminal."""

from app.models.stats import GroupStats, LatencyStats, StatsReport

PERCENTILE_NOTE = (
    "  - Percentiles use nearest-rank on answered requests only. "
    "Failed requests are counted separately."
)
SAMPLE_SIZE_NOTE = "  - p95 needs about 20+ answered requests and p99 about 100+ to mean anything."
SAMPLE_SIZE_THRESHOLD: int = 100


def render_stats_report(report: StatsReport) -> str:
    """Render the report as plain text, or a short message when there are no requests."""
    if report.overall.requests == 0:
        suffix = f" (since {report.since})" if report.since is not None else ""
        return f"No requests found.{suffix}"
    range_text = "all time" if report.since is None else f"since {report.since}"
    lines = [
        f"LLM Gateway stats  |  range: {range_text}  |  generated: {report.generated_at}",
        "",
        "OVERALL",
        *render_group_lines(report.overall, "  "),
        "",
        "BY MODEL",
    ]
    for name, group in report.by_model.items():
        lines += [f"  {name}", *render_group_lines(group, "    "), ""]
    lines += ["NOTES", PERCENTILE_NOTE]
    if report.overall.answered < SAMPLE_SIZE_THRESHOLD:
        lines.append(SAMPLE_SIZE_NOTE)
    return "\n".join(lines)


def render_group_lines(group: GroupStats, indent: str) -> list[str]:
    """Render the four lines shared by OVERALL and each model."""
    return [
        f"{indent}requests: {group.requests} | answered: {group.answered} "
        f"(truncated: {group.truncated}) | errors: {group.errors} "
        f"| error rate: {group.error_rate_percent:.1f}% | fallbacks: {group.fallbacks}",
        f"{indent}tokens in: {group.input_tokens} | tokens out: {group.output_tokens}",
        f"{indent}total cost: ${group.total_cost_usd:.6f} "
        f"| cost per 1,000 answered: ${group.cost_per_1000_answered_usd:.6f}",
        f"{indent}latency ms (answered only): {render_latency(group.latency)}",
    ]


def render_latency(latency: LatencyStats | None) -> str:
    """Render the latency summary, or a note when there are no answered requests."""
    if latency is None:
        return "no answered requests"
    return (
        f"avg {latency.avg_ms:.1f} | p50 {latency.p50_ms:.1f} | p95 {latency.p95_ms:.1f} "
        f"| p99 {latency.p99_ms:.1f} | max {latency.max_ms:.1f}"
    )
