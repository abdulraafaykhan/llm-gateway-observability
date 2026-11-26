"""Plain-text rendering of the router replay table."""

from app.models.routing import ReplayRow, ReplayTable

NOTES: tuple[str, ...] = (
    "  - Uses the saved eval results only; no model calls were made.",
    "  - Cost of a failed small attempt is only counted if the saved run recorded it.",
    '  - Tune the threshold on the "tune" split, then report the number from the "test" split.',
)


def render_replay_table(table: ReplayTable) -> str:
    """Render the header, one aligned row per setting, and the notes."""
    fallback = "on" if table.fallback else "off"
    lines = [
        f"ROUTER REPLAY  |  small: {table.small_label}  |  big: {table.big_label}  |  "
        f"split: {table.split}  |  questions: {table.questions}  |  fallback: {fallback}",
        "",
        f"  {'setting':<14}{'to small':>9}{'to big':>8}{'fallbacks':>11}"
        f"{'passed':>8}{'score':>9}  {'cost'}",
        *(render_row(row) for row in table.rows),
        "",
        "NOTES",
        *NOTES,
    ]
    return "\n".join(lines)


def render_row(row: ReplayRow) -> str:
    """Render one setting's counts, score, and cost in aligned columns."""
    score = row.passed / row.total * 100 if row.total else 0.0
    return (
        f"  {row.label:<14}{row.to_small:>9}{row.to_big:>8}{row.fallbacks:>11}"
        f"{row.passed:>8}{score:>8.1f}%  ${row.cost_usd:.6f}"
    )
