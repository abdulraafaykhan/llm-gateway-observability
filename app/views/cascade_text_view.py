"""Plain-text rendering of token-cascade replay tables."""

from app.models.cascade import CascadeRow, CascadeTable, TokenStats

NOTES: tuple[str, ...] = (
    "  - This replays saved eval runs only. No model calls were made.",
    "  - Escalated questions pay for both models and take small's time plus big's time.",
    "  - A failed small attempt costs only what the saved run recorded "
    "(older runs recorded 0 for errors).",
    "  - A difference of 1 or 2 questions is within normal run-to-run noise at this size.",
    '  - Decision rule: build a token cascade into the gateway only if one limit beats "fallback '
    'only" by at least 2 passed questions on BOTH the tune and test splits and also costs less '
    'than "all big".',
)


def render_cascade_tables(tables: list[CascadeTable]) -> str:
    """Render one block per table, separated by blank lines, then the notes once."""
    blocks = [render_table(table) for table in tables]
    return "\n\n".join(blocks) + "\n\nNOTES\n" + "\n".join(NOTES)


def render_table(table: CascadeTable) -> str:
    """Render one table's header, rows, and token line."""
    if table.questions == 0:
        return f"CASCADE REPLAY  |  split: {table.split}  |  no questions found in both runs."
    all_big_cost = table.rows[0].cost_usd
    lines = [
        f"CASCADE REPLAY  |  small: {table.small_label}  |  big: {table.big_label}  |  "
        f"split: {table.split}  |  questions: {table.questions}",
        "",
        f"  {'setting':<16}{'kept small':>11}{'escalated':>11}{'passed':>8}{'score':>9}"
        f"{'rescued':>9}{'lost':>6}  {'cost':<11}{'cost vs all big'}",
        *(render_row(row, all_big_cost) for row in table.rows),
        "",
        render_token_line(table.small_tokens),
    ]
    return "\n".join(lines)


def render_row(row: CascadeRow, all_big_cost: float) -> str:
    """Render one row in aligned columns."""
    score = row.passed / row.total * 100 if row.total else 0.0
    relative = f"{round(row.cost_usd / all_big_cost * 100)}%" if all_big_cost else "n/a"
    cost = f"${row.cost_usd:.6f}"
    return (
        f"  {row.label:<16}{row.kept_small:>11}{row.escalated:>11}{row.passed:>8}"
        f"{score:>8.1f}%{row.rescued:>9}{row.lost:>6}  {cost:<11}{relative:>4}"
    )


def render_token_line(stats: TokenStats | None) -> str:
    """Render small's output-token spread, or a note when nothing was answered."""
    if stats is None:
        return "SMALL'S OUTPUT TOKENS (answered questions): none"
    return (
        f"SMALL'S OUTPUT TOKENS (answered questions): p50 {stats.p50:.0f} | "
        f"p75 {stats.p75:.0f} | p90 {stats.p90:.0f} | max {stats.max:.0f}"
    )
