"""Sorts a failed request's error message into a small set of failure types (pure)."""

FAILURE_TYPES: tuple[str, ...] = ("empty_answer", "rate_limit", "timeout", "auth", "other")

# Checked in order; the first matching group wins.
FAILURE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("empty_answer", ("empty answer",)),
    ("rate_limit", ("rate limit", "rate_limit", "429", "too many requests")),
    ("timeout", ("timeout", "timed out")),
    ("auth", ("api key", "unauthorized", "401", "403", "authentication")),
)


def classify_failure(error_message: str | None) -> str:
    """These patterns are based on the current provider error messages."""
    text = (error_message or "").lower()
    for failure_type, patterns in FAILURE_PATTERNS:
        if any(pattern in text for pattern in patterns):
            return failure_type
    return "other"
