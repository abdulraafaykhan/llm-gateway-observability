"""Rule-based answer checking for the eval set (no AI judge)."""

import math
import re

from app.models.eval import EvalQuestion

DASH_CHARACTERS: str = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
SPECIAL_SPACES: str = "\u00a0\u202f\u2009"
DASH_AND_SPACE_TABLE: dict[int, str] = {
    **{ord(character): "-" for character in DASH_CHARACTERS},
    **{ord(character): " " for character in SPECIAL_SPACES},
    ord("*"): "",
    ord("`"): "",
}
THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
THOUSANDS_SPECIAL_SPACE = re.compile(rf"(?<=\d)[{SPECIAL_SPACES}](?=\d{{3}}(?!\d))")
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
ANSWER_LINE = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*)?[:\-]?\s*\(?([a-j])\)?[.\s]*$", re.MULTILINE
)
BARE_LETTER_LINE = re.compile(r"^\(?([a-j])\)?[.)]?$")


def normalize(text: str) -> str:
    """Lowercase, unify dashes and special spaces, and drop markdown * and ` characters."""
    return text.lower().translate(DASH_AND_SPACE_TABLE)


def clean_number_text(text: str) -> str:
    """Remove thousands separators (commas and special spaces), then normalize."""
    without_commas = THOUSANDS_COMMA.sub("", text)
    without_spaces = THOUSANDS_SPECIAL_SPACE.sub("", without_commas)
    return normalize(without_spaces)


def check_answer(question: EvalQuestion, answer: str) -> bool:
    """Grade an answer by the question's rule; the number check is lenient (any number counts)."""
    if question.check == "number":
        return _contains_number(answer, float(question.expected[0]))
    if question.check == "choice":
        return find_choice_letter(answer) == question.expected[0].lower()
    normalized_answer = normalize(answer)
    found = [normalize(expected) in normalized_answer for expected in question.expected]
    if question.check == "contains_all":
        return all(found)
    return any(found)


def find_choice_letter(answer: str) -> str | None:
    """choice: the last 'Answer: X' line, or a last line that is just the letter."""
    text = normalize(answer)
    matches = ANSWER_LINE.findall(text)
    if matches:
        return matches[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    bare = BARE_LETTER_LINE.fullmatch(lines[-1])
    return bare.group(1) if bare else None


def _contains_number(answer: str, expected: float) -> bool:
    """Return True if any number in the answer equals the expected value."""
    numbers = NUMBER_PATTERN.findall(clean_number_text(answer))
    return any(
        math.isclose(float(number), expected, rel_tol=1e-9, abs_tol=1e-6) for number in numbers
    )
