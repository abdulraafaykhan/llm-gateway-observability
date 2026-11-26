"""Loads and validates the labelled eval set from a JSONL file."""

import json
import re
from pathlib import Path

from pydantic import ValidationError

from app.models.eval import EvalQuestion

CHOICE_LETTER = re.compile(r"^[A-J]$")


def load_eval_set(path: str) -> list[EvalQuestion]:
    """Read one EvalQuestion per non-blank line, raising ValueError with the line number."""
    eval_path = Path(path)
    if not eval_path.is_file():
        raise FileNotFoundError(f"Eval set file not found: {eval_path}")
    questions: list[EvalQuestion] = []
    seen_ids: set[str] = set()
    lines = eval_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        question = _parse_line(line, line_number, eval_path)
        if question.id in seen_ids:
            raise ValueError(f"{eval_path} line {line_number}: repeated id '{question.id}'")
        seen_ids.add(question.id)
        questions.append(question)
    return questions


def _parse_line(line: str, line_number: int, eval_path: Path) -> EvalQuestion:
    """Parse and validate one JSONL line."""
    where = f"{eval_path} line {line_number}"
    try:
        data = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"{where}: invalid JSON ({error})") from error
    try:
        question = EvalQuestion.model_validate(data)
    except ValidationError as error:
        raise ValueError(f"{where}: missing or invalid field ({error})") from error
    _check_expected(question, where)
    return question


def _check_expected(question: EvalQuestion, where: str) -> None:
    """Make sure expected fits the check type."""
    if not question.expected:
        raise ValueError(f"{where}: expected must not be empty")
    if question.check == "choice":
        if len(question.expected) != 1 or not CHOICE_LETTER.fullmatch(question.expected[0]):
            raise ValueError(f"{where}: a choice check needs exactly one letter A to J")
        return
    if question.check != "number":
        return
    if len(question.expected) != 1:
        raise ValueError(f"{where}: a number check needs exactly one expected value")
    try:
        float(question.expected[0])
    except ValueError as error:
        message = f"{where}: expected '{question.expected[0]}' is not a number"
        raise ValueError(message) from error
