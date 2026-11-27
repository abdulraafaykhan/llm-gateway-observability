"""Scoring rules the rule router adds up to decide between small and big."""

import re
from abc import ABC, abstractmethod

from app.models.routing import RuleHit

NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


class RoutingRule(ABC):
    """Interface for one scoring rule; `name` is the rule name used in its hits."""

    name: str = ""

    @abstractmethod
    def score(self, text: str) -> RuleHit | None:
        """Return a hit with points when the rule fires, or None."""


class LongQuestionRule(RoutingRule):
    """Adds 1 point for a medium-length question and 2 for a long one."""

    name = "long_question"

    def __init__(self, medium_words: int, long_words: int) -> None:
        self._medium_words: int = medium_words
        self._long_words: int = long_words

    def score(self, text: str) -> RuleHit | None:
        """Score the question by its word count."""
        word_count = len(text.split())
        if word_count > self._long_words:
            points = 2
        elif word_count > self._medium_words:
            points = 1
        else:
            return None
        return RuleHit(rule=self.name, points=points, detail=f"{word_count} words")


class KeywordsRule(RoutingRule):
    """Adds points once when any keyword appears (case-insensitive)."""

    name = "keywords"

    def __init__(self, points: int, keywords: list[str]) -> None:
        self._points: int = points
        self._keywords: list[str] = [keyword.lower() for keyword in keywords]

    def score(self, text: str) -> RuleHit | None:
        """Score the question by the first matching keyword."""
        lowered = text.lower()
        for keyword in self._keywords:
            if keyword in lowered:
                return RuleHit(rule=self.name, points=self._points, detail=keyword)
        return None


class ManyNumbersRule(RoutingRule):
    """Adds points when the question contains at least min_numbers numbers."""

    name = "many_numbers"

    def __init__(self, points: int, min_numbers: int) -> None:
        self._points: int = points
        self._min_numbers: int = min_numbers

    def score(self, text: str) -> RuleHit | None:
        """Score the question by how many numbers it contains."""
        count = len(NUMBER_PATTERN.findall(text))
        if count < self._min_numbers:
            return None
        return RuleHit(rule=self.name, points=self._points, detail=f"{count} numbers")


class MultipleQuestionsRule(RoutingRule):
    """Adds points when the text asks several questions (counted by question marks)."""

    name = "multiple_questions"

    def __init__(self, points: int, min_question_marks: int) -> None:
        self._points: int = points
        self._min_question_marks: int = min_question_marks

    def score(self, text: str) -> RuleHit | None:
        """Score the question by its question marks."""
        count = text.count("?")
        if count < self._min_question_marks:
            return None
        detail = f"{count} question marks"
        return RuleHit(rule=self.name, points=self._points, detail=detail)
