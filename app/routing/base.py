"""The router interface."""

from abc import ABC, abstractmethod

from app.models.completion import Message
from app.models.routing import RouteDecision


class Router(ABC):
    """Interface for choosing which model should answer the messages."""

    @abstractmethod
    def choose(self, messages: list[Message]) -> RouteDecision:
        """Return the chosen model name with the reason."""

    def rule_names(self) -> list[str]:
        """Return the names of the rules this router scores with (none by default)."""
        return []
