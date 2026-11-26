"""The router interface."""

from abc import ABC, abstractmethod

from app.models.completion import Message
from app.models.routing import RouteDecision


class Router(ABC):
    """Interface for choosing which model should answer the messages."""

    @abstractmethod
    def choose(self, messages: list[Message]) -> RouteDecision:
        """Return the chosen model name with the reason."""
