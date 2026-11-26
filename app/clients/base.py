"""The interface for asking the gateway a question."""

from abc import ABC, abstractmethod

from app.models.eval import GatewayReply


class GatewayClient(ABC):
    """Interface for sending one question to the gateway and getting its reply."""

    @abstractmethod
    async def ask(self, question: str, model: str) -> GatewayReply:
        """Send the question to the given model and return the reply; never raise for failures."""
