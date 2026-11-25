"""The storage interface for request records."""

from abc import ABC, abstractmethod

from app.models.request_record import RequestRecord


class RequestRepository(ABC):
    """Interface for saving and reading request records."""

    @abstractmethod
    async def initialize(self) -> None:
        """Create the storage (table and index) if it does not exist."""

    @abstractmethod
    async def add(self, record: RequestRecord) -> None:
        """Save one request record."""

    @abstractmethod
    async def list_recent(self, limit: int) -> list[RequestRecord]:
        """Return up to limit records, newest first."""

    @abstractmethod
    async def list_all(self, since: str | None = None) -> list[RequestRecord]:
        """Return all records, or those with created_at >= since (UTC ISO), oldest id first."""
