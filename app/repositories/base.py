"""The storage interface for request records and their attempts."""

from abc import ABC, abstractmethod

from app.models.request_record import AttemptRecord, RequestRecord


class RequestRepository(ABC):
    """Interface for saving and reading request records."""

    @abstractmethod
    async def initialize(self) -> None:
        """Create the storage (tables and indexes) if it does not exist."""

    @abstractmethod
    async def add(self, record: RequestRecord, attempts: list[AttemptRecord]) -> int:
        """Save one request and its attempts together, and return the new request id."""

    @abstractmethod
    async def list_recent(self, limit: int) -> list[RequestRecord]:
        """Return up to limit records, newest first."""

    @abstractmethod
    async def list_all(self, since: str | None = None) -> list[RequestRecord]:
        """Return all records, or those with created_at >= since (UTC ISO), oldest id first."""

    @abstractmethod
    async def list_attempts(self, request_ids: list[int]) -> dict[int, list[AttemptRecord]]:
        """Return each request's attempts in order; requests without attempts are left out."""
