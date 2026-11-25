"""SQLite implementation of the request repository."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from app.models.request_record import RequestRecord
from app.repositories.base import RequestRepository

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    status TEXT NOT NULL,
    error_type TEXT,
    error_message TEXT,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    cost_usd REAL NOT NULL
)
"""

CREATE_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests (created_at)"

INSERT_SQL = """
INSERT INTO requests (
    created_at, model_name, model_id, question, answer, status, error_type,
    error_message, input_tokens, output_tokens, latency_ms, cost_usd
) VALUES (
    :created_at, :model_name, :model_id, :question, :answer, :status, :error_type,
    :error_message, :input_tokens, :output_tokens, :latency_ms, :cost_usd
)
"""

SELECT_RECENT_SQL = "SELECT * FROM requests ORDER BY created_at DESC, id DESC LIMIT ?"

SELECT_ALL_SQL = "SELECT * FROM requests ORDER BY id ASC"

SELECT_SINCE_SQL = "SELECT * FROM requests WHERE created_at >= ? ORDER BY id ASC"


class SqliteRequestRepository(RequestRepository):
    """Stores request records in a SQLite file, one short-lived connection per call."""

    def __init__(self, database_path: Path) -> None:
        self._database_path: Path = database_path

    async def initialize(self) -> None:
        """Create the database folder, enable WAL, and create the table and index."""
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as connection:
            await connection.execute("PRAGMA journal_mode=WAL")
            await connection.execute(CREATE_TABLE_SQL)
            await connection.execute(CREATE_INDEX_SQL)
            await connection.commit()

    async def add(self, record: RequestRecord) -> None:
        """Insert one record."""
        async with self._connect() as connection:
            await connection.execute(INSERT_SQL, record.model_dump(exclude={"id"}))
            await connection.commit()

    async def list_recent(self, limit: int) -> list[RequestRecord]:
        """Return up to limit records, newest first."""
        return await self._fetch_records(SELECT_RECENT_SQL, (limit,))

    async def list_all(self, since: str | None = None) -> list[RequestRecord]:
        """Return all records (or those with created_at >= since), oldest id first."""
        if since is None:
            return await self._fetch_records(SELECT_ALL_SQL, ())
        return await self._fetch_records(SELECT_SINCE_SQL, (since,))

    async def _fetch_records(
        self, sql: str, parameters: tuple[str | int, ...]
    ) -> list[RequestRecord]:
        """Run a parameterized SELECT and turn each row into a RequestRecord."""
        async with self._connect() as connection:
            connection.row_factory = aiosqlite.Row
            async with connection.execute(sql, parameters) as cursor:
                rows = await cursor.fetchall()
        return [RequestRecord(**dict(row)) for row in rows]

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """Open a connection with a busy timeout, and close it afterwards."""
        async with aiosqlite.connect(self._database_path) as connection:
            await connection.execute("PRAGMA busy_timeout=5000")
            yield connection
