"""Data shape for the status of a simulated demo traffic run."""

from typing import Literal

from pydantic import BaseModel

DemoState = Literal["idle", "running", "finished", "stopped"]


class DemoStatus(BaseModel):
    """Whether demos can run, and the progress of the current or last run."""

    available: bool
    reason: str | None
    state: DemoState
    scenario: str | None
    model: str | None
    questions_file: str | None
    seed: int | None
    planned: int
    sent: int
    completed: int
    ok: int
    errors: int
    started_at: str | None
    elapsed_seconds: float
