"""JSON shapes for the demo traffic controls."""

from typing import Literal

from pydantic import BaseModel

from app.models.demo import DemoStatus


class DemoStartRequest(BaseModel):
    """Body of POST /api/demo/start."""

    scenario: Literal["smoke", "normal", "spike"]
    model: str | None = None
    questions_file: str | None = None
    seed: int = 42


class DemoStatusResponse(DemoStatus):
    """Body returned by the demo status, start, and stop endpoints."""

    @classmethod
    def from_status(cls, status: DemoStatus) -> "DemoStatusResponse":
        """Build the response from the service's status."""
        return cls(**status.model_dump())


class DemoOptionsResponse(BaseModel):
    """Body of GET /api/demo/options: what the demo form can offer."""

    scenarios: list[str]
    models: list[str]
    questions_files: list[str]
    database_file: str
