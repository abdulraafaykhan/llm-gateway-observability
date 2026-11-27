"""Runs simulated demo traffic inside the gateway, through the normal chat path (fake mode only)."""

import asyncio
import logging
import random
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from app.models.completion import Message
from app.models.demo import DemoState, DemoStatus
from app.services.chat_service import AUTO_MODEL, ChatService
from app.services.experiments_service import UnknownFileError
from app.services.load_plan import SCENARIOS, build_schedule
from app.services.model_registry import UnknownModelError

logger = logging.getLogger(__name__)

ScheduleBuilder = Callable[[str, int], list[float]]

QUESTIONS_FILE_PATTERN = re.compile(r"^load_questions[A-Za-z0-9_.-]*\.txt$")
DEFAULT_QUESTIONS_FILE = "load_questions.txt"
MAX_PLANNED_REQUESTS: int = 1000
NOT_FAKE_MODE = "The gateway is not in fake mode."
NOT_FAKE_DATABASE = (
    "The database file name must contain 'fake' so simulated traffic never goes into a real "
    "database. Start the gateway with DATABASE_PATH=data/fake.db."
)


class DemoUnavailableError(Exception):
    """Raised when demos are not allowed in the gateway's current setup."""


class DemoBusyError(Exception):
    """Raised when a demo run is already in progress."""


class TooManyDemoRequestsError(Exception):
    """Raised when a scenario plans more requests than allowed."""


class DemoService:
    """Sends a scenario's requests on schedule (open-loop) through ChatService, one at a time."""

    def __init__(
        self,
        chat_service: ChatService,
        model_names: list[str],
        provider_mode: str,
        database_file: str,
        questions_dir: str,
        schedule_builder: ScheduleBuilder = build_schedule,
    ) -> None:
        self._chat_service: ChatService = chat_service
        self._model_names: list[str] = model_names
        self._provider_mode: str = provider_mode
        self._database_file: str = database_file
        self._questions_dir: Path = Path(questions_dir)
        self._schedule_builder: ScheduleBuilder = schedule_builder
        self._runner: asyncio.Task[None] | None = None
        self._requests: set[asyncio.Task[None]] = set()
        self._reset(state="idle", scenario=None, model=None, questions_file=None, seed=None)

    @property
    def database_file(self) -> str:
        """The file name of the database the gateway writes to."""
        return self._database_file

    def unavailable_reason(self) -> str | None:
        """Say why demos cannot run here, or None when they can."""
        if self._provider_mode != "fake":
            return NOT_FAKE_MODE
        if "fake" not in self._database_file.lower():
            return NOT_FAKE_DATABASE
        return None

    def scenarios(self) -> list[str]:
        """Return the scenario names."""
        return list(SCENARIOS)

    def model_choices(self) -> list[str]:
        """Return the model names a demo may ask for."""
        return [*self._model_names, AUTO_MODEL]

    def question_files(self) -> list[str]:
        """Return the question files directly in the questions folder."""
        if not self._questions_dir.is_dir():
            return []
        return sorted(
            path.name
            for path in self._questions_dir.iterdir()
            if path.is_file() and QUESTIONS_FILE_PATTERN.fullmatch(path.name)
        )

    def status(self) -> DemoStatus:
        """Return availability and the progress of the current or last run."""
        reason = self.unavailable_reason()
        end = time.monotonic() if self._state == "running" else self._ended
        elapsed = end - self._started if self._started is not None and end is not None else 0.0
        return DemoStatus(
            available=reason is None,
            reason=reason,
            state=self._state,
            scenario=self._scenario,
            model=self._model,
            questions_file=self._questions_file,
            seed=self._seed,
            planned=self._planned,
            sent=self._sent,
            completed=self._completed,
            ok=self._ok,
            errors=self._errors,
            started_at=self._started_at,
            elapsed_seconds=round(elapsed, 1),
        )

    def start(
        self, scenario: str, model: str | None, questions_file: str | None, seed: int
    ) -> DemoStatus:
        """Check the request, then start sending the scenario in the background."""
        reason = self.unavailable_reason()
        if reason is not None:
            raise DemoUnavailableError(reason)
        if self._state == "running":
            raise DemoBusyError("A demo run is already in progress. Stop it or wait for it to end.")
        if model is not None and model not in self.model_choices():
            raise UnknownModelError(model, self.model_choices())
        file_name = questions_file or DEFAULT_QUESTIONS_FILE
        questions = self._load_questions(file_name)
        schedule = self._schedule_builder(scenario, seed)
        if len(schedule) > MAX_PLANNED_REQUESTS:
            raise TooManyDemoRequestsError(
                f"The plan has {len(schedule)} requests; "
                f"at most {MAX_PLANNED_REQUESTS} are allowed."
            )
        self._reset(
            state="running", scenario=scenario, model=model, questions_file=file_name, seed=seed
        )
        self._planned = len(schedule)
        self._runner = asyncio.create_task(self._run(schedule, questions, seed))
        return self.status()

    async def stop(self) -> DemoStatus:
        """Cancel the pending sends and waiting requests of a running demo."""
        if self._state != "running":
            return self.status()
        tasks = [task for task in (self._runner, *self._requests) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._finish("stopped")
        return self.status()

    async def _run(self, schedule: list[float], questions: list[str], seed: int) -> None:
        """Send each request at its time as its own task, then wait for the rest."""
        picker = random.Random(seed)
        for send_at in schedule:
            delay = self._started + send_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            task = asyncio.create_task(self._send_one(picker.choice(questions)))
            self._requests.add(task)
            task.add_done_callback(self._requests.discard)
            self._sent += 1
        while self._requests:
            await asyncio.gather(*list(self._requests), return_exceptions=True)
        self._finish("finished")

    async def _send_one(self, question: str) -> None:
        """Send one question; any failure counts as an error and never stops the run."""
        try:
            await self._chat_service.chat([Message(role="user", content=question)], self._model)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # every failure is counted, never raised
            logger.warning("Demo request failed: %s", error)
            self._errors += 1
        else:
            self._ok += 1
        self._completed += 1

    def _load_questions(self, file_name: str) -> list[str]:
        """Read one question per non-blank line from a listed questions file."""
        names = self.question_files()
        if not QUESTIONS_FILE_PATTERN.fullmatch(file_name) or file_name not in names:
            raise UnknownFileError(file_name, names)
        lines = (self._questions_dir / file_name).read_text(encoding="utf-8").splitlines()
        questions = [line.strip() for line in lines if line.strip()]
        if not questions:
            raise UnknownFileError(file_name, names, f"File '{file_name}' has no questions.")
        return questions

    def _reset(
        self,
        state: DemoState,
        scenario: str | None,
        model: str | None,
        questions_file: str | None,
        seed: int | None,
    ) -> None:
        """Clear the counters and record what the new run (if any) is."""
        self._state: DemoState = state
        self._scenario: str | None = scenario
        self._model: str | None = model
        self._questions_file: str | None = questions_file
        self._seed: int | None = seed
        self._planned: int = 0
        self._sent: int = 0
        self._completed: int = 0
        self._ok: int = 0
        self._errors: int = 0
        running = state == "running"
        self._started: float | None = time.monotonic() if running else None
        self._ended: float | None = None
        self._started_at: str | None = datetime.now(UTC).isoformat() if running else None
        self._requests = set()

    def _finish(self, state: DemoState) -> None:
        """Mark the run as ended."""
        self._state = state
        self._ended = time.monotonic()
