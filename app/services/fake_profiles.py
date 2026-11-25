"""Loads fake-mode model profiles from config/fake.toml."""

import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.models.fake_profile import FakeProfile


def load_fake_profiles(path: str) -> dict[str, FakeProfile]:
    """Read the TOML file and build a FakeProfile for each [fake.<model name>] table."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Fake config file not found: {config_path}")
    try:
        with config_path.open("rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        message = f"Fake config file is not valid TOML ({config_path}): {error}"
        raise ValueError(message) from error
    tables = data.get("fake")
    if not isinstance(tables, dict) or not tables:
        raise ValueError(f"Fake config file has no [fake.<name>] tables: {config_path}")
    return {name: _build_profile(name, table, config_path) for name, table in tables.items()}


def _build_profile(name: str, table: Any, config_path: Path) -> FakeProfile:
    """Build one FakeProfile, raising a clear error if the table is malformed."""
    if not isinstance(table, dict):
        raise ValueError(f"[fake.{name}] in {config_path} must be a table")
    try:
        return FakeProfile(**table)
    except ValidationError as error:
        raise ValueError(f"Invalid entry [fake.{name}] in {config_path}: {error}") from error
