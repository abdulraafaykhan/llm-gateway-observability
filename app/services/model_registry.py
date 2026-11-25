"""Loads model entries from config/models.toml and looks them up by name."""

import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.models.model_spec import ModelSpec


class UnknownModelError(Exception):
    """Raised when a model name is not in the registry."""

    def __init__(self, name: str, valid_names: list[str]) -> None:
        self.name: str = name
        self.valid_names: list[str] = valid_names
        super().__init__(f"Unknown model '{name}'. Valid names: {', '.join(valid_names)}")


class ModelRegistry:
    """Holds the configured models, keyed by their name (the TOML table key)."""

    def __init__(self, config_path: Path) -> None:
        self._specs: dict[str, ModelSpec] = _load_specs(config_path)

    def get(self, name: str) -> ModelSpec:
        """Return the model with this name, or raise UnknownModelError."""
        if name not in self._specs:
            raise UnknownModelError(name, self.names())
        return self._specs[name]

    def names(self) -> list[str]:
        """Return the names of all configured models."""
        return list(self._specs)


def _load_specs(config_path: Path) -> dict[str, ModelSpec]:
    """Read the TOML file and build a ModelSpec for each [models.<name>] table."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Models config file not found: {config_path}")
    try:
        with config_path.open("rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        message = f"Models config file is not valid TOML ({config_path}): {error}"
        raise ValueError(message) from error
    tables = data.get("models")
    if not isinstance(tables, dict) or not tables:
        raise ValueError(f"Models config file has no [models.<name>] tables: {config_path}")
    return {name: _build_spec(name, table, config_path) for name, table in tables.items()}


def _build_spec(name: str, table: Any, config_path: Path) -> ModelSpec:
    """Build one ModelSpec, raising a clear error if the table is malformed."""
    if not isinstance(table, dict):
        raise ValueError(f"[models.{name}] in {config_path} must be a table")
    try:
        return ModelSpec(name=name, **table)
    except ValidationError as error:
        raise ValueError(f"Invalid entry [models.{name}] in {config_path}: {error}") from error
