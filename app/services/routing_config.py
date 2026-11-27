"""Loads and validates the routing settings from config/routing.toml."""

import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.models.routing import RoutingConfig
from app.services.model_registry import ModelRegistry

# RoutingConfig field -> (TOML table, key in that table)
CONFIG_KEYS: dict[str, tuple[str, str]] = {
    "small_model": ("router", "small_model"),
    "big_model": ("router", "big_model"),
    "threshold": ("router", "threshold"),
    "fallback_enabled": ("router", "fallback_enabled"),
    "fallback_model": ("router", "fallback_model"),
    "medium_words": ("rules.long_question", "medium_words"),
    "long_words": ("rules.long_question", "long_words"),
    "keywords_points": ("rules.keywords", "points"),
    "keywords": ("rules.keywords", "words"),
    "many_numbers_points": ("rules.many_numbers", "points"),
    "min_numbers": ("rules.many_numbers", "min_numbers"),
    "multiple_questions_points": ("rules.multiple_questions", "points"),
    "min_question_marks": ("rules.multiple_questions", "min_question_marks"),
}
# Optional keys: RoutingConfig field -> (TOML table, key); a missing key keeps the model default.
OPTIONAL_KEYS: dict[str, tuple[str, str]] = {
    "fallback_on_truncated": ("router", "fallback_on_truncated"),
}
MODEL_SETTINGS: tuple[str, ...] = ("small_model", "big_model", "fallback_model")


def load_routing_config(path: str) -> RoutingConfig:
    """Read the routing TOML, raising a clear error if it is missing, malformed, or incomplete."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Routing config file not found: {config_path}")
    try:
        with config_path.open("rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        message = f"Routing config file is not valid TOML ({config_path}): {error}"
        raise ValueError(message) from error
    values = {
        field: _read_key(data, table, key, config_path)
        for field, (table, key) in CONFIG_KEYS.items()
    }
    for field, (table, key) in OPTIONAL_KEYS.items():
        section = data.get(table)
        if isinstance(section, dict) and key in section:
            values[field] = section[key]
    try:
        return RoutingConfig(**values)
    except ValidationError as error:
        raise ValueError(f"Invalid routing config {config_path}: {error}") from error


def _read_key(data: dict[str, Any], table: str, key: str, config_path: Path) -> Any:
    """Return data[table][key], raising a clear error when the table or key is missing."""
    current: Any = data
    for part in table.split("."):
        current = current.get(part) if isinstance(current, dict) else None
    if not isinstance(current, dict) or key not in current:
        raise ValueError(f"Routing config {config_path} is missing '{key}' in [{table}]")
    return current[key]


def validate_routing_models(config: RoutingConfig, registry: ModelRegistry) -> None:
    """Raise a clear error if a model named in the routing config is not in the registry."""
    names = registry.names()
    for setting in MODEL_SETTINGS:
        value = getattr(config, setting)
        if value not in names:
            raise ValueError(
                f"Routing config {setting} '{value}' is not a configured model. "
                f"Valid names: {', '.join(names)}"
            )
