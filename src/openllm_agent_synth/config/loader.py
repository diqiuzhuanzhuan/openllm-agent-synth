"""Load YAML configuration files into validated application models."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import AppConfig


class ConfigError(ValueError):
    """Raised when a YAML config file is missing or invalid."""


def load_app_config(config_path: str | Path) -> AppConfig:
    """Read and validate a YAML configuration file."""

    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw_config, dict):
        raise ConfigError(f"Config file must define a YAML mapping at the top level: {path}")

    try:
        return AppConfig.model_validate(raw_config)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {path}:\n{exc}") from exc
